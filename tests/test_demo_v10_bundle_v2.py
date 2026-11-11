from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from demo.backend.v10_artifacts import (
    BACKBONE_SHA256,
    EXPECTED_RECIPE,
    SCHEMA_V2,
    TASK_CLASSES,
    V2_RECIPE,
    V2_REQUIRED_FILES,
    V10ArtifactError,
)
from demo.backend.v10_service import MoCoModelService

PREPROCESS_SOURCE = (
    "import numpy as np\n"
    "def preprocess_image(image, size=224):\n"
    "    return np.zeros((1, 3, size, size), dtype=np.float32)\n"
).encode()


class V2Session:
    """Stand-in for the v2 ONNX graph: both heads, D4 TTA already folded in."""

    def __init__(self):
        self.feed_shapes = []

    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values")]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in ("ham7_probabilities", "pad6_probabilities")]

    def run(self, output_names, feed):
        self.feed_shapes.append(feed["pixel_values"].shape)
        ham = np.array([[0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1]], dtype=np.float32)
        pad = np.array([[0.05, 0.05, 0.6, 0.1, 0.1, 0.1]], dtype=np.float32)
        return [ham, pad]


def _payloads(drop=()):
    payloads = {name: b"synthetic release asset" for name in V2_REQUIRED_FILES}
    payloads["model.onnx"] = b"synthetic onnx"
    payloads["preprocess.py"] = PREPROCESS_SOURCE
    payloads["code/dermssl/release/moco_preprocess.py"] = PREPROCESS_SOURCE
    for name in drop:
        payloads.pop(name)
    return payloads


def _bundle(tmp_path, *, schema=SCHEMA_V2, input_size=448, recipe=None, drop=()):
    files = {}
    for name, payload in _payloads(drop).items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "schema": schema,
        "release": "v10.1.0",
        "seed": 42,
        "backbone_sha256": BACKBONE_SHA256,
        "tasks": {
            "HAM": {"classes": TASK_CLASSES["MOCO_HAM7"], "output_name": "ham7_probabilities", "model_key": "MOCO_HAM7"},
            "PAD": {"classes": TASK_CLASSES["MOCO_PAD6"], "output_name": "pad6_probabilities", "model_key": "MOCO_PAD6"},
        },
        "recipe": V2_RECIPE if recipe is None else recipe,
        "feature_dim": 3072,
        "input": {"name": "pixel_values", "dtype": "float32", "shape": ["batch", 3, input_size, input_size]},
        "files": files,
    }
    content = json.dumps(manifest).encode("utf-8")
    (tmp_path / "manifest.json").write_bytes(content)
    return hashlib.sha256(content).hexdigest(), files


def _service(tmp_path, session, **kwargs):
    pin, files = _bundle(tmp_path, **kwargs)
    service = MoCoModelService(
        bundle_dir=tmp_path,
        expected_manifest_sha256=pin,
        session_factory=lambda _: session,
    )
    return service, files


def test_v2_bundle_runs_at_448_with_d4_identity(tmp_path):
    session = V2Session()
    service, files = _service(tmp_path, session)

    result = service.predict(Image.new("RGB", (32, 24)), model_key="MOCO_HAM7")

    assert session.feed_shapes and all(shape == (1, 3, 448, 448) for shape in session.feed_shapes)
    assert result["pred"] == "mel"
    assert result["views"] == 8
    assert result["inference_mode"] == "d4_tta_8view"
    assert result["identity"]["head_sha256"] == files["heads/HAM-reader.json"]
    assert result["identity"]["task_sha256"] == files["heads/HAM-reader.json"]


def test_v2_bundle_is_ready_without_report_json(tmp_path):
    service, _ = _service(tmp_path, V2Session())

    status = service.status()

    assert status["classification_ready"] is True
    assert status["release"] == "v10.1.0"
    assert status["identity"]["release"] == "v10.1.0"


def test_v2_rejects_224_input_declared_under_v2_schema(tmp_path):
    service, _ = _service(tmp_path, V2Session(), input_size=224)

    with pytest.raises(V10ArtifactError):
        service.predict(Image.new("RGB", (8, 8)), model_key="MOCO_HAM7")


def test_v2_rejects_v1_recipe(tmp_path):
    service, _ = _service(tmp_path, V2Session(), recipe=EXPECTED_RECIPE)

    with pytest.raises(V10ArtifactError):
        service.predict(Image.new("RGB", (8, 8)), model_key="MOCO_HAM7")


def test_v2_missing_pad_head_fails_closed(tmp_path):
    service, _ = _service(tmp_path, V2Session(), drop=("heads/PAD-reader.json",))

    with pytest.raises(V10ArtifactError):
        service.predict(Image.new("RGB", (8, 8)), model_key="MOCO_PAD6")


def test_v2_rejects_unknown_schema(tmp_path):
    service, _ = _service(tmp_path, V2Session(), schema="dermssl.moco-bundle.v9")

    with pytest.raises(V10ArtifactError):
        service.predict(Image.new("RGB", (8, 8)), model_key="MOCO_HAM7")


def test_v2_manifest_sha_pin_is_still_enforced(tmp_path):
    _, _ = _bundle(tmp_path)
    service = MoCoModelService(
        bundle_dir=tmp_path,
        expected_manifest_sha256=hashlib.sha256(b"other").hexdigest(),
        session_factory=lambda _: V2Session(),
    )

    with pytest.raises(V10ArtifactError):
        service.predict(Image.new("RGB", (8, 8)), model_key="MOCO_HAM7")
