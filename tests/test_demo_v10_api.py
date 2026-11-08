from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from demo.backend.v10_artifacts import BACKBONE_SHA256, EXPECTED_RECIPE, REQUIRED_FILES, TASK_CLASSES
from demo.backend.v10_service import MoCoArtifactError, MoCoModelService, MoCoUnsupportedModeError


class FakeSession:
    def __init__(self, pad_width: int = 6):
        self.requested_outputs = []
        self.pad_width = pad_width

    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values")]

    def get_outputs(self):
        return [SimpleNamespace(name=name) for name in ("ham7_probabilities", "pad6_probabilities")]

    def run(self, output_names, feed):
        self.requested_outputs.append(output_names)
        ham = np.full((1, 7), 1 / 7, dtype=np.float32)
        pad = np.full((1, self.pad_width), 1 / self.pad_width, dtype=np.float32)
        if self.pad_width == 6:
            pad = np.array([[0.05, 0.05, 0.6, 0.1, 0.1, 0.1]], dtype=np.float32)
        return [ham, pad]


def _tasks():
    return {
        "HAM": {"classes": TASK_CLASSES["MOCO_HAM7"], "output_name": "ham7_probabilities", "model_key": "MOCO_HAM7"},
        "PAD": {"classes": TASK_CLASSES["MOCO_PAD6"], "output_name": "pad6_probabilities", "model_key": "MOCO_PAD6"},
    }


def _write_manifest(root, files, *, seed=42, schema="dermssl.moco-bundle.v1", tasks=None, recipe=None, feature_dim=3072):
    manifest = {
        "schema": schema,
        "release": "v10.0.0",
        "seed": seed,
        "backbone_sha256": BACKBONE_SHA256,
        "tasks": _tasks() if tasks is None else tasks,
        "recipe": EXPECTED_RECIPE if recipe is None else recipe,
        "feature_dim": feature_dim,
        "input": {"name": "pixel_values", "dtype": "float32", "shape": ["batch", 3, 224, 224]},
        "report": "report.json",
        "files": files,
    }
    content = json.dumps(manifest).encode("utf-8")
    (root / "manifest.json").write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _service(tmp_path, session=None, preprocess_fn=None):
    source = (
        "import numpy as np\n"
        "def preprocess_image(image):\n"
        "    return np.zeros((1, 3, 224, 224), dtype=np.float32)\n"
    ).encode()
    file_data = {name: b"synthetic release asset" for name in REQUIRED_FILES}
    file_data["model.onnx"] = b"synthetic onnx"
    file_data["preprocess.py"] = source
    file_data["code/dermssl/release/moco_preprocess.py"] = source
    files = {}
    for name, payload in file_data.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    pin = _write_manifest(tmp_path, files)
    return MoCoModelService(
        bundle_dir=tmp_path,
        expected_manifest_sha256=pin,
        session_factory=(lambda _: session) if session is not None else None,
        preprocess_fn=preprocess_fn,
    ), files


def test_moco_task_class_orders_are_dataset_native() -> None:
    assert TASK_CLASSES == {
        "MOCO_HAM7": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "MOCO_PAD6": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
    }


def test_service_requires_independent_manifest_pin() -> None:
    with pytest.raises(MoCoArtifactError, match="manifest SHA256 pin is required"):
        MoCoModelService(bundle_dir="/missing", expected_manifest_sha256=None)


def test_service_rejects_manifest_that_does_not_match_pin(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    service = MoCoModelService(bundle_dir=tmp_path, expected_manifest_sha256="a" * 64)
    with pytest.raises(MoCoArtifactError, match="manifest SHA256 mismatch"):
        service._manifest()


def test_service_rejects_unknown_manifest_schema(tmp_path) -> None:
    content = b'{"schema":"unknown"}'
    (tmp_path / "manifest.json").write_bytes(content)
    service = MoCoModelService(bundle_dir=tmp_path, expected_manifest_sha256=hashlib.sha256(content).hexdigest())
    with pytest.raises(MoCoArtifactError, match="unsupported V10 manifest schema"):
        service._manifest()


def test_service_rejects_wrong_native_task_classes(tmp_path) -> None:
    pin = _write_manifest(tmp_path, {}, tasks={})
    service = MoCoModelService(bundle_dir=tmp_path, expected_manifest_sha256=pin)
    with pytest.raises(MoCoArtifactError, match="V10 manifest tasks do not match"):
        service._manifest()


def test_service_rejects_legacy_tta_recipe(tmp_path) -> None:
    pin = _write_manifest(tmp_path, {}, recipe={**EXPECTED_RECIPE, "tta": "D4"})
    service = MoCoModelService(bundle_dir=tmp_path, expected_manifest_sha256=pin)
    with pytest.raises(MoCoArtifactError, match="recipe does not match single-view CLS4"):
        service._manifest()


def test_service_rejects_wrong_feature_and_input_contract(tmp_path) -> None:
    pin = _write_manifest(tmp_path, {}, feature_dim=768)
    service = MoCoModelService(bundle_dir=tmp_path, expected_manifest_sha256=pin)
    with pytest.raises(MoCoArtifactError, match="V10 manifest feature/input contract mismatch"):
        service._manifest()


def test_service_rejects_unapproved_seed_or_backbone(tmp_path) -> None:
    pin = _write_manifest(tmp_path, {}, seed=41)
    service = MoCoModelService(bundle_dir=tmp_path, expected_manifest_sha256=pin)
    with pytest.raises(MoCoArtifactError, match="V10 manifest provenance mismatch"):
        service._manifest()


def test_service_rejects_unsafe_bundle_paths(tmp_path) -> None:
    service = MoCoModelService(bundle_dir=tmp_path, expected_manifest_sha256="a" * 64)
    with pytest.raises(MoCoArtifactError, match="unsafe V10 bundle path"):
        service._verify_bundle_files({"files": {"../outside": "a" * 64}})


def test_service_requires_hashed_runtime_and_head_assets(tmp_path) -> None:
    model = b"synthetic model"
    (tmp_path / "model.onnx").write_bytes(model)
    service = MoCoModelService(bundle_dir=tmp_path, expected_manifest_sha256="a" * 64)
    with pytest.raises(MoCoArtifactError, match="V10 manifest is missing required runtime files"):
        service._verify_bundle_files({"files": {"model.onnx": hashlib.sha256(model).hexdigest()}})


def test_service_rejects_model_file_that_does_not_match_manifest_hash(tmp_path) -> None:
    service, files = _service(tmp_path)
    (tmp_path / "model.onnx").write_bytes(b"changed model")
    with pytest.raises(MoCoArtifactError, match="V10 SHA256 mismatch for model.onnx"):
        service._verify_bundle_files({"files": files})


def test_service_rechecks_report_hash_before_reading_report(tmp_path) -> None:
    service, _ = _service(tmp_path)
    (tmp_path / "report.json").write_text('{"release":"v10.0.0"}', encoding="utf-8")

    with pytest.raises(MoCoArtifactError, match="V10 SHA256 mismatch for report.json"):
        service.verify_artifact("report.json")


def test_api_does_not_serve_report_that_differs_from_pinned_bundle(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from demo.backend.v10_app import create_app

    service, _ = _service(tmp_path, session=FakeSession())
    report_path = tmp_path / "report.json"
    report_path.write_text('{"release":"v10.0.0"}', encoding="utf-8")
    client = TestClient(create_app(
        moco_service=service, legacy_services={}, report_path=report_path
    ))

    response = client.get("/api/report")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "REPORT_NOT_READY"


def test_status_warms_both_named_outputs_before_reporting_ready(tmp_path) -> None:
    session = FakeSession()
    service, _ = _service(tmp_path, session=session)
    status = service.status()
    assert status["classification_ready"] is True
    assert session.requested_outputs == [["ham7_probabilities", "pad6_probabilities"]]


def test_status_fails_closed_when_pad_warmup_shape_is_wrong(tmp_path) -> None:
    service, _ = _service(tmp_path, session=FakeSession(pad_width=7))
    status = service.status()
    assert status["classification_ready"] is False
    assert "invalid probability shape" in status["reason"]


def test_prediction_uses_requested_head_and_returns_sha_identity(tmp_path) -> None:
    service, files = _service(tmp_path, session=FakeSession())
    prediction = service.predict(Image.new("RGB", (8, 8)), model_key="MOCO_PAD6")
    assert prediction["pred"] == "MEL"
    assert prediction["classes"] == TASK_CLASSES["MOCO_PAD6"]
    assert prediction["model_key"] == "MOCO_PAD6"
    assert prediction["views"] == 1
    assert prediction["inference_mode"] == "single_view"
    assert prediction["identity"]["head_sha256"] == files["heads/PAD-reader.npz"]
    assert prediction["identity"]["task_sha256"] == files["heads/PAD-reader.json"]
    assert prediction["identity"]["backbone_sha256"] == BACKBONE_SHA256
    assert prediction["confidence_calibrated"] is False


def test_moco_tta_request_is_rejected_before_loading_bundle() -> None:
    service = MoCoModelService(bundle_dir="/missing", expected_manifest_sha256="a" * 64)
    with pytest.raises(MoCoUnsupportedModeError, match="TTA is not supported for MoCo V10"):
        service.predict(None, model_key="MOCO_HAM7", mode="tta")
