from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from demo.backend import v9_service
from demo.backend.v9_service import V9ModelService, V9ServiceError


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
PREPROCESS = Path(__file__).parent / "fixtures" / "v9_preprocess.py"


@pytest.fixture(autouse=True)
def fake_artifact_pins(monkeypatch):
    monkeypatch.setitem(v9_service.MODEL_HASHES, "B71", hashlib.sha256(b"fake onnx").hexdigest())
    monkeypatch.setitem(v9_service.RUNTIME_HASHES, "runner.py", hashlib.sha256(b"runner").hexdigest())


def bundle(tmp_path, arm="B71"):
    source_preprocess = PREPROCESS
    files = {"model.onnx": b"fake onnx", "runner.py": b"runner", "preprocess.py": source_preprocess.read_bytes()}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    recipe = {"feature": "cls4", "tau": 0.0, "tta": "D4", "aggregation": "mean_probabilities", "n_views": 8}
    manifest = {
        "schema": "dermssl.clean-bundle.v1", "arm": arm, "classes": CLASSES,
        "recipe": recipe, "verified": True,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


class FakeSession:
    instances = []

    def __init__(self, path, sess_options, providers):
        self.path, self.options, self.providers = path, sess_options, providers
        self.calls = []
        self.instances.append(self)

    def get_inputs(self):
        return [type("Input", (), {"name": "images"})()]

    def run(self, _, feed):
        batch = feed["images"]
        self.calls.append(batch.copy())
        probs = np.zeros((len(batch), 7), dtype=np.float32)
        probs[:, len(self.calls) % 7] = 1
        return [probs]


class FakeOrt:
    class SessionOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

    InferenceSession = FakeSession


def test_v9_predict_uses_d4_mean_probabilities_for_base_and_tta(monkeypatch, tmp_path):
    FakeSession.instances.clear()
    monkeypatch.setattr("demo.backend.v9_service._load_onnxruntime", lambda: FakeOrt)
    service = V9ModelService(bundle(tmp_path))
    image = Image.new("RGB", (250, 255), color=(40, 80, 120))

    result = service.predict(image, "base")

    assert len(FakeSession.instances[0].calls) == 8
    assert FakeSession.instances[0].options.intra_op_num_threads == 2
    assert FakeSession.instances[0].options.inter_op_num_threads == 1
    assert result["inference_mode"] == "D4"
    assert result["requested_mode"] == "base"
    assert result["mode"] == "base"
    assert result["views"] == 8
    assert result["feature_dim"] == 3072
    assert sum(result["probs"].values()) == pytest.approx(1.0)

    second = service.predict(image, "tta")
    assert len(FakeSession.instances) == 1
    assert len(FakeSession.instances[0].calls) == 16
    assert second["requested_mode"] == "tta"
    assert second["mode"] == "tta"
    assert second["inference_mode"] == "D4"


def test_v9_manifest_hash_mismatch_fails_closed_before_session(monkeypatch, tmp_path):
    FakeSession.instances.clear()
    monkeypatch.setattr("demo.backend.v9_service._load_onnxruntime", lambda: FakeOrt)
    root = bundle(tmp_path)
    (root / "model.onnx").write_bytes(b"tampered")
    service = V9ModelService(root)

    with pytest.raises(V9ServiceError, match="SHA256 mismatch"):
        service.predict(Image.new("RGB", (224, 224)), "base")
    assert FakeSession.instances == []


def test_v9_unloads_resident_session_when_manifest_changes(monkeypatch, tmp_path):
    FakeSession.instances.clear()
    monkeypatch.setattr("demo.backend.v9_service._load_onnxruntime", lambda: FakeOrt)
    root = bundle(tmp_path)
    service = V9ModelService(root)
    image = Image.new("RGB", (224, 224), color=(20, 30, 40))
    service.predict(image)
    old_session = service._session
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["recipe"]["model_config_revision"] = 2
    (root / "manifest.json").write_text(json.dumps(manifest))

    service.predict(image, "tta")

    assert len(FakeSession.instances) == 2
    assert service._session is not old_session


def test_v9_manifest_recipe_mismatch_fails_closed(tmp_path):
    root = bundle(tmp_path)
    data = json.loads((root / "manifest.json").read_text())
    data["recipe"]["n_views"] = 1
    (root / "manifest.json").write_text(json.dumps(data))

    with pytest.raises(V9ServiceError, match="full D4 recipe"):
        V9ModelService(root)._manifest()


def test_manifest_with_wrong_but_valid_model_digest_is_rejected(tmp_path):
    root = bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["model.onnx"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(manifest))
    service = V9ModelService(root)

    with pytest.raises(V9ServiceError, match="SHA256 mismatch"):
        service._verify_bundle_files(service._manifest())


def test_rewriting_both_model_and_manifest_cannot_replace_the_published_model(tmp_path):
    root = bundle(tmp_path)
    (root / "model.onnx").write_bytes(b"replacement")
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["model.onnx"] = hashlib.sha256(b"replacement").hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest))

    assert V9ModelService(root).status()["classification_ready"] is False


def test_model_directory_cannot_swap_a71_for_b71(tmp_path):
    service = V9ModelService(bundle(tmp_path), expected_arm="A71")

    with pytest.raises(V9ServiceError, match="requested model"):
        service._manifest()


def test_unsupported_feature_endpoint_returns_specific_error(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from demo.backend.app import create_app

    monkeypatch.setenv("DEMO_V9_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("DEMO_API_ONLY", "1")
    response = TestClient(create_app()).post("/api/features", files={"file": ("x.png", b"x", "image/png")})

    assert response.status_code == 501
    assert response.json()["error"]["code"] == "FEATURES_UNSUPPORTED"


def test_v9_canonical_preprocessing_matches_release_implementation(monkeypatch, tmp_path):
    from importlib.util import module_from_spec, spec_from_file_location

    source = PREPROCESS
    spec = spec_from_file_location("release_preprocess", source)
    release_preprocess = module_from_spec(spec)
    spec.loader.exec_module(release_preprocess)
    monkeypatch.setattr("demo.backend.v9_service._load_onnxruntime", lambda: FakeOrt)
    service = V9ModelService(bundle(tmp_path))
    image = Image.new("RGB", (251, 257), color=(17, 91, 222))

    service.predict(image, "base")

    captured = FakeSession.instances[-1].calls[0]
    np.testing.assert_array_equal(captured, release_preprocess.preprocess_image(image)[None, ...])
