from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from demo.backend.model_identity import describe_model, verify_model
from demo.backend.model_service import ModelService, ModelUnavailableError


def manifest(tmp_path):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"frozen backbone")
    data = {
        "arm": "ssl_proposed_vitb14",
        "backbone": "dinov2_vitb14",
        "backbone_detail": {
            "release": "proposed-cluster-v1",
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        },
        "feature_extraction": {"dim": 768},
        "preprocess": {"resize_to": 255, "center_crop": 224},
    }
    path = tmp_path / "classifier.json"
    path.write_text(json.dumps(data))
    return path, weights


def test_identity_distinguishes_declared_from_verified_weights(tmp_path):
    path, weights = manifest(tmp_path)
    declared = describe_model(path)
    assert declared["model_id"] == "ssl_proposed_vitb14"
    assert declared["release"] == "proposed-cluster-v1"
    assert declared["sha256_verified"] is False
    assert declared["preprocess"]["resize_to"] == 255
    verified = verify_model(path, weights)
    assert verified["sha256_verified"] is True
    assert verified["backbone_sha256"] == declared["backbone_sha256"]


def test_modified_weights_fail_closed(tmp_path):
    path, weights = manifest(tmp_path)
    weights.write_bytes(b"different backbone")
    with pytest.raises(ValueError, match="checksum"):
        verify_model(path, weights)


def test_missing_manifest_is_not_a_model_identity(tmp_path):
    assert describe_model(tmp_path / "missing.json") is None


def test_status_exposes_identity_without_claiming_hash_verification(tmp_path):
    path, weights = manifest(tmp_path)
    service = ModelService(weights, path, tmp_path / "probe.npz", classifier_manifest_path=path)
    assert service.status()["identity"] == describe_model(path)


def test_loading_rejects_tampering_before_remote_code_or_torch_loading(tmp_path, monkeypatch):
    path, weights = manifest(tmp_path)
    weights.write_bytes(b"tampered")

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before loading torch hub")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(hub=SimpleNamespace(load=forbidden)))
    service = ModelService(weights, path, tmp_path / "probe.npz", classifier_manifest_path=path)
    with pytest.raises(ModelUnavailableError) as error:
        service._load_backbone()
    assert isinstance(error.value.__cause__, ValueError)
    assert "checksum" in str(error.value.__cause__)
