from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from demo.backend.v9_service import V9ModelService, V9ServiceError


RELEASE_ROOT = Path(os.getenv("V9_RELEASE_ROOT", "demo/models/v9"))


@pytest.mark.parametrize("arm", ["A71", "B71"])
def test_published_v9_bundle_identity_and_hashes_are_verified(arm):
    bundle = RELEASE_ROOT / arm
    if not bundle.is_dir():
        pytest.skip("set V9_RELEASE_ROOT to verify published release artifacts")
    manifest = json.loads((bundle / "manifest.json").read_text())
    service = V9ModelService(bundle)

    assert service.status()["classification_ready"] is True
    assert service.status()["release"] == "v9.0.0"
    assert service.status()["release_arm"] == arm
    assert manifest["recipe"]["n_views"] == 8
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == expected


def test_missing_manifest_does_not_claim_release_ready(tmp_path):
    status = V9ModelService(tmp_path).status()

    assert status["classification_ready"] is False
    assert status["release"] == "v9.0.0"
    assert status["reason"]


def test_unexpected_model_file_fails_closed_even_when_manifest_hash_is_malformed(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"model")
    manifest = {"schema": "dermssl.clean-bundle.v1", "arm": "B71", "verified": True,
                "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                "recipe": {"feature": "cls4", "tau": 0, "tta": "D4",
                           "aggregation": "mean_probabilities", "n_views": 8},
                "files": {"model.onnx": "not-a-hash"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(V9ServiceError, match="invalid model.onnx SHA256"):
        V9ModelService(tmp_path)._checked_model(manifest)
