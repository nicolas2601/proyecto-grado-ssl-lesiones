from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from demo.backend.model_service import (
    ModelService,
    ProbeArtifactError,
    predict_probabilities,
)


def test_predict_probabilities_applies_probe_standardization() -> None:
    artifact = {
        "classes": ["akiec", "mel"],
        "mean": [0.0, 1.0],
        "scale": [1.0, 2.0],
        "coef": [[-1.0, -1.0], [1.0, 1.0]],
        "intercept": [0.0, 0.0],
        "feature_dim": 2,
    }

    result = predict_probabilities(np.array([2.0, 3.0]), artifact)

    assert result["pred"] == "mel"
    assert result["classes"] == ["akiec", "mel"]
    assert result["probs"]["mel"] > result["probs"]["akiec"]
    assert sum(result["probs"].values()) == pytest.approx(1.0)


def test_predict_probabilities_rejects_wrong_feature_dimension() -> None:
    artifact = {
        "classes": ["akiec", "mel"],
        "mean": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "coef": [[1.0, 1.0], [-1.0, -1.0]],
        "intercept": [0.0, 0.0],
        "feature_dim": 2,
    }

    with pytest.raises(ProbeArtifactError, match="feature dimension"):
        predict_probabilities(np.array([1.0]), artifact)


def test_model_service_reports_missing_weights_without_loading_torch(tmp_path: Path) -> None:
    service = ModelService(
        weights_path=tmp_path / "missing.pt",
        manifest_path=tmp_path / "missing_run.json",
        probe_path=tmp_path / "missing_probe.npz",
    )

    status = service.status()

    assert status["backbone_ready"] is False
    assert status["classification_ready"] is False
    assert status["reason"] == "weights_missing"


def test_model_service_rejects_a_manifest_for_the_wrong_architecture(tmp_path: Path) -> None:
    weights = tmp_path / "ssl_proposed_vitb14_backbone.pt"
    manifest = tmp_path / "ssl_proposed_vitb14_run.json"
    probe = tmp_path / "probe.npz"
    weights.write_bytes(b"not-loaded-in-this-status-check")
    manifest.write_text('{"backbone": "dinov2_vits14"}', encoding="utf-8")

    service = ModelService(weights, manifest, probe)

    status = service.status()

    assert status["backbone_ready"] is False
    assert status["classification_ready"] is False
    assert status["reason"] == "manifest_invalid"


def test_predict_tta_averages_three_views_without_changing_probe_formula(tmp_path: Path) -> None:
    service = ModelService(
        tmp_path / "weights.pt",
        tmp_path / "run.json",
        tmp_path / "probe.npz",
    )
    service._probe = {
        "classes": ["akiec", "mel"],
        "mean": [0.0, 0.0],
        "scale": [1.0, 1.0],
        "coef": [[-1.0, -1.0], [1.0, 1.0]],
        "intercept": [0.0, 0.0],
        "feature_dim": 2,
    }
    vectors = iter(
        [
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        ]
    )
    service._features = lambda image: next(vectors)

    result = service.predict_tta(Image.new("RGB", (8, 8)))

    assert result["inference_mode"] == "tta_3views"
    assert result["views"] == 3
    assert result["pred"] == "mel"
