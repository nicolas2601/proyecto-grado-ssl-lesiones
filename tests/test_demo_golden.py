import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from demo.backend.app import create_app
from demo.backend.model_service import ModelService


@pytest.mark.skipif(os.getenv("DEMO_GOLDEN") != "1", reason="opt-in real checkpoint integration")
def test_legacy_golden_matches_with_cached_architecture_only(monkeypatch):
    import torch

    torch.set_num_threads(4)
    architecture = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    assert (architecture / "hubconf.py").is_file(), "cached architecture required; no downloads"
    original_load = torch.hub.load

    def local_load(repository, model, **kwargs):
        assert repository == "facebookresearch/dinov2"
        assert kwargs["pretrained"] is False
        return original_load(str(architecture), model, source="local", **kwargs)

    monkeypatch.setattr(torch.hub, "load", local_load)
    models = Path(__file__).resolve().parents[1] / "demo" / "models"
    service = ModelService(
        models / "ssl_proposed_vitb14_backbone.pt",
        models / "ssl_proposed_vitb14_run.json",
        models / "ssl_proposed_vitb14_probe.npz",
        device="cpu",
    )
    golden = json.loads((models / "ssl_proposed_vitb14_golden.json").read_text())
    client = TestClient(create_app(service=service))
    differences = []
    for case in golden["cases"]:
        path = models / "golden_images" / f"{case['image_id']}.jpg"
        response = client.post(
            "/api/predict?mode=base",
            files={"file": (path.name, path.read_bytes(), "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        prediction = response.json()
        actual = np.array([prediction["probs"][label] for label in golden["classes"]])
        expected = np.array([case["probabilities"][label] for label in golden["classes"]])
        difference = float(np.max(np.abs(actual - expected)))
        differences.append(difference)
        assert difference <= golden["tolerance"], (case["image_id"], difference)
    assert service.status()["identity"]["sha256_verified"] is True
    print(f"golden: {len(differences)}/{len(golden['cases'])}, max_abs_diff={max(differences):.8f}")
