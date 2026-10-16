import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from demo.backend.app import create_app


@pytest.mark.skipif(os.getenv("DEMO_PHONE_GOLDEN") != "1", reason="requires verified A71/B71 export artifacts")
def test_real_a71_b71_http_consumer_matches_independent_cuda_references():
    import torch
    torch.set_num_threads(4)
    models = Path(os.environ["PROPOSED_MODEL_DIR"])
    images = Path(__file__).resolve().parents[1] / "demo/models/golden_images"
    app = create_app()
    client = TestClient(app)
    summaries = {}
    for arm in ("A71", "B71"):
        manifest = json.loads((models / f"{arm}_classifier_manifest.json").read_text())
        provenance = manifest["provenance"]
        assert provenance["c1_train_ids_sha256"] == "d347edd52d0d764835527d6dd140dedf9ec7206590c9249bc54ebd681426fa97"
        assert provenance["c1_val_ids_sha256"] == "6ff4c17f04a9eae2f0d058aaa41084ea65b9aff91e4d5e5b05c924b4465a2fc6"
        assert hashlib.sha256((models / manifest["probe"]["file"]).read_bytes()).hexdigest() == manifest["probe"]["sha256"]
        golden = json.loads((models / f"{arm}_golden.json").read_text())
        assert len(golden["cases"]) == 7 and golden["tolerance"] == 0.01
        differences = []
        for case in golden["cases"]:
            path = images / f"{case['image_id']}.jpg"
            content = path.read_bytes()
            assert hashlib.sha256(content).hexdigest() == case["image_sha256"]
            response = client.post(f"/api/predict?model={arm}&mode=base", files={"file": (path.name, content, "image/jpeg")})
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["model_key"] == arm and result["views"] == 1
            assert result["identity"]["sha256_verified"] is True
            actual = np.array([result["probs"][label] for label in golden["classes"]])
            expected = np.array([case["probabilities"][label] for label in golden["classes"]])
            difference = float(np.max(np.abs(actual - expected)))
            assert difference <= 0.01, (arm, case["image_id"], difference)
            differences.append(difference)
        summaries[arm] = {"cases": len(differences), "max_abs_probability_difference": max(differences)}
    assert app.state.model_services["A71"]._backbone is None
    print("REAL PHONE GOLDEN", json.dumps(summaries, sort_keys=True))


@pytest.mark.skipif(os.getenv("DEMO_PHONE_PRECISION") != "1", reason="requires independent GPU float32 diagnostic")
def test_cpu_matches_independent_gpu_float32_features_and_probabilities(tmp_path):
    import torch
    torch.set_num_threads(4)
    models = Path(os.environ["PROPOSED_MODEL_DIR"])
    root = Path(__file__).resolve().parents[2]
    precision = root / "export/precision"
    images = Path(__file__).resolve().parents[1] / "demo/models/golden_images"
    app = create_app()
    client = TestClient(app)
    report = {}
    for arm in ("A71", "B71"):
        reference = json.loads((precision / f"{arm}_fp32_math.json").read_text())
        with np.load(precision / reference["features_file"], allow_pickle=False) as data:
            gpu_features = data["features"][:, 0, :].copy()
            assert data["ids"].tolist() == [case["image_id"] for case in reference["cases"]]
        service = app.state.model_services[arm]
        original = service._features
        captured = []
        def capture(image, _original=original, _captured=captured):
            result = _original(image)
            _captured.append(result)
            return result
        service._features = capture
        rows = []
        for index, case in enumerate(reference["cases"]):
            path = images / f"{case['image_id']}.jpg"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == case["image_sha256"]
            response = client.post(f"/api/predict?model={arm}&mode=base", files={"file": (path.name, path.read_bytes(), "image/jpeg")})
            assert response.status_code == 200, response.text
            actual = np.array([response.json()["probs"][label] for label in reference["classes"]])
            expected = np.array([case["probabilities"][label] for label in reference["classes"]])
            probability_difference = float(np.max(np.abs(actual - expected)))
            assert probability_difference <= 1e-3
            cpu, gpu = captured[-1].astype(float), gpu_features[index].astype(float)
            rows.append({"image_id": case["image_id"], "max_abs_probability_difference": probability_difference,
                         "max_abs_feature_difference": float(np.max(np.abs(cpu - gpu))),
                         "feature_cosine": float(cpu @ gpu / (np.linalg.norm(cpu) * np.linalg.norm(gpu)))})
        report[arm] = rows
    output = tmp_path / "cpu_comparison.json"
    output.write_text(json.dumps(report, indent=2))
    print("CPU_GPU_FLOAT32_COMPARISON", output, json.dumps(report, sort_keys=True))
