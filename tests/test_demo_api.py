from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from demo.backend.app import create_app


class StubModelService:
    def status(self):
        return {"backbone_ready": True, "classification_ready": True}

    def predict(self, image, mode="base"):
        return {
            "pred": "nv",
            "confidence": 0.43,
            "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
            "probs": {"mel": 0.415, "nv": 0.43},
            "feature_dim": 768,
            "mode": mode,
        }


def test_report_endpoint_returns_the_documented_result() -> None:
    client = TestClient(create_app())

    response = client.get("/api/report")

    assert response.status_code == 200
    body = response.json()
    assert body["classification"]["ssl_proposed_vitb14"]["macro_f1"] == 0.5237
    assert body["classification_verdict"]["global"] == "loss"


def test_prediction_rejects_non_image_upload() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_prediction_marks_the_known_golden_ground_truth() -> None:
    root = Path(__file__).resolve().parents[1]
    client = TestClient(create_app(service=StubModelService()))
    image = root / "demo" / "models" / "golden_images" / "ISIC_0032840.jpg"

    with image.open("rb") as handle:
        response = client.post(
            "/api/predict",
            files={"file": (image.name, handle, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["pred"] == "nv"
    assert response.json()["benchmark"] == {
        "image_id": "ISIC_0032840",
        "true_class": "mel",
    }


def test_prediction_forwards_tta_mode_to_the_model_service() -> None:
    root = Path(__file__).resolve().parents[1]
    client = TestClient(create_app(service=StubModelService()))
    image = root / "demo" / "models" / "golden_images" / "ISIC_0024504.jpg"

    with image.open("rb") as handle:
        response = client.post(
            "/api/predict?mode=tta",
            files={"file": (image.name, handle, "image/jpeg")},
        )

    assert response.status_code == 200
    assert response.json()["mode"] == "tta"
