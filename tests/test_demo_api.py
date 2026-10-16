from __future__ import annotations

from pathlib import Path

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

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


@pytest.mark.parametrize("endpoint", ["predict", "features"])
def test_image_dimensions_are_bounded_before_decoding(monkeypatch, endpoint) -> None:
    monkeypatch.setattr("demo.backend.app.MAX_IMAGE_PIXELS", 100, raising=False)
    image = io.BytesIO()
    Image.new("RGB", (11, 10)).save(image, format="PNG")
    client = TestClient(create_app(service=StubModelService()))

    response = client.post(
        f"/api/{endpoint}",
        files={"file": ("large.png", image.getvalue(), "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_pillow_decompression_bomb_is_rejected(monkeypatch) -> None:
    image = io.BytesIO()
    Image.new("RGB", (11, 10)).save(image, format="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)
    client = TestClient(create_app(service=StubModelService()))

    response = client.post(
        "/api/predict",
        files={"file": ("bomb.png", image.getvalue(), "image/png")},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_total_request_is_rejected_before_fastapi_parses_it() -> None:
    client = TestClient(create_app(service=StubModelService()))
    response = client.post(
        "/api/predict", content=b"x" * (10 * 1024 * 1024 + 65537),
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 413
    assert response.headers["Cache-Control"] == "no-store"


def test_models_are_selected_explicitly_without_replacing_legacy():
    client = TestClient(create_app(service=StubModelService(), public_service=StubModelService()))
    image = io.BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    response = client.post("/api/predict?model=public", files={"file": ("x.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["model_key"] == "public"
    assert {"legacy", "public"}.issubset(client.get("/api/report").json()["models"])
    assert client.post("/api/predict?model=unknown").status_code == 422


def test_cors_only_allows_configured_frontend(monkeypatch):
    monkeypatch.setenv("DEMO_FRONTEND_ORIGIN", "https://dermssl.vercel.app")
    client = TestClient(create_app(service=StubModelService()))
    allowed = client.get("/api/status", headers={"Origin": "https://dermssl.vercel.app"})
    denied = client.get("/api/status", headers={"Origin": "https://other.example"})
    assert allowed.headers["access-control-allow-origin"] == "https://dermssl.vercel.app"
    assert "access-control-allow-origin" not in denied.headers


def test_swagger_resources_are_allowed_only_on_docs():
    client = TestClient(create_app(service=StubModelService()))
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "https://cdn.jsdelivr.net" in docs.headers["content-security-policy"]
    assert "https://cdn.jsdelivr.net" not in client.get("/").headers["content-security-policy"]
    schema = client.get("/openapi.json").json()
    predict = schema["paths"]["/api/predict"]["post"]
    assert {item["name"] for item in predict["parameters"]} == {"mode", "model"}
    assert "multipart/form-data" in predict["requestBody"]["content"]


def test_proposed_models_have_explicit_separate_identities():
    app = create_app(service=StubModelService())
    client = TestClient(app)
    models = client.get("/api/report").json()["models"]
    assert {"A71", "B71"}.issubset(models)
    assert app.state.model_services["A71"].weights_path != app.state.model_services["B71"].weights_path
    assert "weights_path" not in models["B71"]
    assert models["B71"]["classification_ready"] is False


def test_deployment_default_is_b71_and_api_is_not_cached(monkeypatch):
    monkeypatch.setenv("DEMO_DEFAULT_MODEL", "B71")
    client = TestClient(create_app(service=StubModelService()))
    response = client.get("/api/status")
    assert response.json()["model_key"] == "B71"
    assert "weights_path" not in response.json()
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["strict-transport-security"] == "max-age=31536000"
    parameters = client.get("/openapi.json").json()["paths"]["/api/predict"]["post"]["parameters"]
    assert next(item for item in parameters if item["name"] == "model")["schema"]["default"] == "B71"


def test_switching_proposed_models_releases_previous_backbone():
    app = create_app(service=StubModelService())
    services = app.state.model_services
    services["A71"]._backbone = object()
    def predict(image, mode):
        assert services["A71"]._backbone is None
        services["B71"]._backbone = object()
        return {"pred": "nv", "probs": {"nv": 1.0}}
    services["B71"].predict = predict
    image = io.BytesIO()
    Image.new("RGB", (4, 4)).save(image, format="PNG")
    response = TestClient(app).post("/api/predict?model=B71", files={"file": ("x.png", image.getvalue(), "image/png")})
    assert response.status_code == 200
    assert response.json()["model_key"] == "B71"


def test_api_deployment_does_not_publish_legacy_frontend(monkeypatch):
    monkeypatch.setenv("DEMO_API_ONLY", "1")
    client = TestClient(create_app(service=StubModelService()))
    response = client.get("/")
    assert response.json() == {"service": "DermSSL research API", "docs": "/docs"}
    assert set(client.get("/api/report").json()["models"]) == {"A71", "B71"}
    parameters = client.get("/openapi.json").json()["paths"]["/api/predict"]["post"]["parameters"]
    model = next(item for item in parameters if item["name"] == "model")["schema"]
    assert model["pattern"] == "^(A71|B71)$"
    assert model["default"] == "B71"
