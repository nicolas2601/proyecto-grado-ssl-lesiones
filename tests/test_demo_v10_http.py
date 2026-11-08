from __future__ import annotations

from fastapi.testclient import TestClient

from demo.backend.v10_app import create_app


class StubService:
    def status(self):
        return {"backbone_ready": True, "classification_ready": True, "reason": "ready"}

    def predict(self, image, model_key="MOCO_HAM7", mode="base"):
        return {
            "pred": "nv",
            "confidence": 0.8,
            "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
            "probs": {"nv": 0.8},
            "feature_dim": 3072,
            "mode": mode,
            "requested_mode": mode,
            "inference_mode": "single_view",
            "views": 1,
            "model_key": model_key,
            "identity": {"release": "v10.0.0"},
        }


def test_default_status_and_health_use_moco_ham7(tmp_path) -> None:
    app = create_app(
        moco_service=StubService(), legacy_services={}, report_path=tmp_path / "report.json"
    )
    client = TestClient(app)

    status = client.get("/api/status")
    health = client.get("/health")

    assert status.status_code == 200
    assert status.json()["model_key"] == "MOCO_HAM7"
    assert health.status_code == 200
    assert health.json()["status"] == "ready"


def test_tta_is_rejected_with_explicit_error_code(tmp_path) -> None:
    app = create_app(
        moco_service=StubService(), legacy_services={}, report_path=tmp_path / "report.json"
    )
    client = TestClient(app)

    response = client.post(
        "/api/predict?model=MOCO_HAM7&mode=tta",
        files={"file": ("image.png", b"not decoded", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNSUPPORTED_INFERENCE_MODE"


def test_report_endpoint_fails_closed_when_generated_report_is_missing(tmp_path) -> None:
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={}, report_path=tmp_path / "missing.json"
    ))

    response = client.get("/api/report")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "REPORT_NOT_READY"


def test_non_image_mime_is_rejected_before_inference(tmp_path) -> None:
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={}, report_path=tmp_path / "report.json"
    ))

    response = client.post(
        "/api/predict", files={"file": ("notes.txt", b"anything", "text/plain")}
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_image_exif_orientation_is_applied_before_inference(tmp_path) -> None:
    from io import BytesIO

    from PIL import Image

    class CaptureService(StubService):
        dimensions = None

        def predict(self, image, model_key="MOCO_HAM7", mode="base"):
            self.dimensions = image.size
            return super().predict(image, model_key, mode)

    image = Image.new("RGB", (2, 3), color="red")
    exif = Image.Exif()
    exif[274] = 6
    content = BytesIO()
    image.save(content, format="JPEG", exif=exif)
    service = CaptureService()
    client = TestClient(create_app(
        moco_service=service, legacy_services={}, report_path=tmp_path / "report.json"
    ))

    response = client.post(
        "/api/predict", files={"file": ("photo.jpg", content.getvalue(), "image/jpeg")}
    )

    assert response.status_code == 200
    assert service.dimensions == (3, 2)


def test_supported_mime_with_corrupt_image_bytes_returns_400(tmp_path) -> None:
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={}, report_path=tmp_path / "report.json"
    ))

    response = client.post(
        "/api/predict", files={"file": ("bad.png", b"not an image", "image/png")}
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"


def test_report_rejects_historical_release_metrics(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        '{"release":"v9.0.0","classification":{"macro_f1":0.99}}', encoding="utf-8"
    )
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={}, report_path=report_path
    ))

    response = client.get("/api/report")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "REPORT_NOT_READY"


def test_switching_from_legacy_to_moco_unloads_legacy_service(tmp_path) -> None:
    class LegacyService(StubService):
        unload_count = 0

        def unload(self):
            self.unload_count += 1

    legacy = LegacyService()
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={"A71": legacy},
        report_path=tmp_path / "report.json",
    ))

    assert client.get("/api/status?model=A71").status_code == 200
    assert client.get("/api/status?model=MOCO_HAM7").status_code == 200
    assert legacy.unload_count == 1


def test_legacy_a71_keeps_v9_identity_and_tta_behavior(tmp_path) -> None:
    from io import BytesIO

    from PIL import Image

    class LegacyService(StubService):
        requested_mode = None

        def status(self):
            return {"identity": {"release": "v9.0.0", "arm": "A71"}}

        def predict(self, image, mode="base"):
            self.requested_mode = mode
            return {"pred": "nv", "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
                    "mode": mode, "inference_mode": "D4", "views": 8}

    image = Image.new("RGB", (4, 4), color="blue")
    content = BytesIO()
    image.save(content, format="PNG")
    legacy = LegacyService()
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={"A71": legacy},
        report_path=tmp_path / "report.json",
    ))

    response = client.post(
        "/api/predict?model=A71&mode=tta",
        files={"file": ("image.png", content.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["identity"]["release"] == "v9.0.0"
    assert response.json()["model_key"] == "A71"
    assert legacy.requested_mode == "tta"


def test_image_pixel_limit_rejects_oversized_dimensions(tmp_path, monkeypatch) -> None:
    from io import BytesIO

    from PIL import Image

    from demo.backend import v10_app

    monkeypatch.setattr(v10_app, "MAX_IMAGE_PIXELS", 4)
    image = Image.new("RGB", (3, 3), color="red")
    content = BytesIO()
    image.save(content, format="PNG")
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={}, report_path=tmp_path / "report.json"
    ))

    response = client.post(
        "/api/predict", files={"file": ("large.png", content.getvalue(), "image/png")}
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_upload_byte_limit_rejects_oversized_image(tmp_path, monkeypatch) -> None:
    from io import BytesIO

    from PIL import Image

    from demo.backend import v10_app

    monkeypatch.setattr(v10_app, "MAX_UPLOAD_BYTES", 8)
    content = BytesIO()
    Image.new("RGB", (4, 4), color="red").save(content, format="PNG")
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={}, report_path=tmp_path / "report.json"
    ))

    response = client.post(
        "/api/predict", files={"file": ("large.png", content.getvalue(), "image/png")}
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_v10_report_preserves_evidence_scope_and_adds_research_disclaimer(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text(
        '{"release":"v10.0.0","scope":"PAD transductive; HAM development selection-exposed",'
        '"classification":{"split":"development"}}',
        encoding="utf-8",
    )
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={}, report_path=report_path
    ))

    response = client.get("/api/report")
    body = response.json()

    assert response.status_code == 200
    assert body["scope"] == "PAD transductive; HAM development selection-exposed"
    assert "research use only" in body["api_scope"]
    assert "not calibrated clinical probabilities" in body["disclaimer"]
    assert body["classification"]["split"] == "development"


def test_cors_allows_only_the_existing_phone_frontend(tmp_path) -> None:
    client = TestClient(create_app(
        moco_service=StubService(), legacy_services={}, report_path=tmp_path / "report.json"
    ))
    headers = {
        "Origin": "https://dermssl-phone.vercel.app",
        "Access-Control-Request-Method": "GET",
    }
    allowed = client.options("/api/status", headers=headers)
    blocked = client.options(
        "/api/status",
        headers={**headers, "Origin": "https://attacker.example"},
    )

    assert allowed.headers["access-control-allow-origin"] == "https://dermssl-phone.vercel.app"
    assert "access-control-allow-origin" not in blocked.headers
    assert allowed.headers["x-content-type-options"] == "nosniff"
