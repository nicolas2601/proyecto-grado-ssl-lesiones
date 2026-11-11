from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from demo.backend.analyze_pipeline import run_extras
from demo.backend.segmenter_geometry import encode_mask_png
from demo.backend.segmenter_service import SegmenterError, SegmenterService, Segmentation
from demo.backend.v10_app import create_app
from demo.backend.v10_service import MoCoServiceError

SKIN = (200, 150, 120)
CLASSIFICATION = {"pred": "nv", "confidence": 0.5, "classes": ["nv"], "probs": {"nv": 1.0}, "model_key": "MOCO_HAM7"}


class FakeMoCo:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def status(self):
        return {"backbone_ready": True, "classification_ready": True, "reason": "ready", "release": "v10.1.0"}

    def predict(self, image, model_key="MOCO_HAM7", mode="base"):
        self.calls.append((model_key, mode, image.size))
        if self.error is not None:
            raise self.error
        return dict(CLASSIFICATION)


class FakeSegmenter:
    def __init__(self, error=None, empty=False):
        self.error = error
        self.empty = empty

    def status(self):
        return {"ready": True, "reason": "ready", "detail": None, "identity": {"model_id": "fake"}}

    def segment(self, rgb):
        if self.error is not None:
            raise self.error
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        if not self.empty:
            mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = True
        if not mask.any():
            raise SegmenterError("empty_mask")
        payload = {
            "image_width": width,
            "image_height": height,
            "mask_width": width,
            "mask_height": height,
            "mask_png_base64": base64.b64encode(encode_mask_png(mask)).decode("ascii"),
            "contour": [[width // 4, height // 4], [3 * width // 4, height // 4]],
            "area_fraction": 0.25,
            "threshold": 0.5,
            "identity": {"model_id": "fake", "input_size": 224},
        }
        return Segmentation(payload=payload, full_mask=mask)


def _png(color=SKIN, size=(200, 160)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _client(moco=None, segmenter=None):
    app = create_app(moco_service=moco or FakeMoCo(), report_path=None, segmenter_service=segmenter)
    return TestClient(app)


def _mask_b64(width=200, height=160):
    mask = np.zeros((height, width), dtype=bool)
    mask[40:120, 50:150] = True
    return base64.b64encode(encode_mask_png(mask)).decode("ascii")


def test_analyze_returns_classification_segmentation_and_photometry():
    client = _client(segmenter=FakeSegmenter())

    response = client.post("/api/analyze?model=MOCO_HAM7", files={"file": ("a.png", _png(), "image/png")})

    body = response.json()
    assert response.status_code == 200
    assert body["schema"] == "dermssl.analyze.v1"
    assert body["scope"] == "research"
    assert body["classification"] == CLASSIFICATION
    assert body["segmentation"]["status"] == "ok"
    assert body["segmentation"]["reason"] is None
    assert body["segmentation"]["image_width"] == 200
    assert body["photometry"]["status"] == "ok"
    assert body["photometry"]["region"] == "perilesional_ring"
    assert set(body["timing_ms"]) == {"classification", "segmentation", "photometry"}


def test_segmenter_missing_never_fails_the_request():
    client = _client(segmenter=SegmenterService("/nonexistent", None))

    response = client.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")})

    body = response.json()
    assert response.status_code == 200
    assert body["classification"] == CLASSIFICATION
    assert body["segmentation"]["status"] == "unavailable"
    assert body["segmentation"]["reason"] == "segmenter_error"
    assert body["photometry"]["status"] == "unavailable"
    assert body["photometry"]["reason"] == "no_mask"


def test_empty_mask_reason_is_reported_without_failing():
    client = _client(segmenter=FakeSegmenter(empty=True))

    body = client.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")}).json()

    assert body["segmentation"]["reason"] == "empty_mask"
    assert body["photometry"]["reason"] == "no_mask"


def test_unexpected_segmenter_crash_is_contained():
    client = _client(segmenter=FakeSegmenter(error=RuntimeError("boom")))

    response = client.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")})

    assert response.status_code == 200
    assert response.json()["segmentation"]["reason"] == "segmenter_error"


def test_classification_failure_fails_the_request_with_error_shape():
    client = _client(moco=FakeMoCo(error=MoCoServiceError("down")), segmenter=FakeSegmenter())

    response = client.post("/api/analyze", files={"file": ("a.png", _png(), "image/png")})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"


def test_analyze_rejects_unsupported_media_type():
    client = _client(segmenter=FakeSegmenter())

    response = client.post("/api/analyze", files={"file": ("a.gif", b"GIF89a", "image/gif")})

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_status_reports_segmenter_readiness_as_additive_field():
    client = _client(segmenter=FakeSegmenter())

    body = client.get("/api/status").json()

    assert body["classification_ready"] is True
    assert body["segmenter"]["ready"] is True


def test_simulate_tone_returns_webp_and_preserved_lesion():
    client = _client(segmenter=FakeSegmenter())

    response = client.post(
        "/api/simulate-tone",
        files={"file": ("a.png", _png(), "image/png")},
        data={"mask_png_base64": _mask_b64(), "delta_l": "8"},
    )

    body = response.json()
    assert response.status_code == 200
    assert body["schema"] == "dermssl.simulate-tone.v1"
    assert body["scope"] == "research-simulation"
    assert body["delta_l"] == 8.0
    assert body["lesion_pixels_preserved"] is True
    assert max(body["width"], body["height"]) <= 768
    decoded = Image.open(io.BytesIO(base64.b64decode(body["image_webp_base64"])))
    assert decoded.format == "WEBP"
    assert body["photometry"]["region"] == "perilesional_ring"


@pytest.mark.parametrize("delta", ["20", "-16.5", "abc", "nan"])
def test_simulate_tone_rejects_out_of_range_delta(delta):
    client = _client(segmenter=FakeSegmenter())

    response = client.post(
        "/api/simulate-tone",
        files={"file": ("a.png", _png(), "image/png")},
        data={"mask_png_base64": _mask_b64(), "delta_l": delta},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_DELTA"


def test_simulate_tone_rejects_degenerate_or_undecodable_mask():
    client = _client(segmenter=FakeSegmenter())
    full = base64.b64encode(encode_mask_png(np.ones((160, 200), dtype=bool))).decode("ascii")

    for mask in (full, "not-base64!!", base64.b64encode(b"junk").decode("ascii")):
        response = client.post(
            "/api/simulate-tone",
            files={"file": ("a.png", _png(), "image/png")},
            data={"mask_png_base64": mask, "delta_l": "4"},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_MASK"


def test_run_extras_contains_unexpected_photometry_failure(monkeypatch):
    import demo.backend.analyze_pipeline as pipeline

    def explode(*_args):
        raise ValueError("unexpected")

    monkeypatch.setattr(pipeline, "measure_photometry", explode)
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    result = run_extras(FakeSegmenter(), image)

    assert result["photometry"]["status"] == "unavailable"
    assert result["photometry"]["reason"] == "photometry_error"
    assert result["segmentation"]["status"] == "ok"
