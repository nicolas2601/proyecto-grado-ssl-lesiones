from __future__ import annotations

import base64
import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from demo.backend.segmenter_geometry import (
    decode_mask_png,
    douglas_peucker,
    encode_mask_png,
    largest_component,
    outer_contour,
    simplify_contour,
)
from demo.backend.segmenter_service import SegmenterError, SegmenterService

SIZE = 224


def _square(h=10, w=12, top=2, left=3):
    mask = np.zeros((h, w), dtype=bool)
    mask[top : top + 4, left : left + 5] = True
    return mask


def test_largest_component_keeps_biggest_blob_only():
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:3, 1:3] = True
    mask[8:15, 8:15] = True

    kept = largest_component(mask)

    assert kept.sum() == 49
    assert not kept[1, 1]


def test_outer_contour_traces_square_boundary():
    mask = _square()

    contour = outer_contour(mask)

    assert contour[0] == (3, 2)
    assert all(mask[y, x] for x, y in contour)
    xs = [x for x, _ in contour]
    ys = [y for _, y in contour]
    assert (min(xs), max(xs)) == (3, 7)
    assert (min(ys), max(ys)) == (2, 5)


def test_outer_contour_of_single_pixel_is_that_pixel():
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 3] = True

    assert outer_contour(mask) == [(3, 2)]


def test_douglas_peucker_collapses_collinear_points():
    points = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 3)]

    assert douglas_peucker(points, 0.5) == [(0, 0), (3, 0), (3, 3)]


def test_simplify_contour_caps_points_for_large_circle():
    yy, xx = np.mgrid[:400, :400]
    disk = (yy - 200) ** 2 + (xx - 200) ** 2 <= 180**2

    points = simplify_contour(outer_contour(disk), max_points=256)

    assert 3 <= len(points) <= 256


def test_mask_png_round_trips_as_binary_grayscale():
    mask = _square()

    payload = encode_mask_png(mask)

    assert np.array_equal(decode_mask_png(payload), mask)
    assert Image.open(io.BytesIO(payload)).mode == "L"


def _write_segmenter(tmp_path, *, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), threshold=0.5):
    onnx_bytes = b"synthetic segmenter"
    (tmp_path / "segmenter.onnx").write_bytes(onnx_bytes)
    manifest = {
        "schema": "dermssl.segmenter.v1",
        "model_id": "dermlip-resnet34-unet-teacher",
        "run": "abl0xx-ima-sNN",
        "release": "v10.2.0",
        "supervision": "semi-supervised (PH2 + IMA++ masks, unlabeled ISIC pool)",
        "input": {
            "name": "pixel_values",
            "dtype": "float32",
            "shape": ["batch", 3, SIZE, SIZE],
            "color": "RGB",
            "scale": 255.0,
            "mean": list(mean),
            "std": list(std),
        },
        "output": {"name": "logits", "shape": ["batch", 1, SIZE, SIZE]},
        "threshold": threshold,
        "files": {"segmenter.onnx": hashlib.sha256(onnx_bytes).hexdigest()},
    }
    content = json.dumps(manifest).encode("utf-8")
    (tmp_path / "segmenter_manifest.json").write_bytes(content)
    return hashlib.sha256(content).hexdigest()


class FakeSegSession:
    def __init__(self, logits=None):
        self.feeds = []
        self._logits = logits

    def get_inputs(self):
        return [SimpleNamespace(name="pixel_values")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, names, feed):
        self.feeds.append(feed["pixel_values"])
        if self._logits is not None:
            return [self._logits]
        yy, xx = np.mgrid[:SIZE, :SIZE]
        disk = (yy - 112) ** 2 + (xx - 112) ** 2 <= 60**2
        logits = np.where(disk, 5.0, -5.0).astype(np.float32)
        logits[0:3, 0:3] = 5.0  # isolated speck, must be dropped
        return [logits[None, None]]


def _image(width=640, height=480, value=(128, 96, 80)):
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_segment_returns_largest_component_mask_and_contour(tmp_path):
    pin = _write_segmenter(tmp_path)
    service = SegmenterService(tmp_path, pin, session_factory=lambda _: FakeSegSession())

    result = service.segment(_image())

    assert result.payload["image_width"] == 640 and result.payload["image_height"] == 480
    assert max(result.payload["mask_width"], result.payload["mask_height"]) <= 512
    assert result.payload["threshold"] == 0.5
    assert result.payload["area_fraction"] == pytest.approx(np.pi * 60**2 / SIZE**2, abs=0.02)
    assert result.full_mask.shape == (480, 640)
    contour = result.payload["contour"]
    assert len(contour) <= 256
    assert min(x for x, _ in contour) > 0.05 * 640
    assert result.payload["identity"]["model_sha256"]
    mask_png = base64.b64decode(result.payload["mask_png_base64"])
    assert set(np.unique(np.asarray(Image.open(io.BytesIO(mask_png))))) <= {0, 255}


def test_preprocessing_follows_manifest_input_block(tmp_path):
    pin = _write_segmenter(tmp_path, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    session = FakeSegSession()
    service = SegmenterService(tmp_path, pin, session_factory=lambda _: session)

    service.segment(_image(value=(255, 255, 255)))

    feed = session.feeds[0]
    assert feed.shape == (1, 3, SIZE, SIZE)
    assert feed.dtype == np.float32
    assert np.allclose(feed, 1.0)


def test_empty_mask_is_reported_as_reason(tmp_path):
    pin = _write_segmenter(tmp_path)
    session = FakeSegSession(logits=np.full((1, 1, SIZE, SIZE), -5.0, dtype=np.float32))
    service = SegmenterService(tmp_path, pin, session_factory=lambda _: session)

    with pytest.raises(SegmenterError) as caught:
        service.segment(_image())
    assert caught.value.reason == "empty_mask"


def test_degenerate_mask_is_reported_as_reason(tmp_path):
    pin = _write_segmenter(tmp_path)
    session = FakeSegSession(logits=np.full((1, 1, SIZE, SIZE), 5.0, dtype=np.float32))
    service = SegmenterService(tmp_path, pin, session_factory=lambda _: session)

    with pytest.raises(SegmenterError) as caught:
        service.segment(_image())
    assert caught.value.reason == "degenerate_mask"


def test_wrong_output_shape_is_segmenter_error(tmp_path):
    pin = _write_segmenter(tmp_path)
    session = FakeSegSession(logits=np.zeros((1, 1, 8, 8), dtype=np.float32))
    service = SegmenterService(tmp_path, pin, session_factory=lambda _: session)

    with pytest.raises(SegmenterError) as caught:
        service.segment(_image())
    assert caught.value.reason == "segmenter_error"


def test_manifest_pin_mismatch_marks_segmenter_unavailable(tmp_path):
    _write_segmenter(tmp_path)
    service = SegmenterService(tmp_path, "0" * 64, session_factory=lambda _: FakeSegSession())

    assert service.status()["ready"] is False
    with pytest.raises(SegmenterError) as caught:
        service.segment(_image())
    assert caught.value.reason == "segmenter_error"


def test_missing_artifacts_are_unavailable_not_fatal(tmp_path):
    service = SegmenterService(tmp_path, "0" * 64)

    assert service.status()["ready"] is False
    assert service.status()["reason"]
