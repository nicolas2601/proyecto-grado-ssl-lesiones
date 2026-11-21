from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from demo.backend.analyze_pipeline import PHOTOMETRY_HAIR, preprocessing_of
from demo.backend.segmenter_letterbox import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ROW_CHUNK,
    SOFTMAX_ACTIVATION,
    letterbox_size,
    upsample_threshold,
)
from demo.backend.segmenter_service import SegmenterError, SegmenterService

SIZE = 448
STEPS = (f"letterbox {SIZE} bilinear, pad right-bottom", "float32 /255", "normalize imagenet")


def _write_v2(tmp_path, *, steps=STEPS, activation=SOFTMAX_ACTIVATION, release="v10.4.2"):
    onnx_bytes = b"synthetic letterbox segmenter"
    (tmp_path / "segmenter.onnx").write_bytes(onnx_bytes)
    manifest = {
        "schema": "dermssl.segmenter.v2",
        "release": release,
        "model_id": "dinov2-base-dpt",
        "run": "unified-dpt-s42",
        "onnx": {"file": "segmenter.onnx", "sha256": hashlib.sha256(onnx_bytes).hexdigest(), "opset": 18},
        "input": {"name": "pixel_values", "dtype": "float32", "shape": [1, 3, SIZE, SIZE], "layout": "NCHW",
                  "resize": "bilinear", "preprocessing_required": list(steps)},
        "output": {"name": "logits", "shape": [1, 2, SIZE, SIZE], "dtype": "float32", "activation": activation},
        "threshold": 0.5,
    }
    content = json.dumps(manifest).encode("utf-8")
    (tmp_path / "segmenter_manifest.json").write_bytes(content)
    return hashlib.sha256(content).hexdigest()


class FakeDptSession:
    """Foreground disk inside the valid region and a foreground block inside the padding."""

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
        disk = (yy - 168) ** 2 + (xx - 224) ** 2 <= 80**2
        foreground = np.where(disk, 4.0, -4.0).astype(np.float32)
        foreground[400:, :] = 9.0  # padding rows for a 4:3 landscape image: must be ignored
        return [np.stack([-foreground, foreground])[None]]


def _image(width=640, height=480, value=(255, 255, 255)):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _service(tmp_path, session, **manifest):
    return SegmenterService(tmp_path, _write_v2(tmp_path, **manifest), session_factory=lambda _: session)


def test_v2_manifest_loads_with_letterbox_input_size(tmp_path):
    service = _service(tmp_path, FakeDptSession())

    status = service.status()

    assert status["ready"] is True
    assert status["identity"]["input_size"] == SIZE
    assert status["identity"]["release"] == "v10.4.2"


def test_letterbox_keeps_aspect_and_pads_with_zeros(tmp_path):
    session = FakeDptSession()
    service = _service(tmp_path, session)

    service.segment(_image())

    feed = session.feeds[0]
    assert feed.shape == (1, 3, SIZE, SIZE) and feed.dtype == np.float32
    assert letterbox_size(640, 480, SIZE) == (448, 336)
    white = (1.0 - IMAGENET_MEAN) / IMAGENET_STD
    assert np.allclose(feed[0, :, :336, :448], white[:, None, None], atol=1e-6)
    assert not feed[0, :, 336:, :].any()


def test_padding_logits_do_not_reach_the_mask(tmp_path):
    service = _service(tmp_path, FakeDptSession())

    result = service.segment(_image())

    assert result.full_mask.shape == (480, 640)
    assert not result.full_mask[-20:, :].any()
    assert result.payload["area_fraction"] == pytest.approx(np.pi * 80**2 / (448 * 336), abs=0.02)


def _reference_upsample(probability, width, height):
    """Whole-array bilinear (align_corners=False), as in the release contract script."""
    in_h, in_w = probability.shape
    sy, sx = np.float32(in_h) / np.float32(height), np.float32(in_w) / np.float32(width)
    yc = np.clip((np.arange(height, dtype=np.float32) + np.float32(0.5)) * sy - np.float32(0.5), 0, in_h - 1)
    xc = np.clip((np.arange(width, dtype=np.float32) + np.float32(0.5)) * sx - np.float32(0.5), 0, in_w - 1)
    y0, x0 = np.floor(yc).astype(np.int64), np.floor(xc).astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, in_h - 1), np.minimum(x0 + 1, in_w - 1)
    wy, wx = (yc - y0.astype(np.float32))[:, None], (xc - x0.astype(np.float32))[None, :]
    top = probability[y0[:, None], x0[None, :]] * (1.0 - wx) + probability[y0[:, None], x1[None, :]] * wx
    bottom = probability[y1[:, None], x0[None, :]] * (1.0 - wx) + probability[y1[:, None], x1[None, :]] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32) >= np.float32(0.5)


@pytest.mark.parametrize("width,height", [(640, 480), (333, ROW_CHUNK * 2 + 17), (97, 1201)])
def test_chunked_upsample_matches_whole_array_reference(width, height):
    rng = np.random.default_rng(7)
    probability = rng.random((211, 173), dtype=np.float32)

    assert np.array_equal(upsample_threshold(probability, width, height, 0.5),
                          _reference_upsample(probability, width, height))


@pytest.mark.parametrize("steps", [
    (STEPS[0], STEPS[1]),
    (STEPS[1], STEPS[0], STEPS[2]),
    ("letterbox 224 bilinear, pad right-bottom", STEPS[1], STEPS[2]),
    (STEPS[0], STEPS[1], "normalize clip"),
])
def test_unknown_or_mismatched_letterbox_steps_fail_closed(tmp_path, steps):
    service = _service(tmp_path, FakeDptSession(), steps=steps)

    assert service.status()["ready"] is False


def test_wrong_activation_fails_closed(tmp_path):
    service = _service(tmp_path, FakeDptSession(), activation="apply sigmoid outside the graph")

    assert service.status()["ready"] is False


def test_single_channel_logits_are_segmenter_error(tmp_path):
    session = FakeDptSession(logits=np.zeros((1, 1, SIZE, SIZE), dtype=np.float32))
    service = _service(tmp_path, session)

    with pytest.raises(SegmenterError) as caught:
        service.segment(_image())
    assert caught.value.reason == "segmenter_error"


def test_photometry_keeps_hair_exclusion_with_letterbox_segmenter(tmp_path):
    service = _service(tmp_path, FakeDptSession())

    config = preprocessing_of(service)

    assert config is PHOTOMETRY_HAIR
    assert config.size == 224 and config.hair is not None
