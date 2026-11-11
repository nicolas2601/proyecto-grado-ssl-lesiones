"""Parity of the API segmenter with the release predictions (real ONNX; skipped when artifacts are absent)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageOps

from demo.backend.segmenter_service import SegmenterService

RELEASE_DIR = Path(os.environ.get("SEGMENTER_RELEASE_DIR", Path.home() / "Documentos" / "seg-release-v10.2"))
SAMPLES = RELEASE_DIR / "samples"
MIN_IOU = 0.99


def _artifacts_present() -> bool:
    return (RELEASE_DIR / "segmenter.onnx").is_file() and (RELEASE_DIR / "segmenter_manifest.json").is_file()


pytestmark = pytest.mark.skipif(not _artifacts_present(), reason="real segmenter artifacts are not present")


def _pairs():
    for image in sorted(SAMPLES.glob("*_image.jpg")):
        reference = image.with_name(image.name.replace("_image.jpg", "_mask_pred.png"))
        if reference.is_file():
            yield image, reference


def test_api_masks_match_release_predictions_on_samples():
    manifest = RELEASE_DIR / "segmenter_manifest.json"
    pin = hashlib.sha256(manifest.read_bytes()).hexdigest()
    service = SegmenterService(RELEASE_DIR, pin)
    pairs = list(_pairs())
    assert pairs, "no release sample pairs found"

    for image_path, reference_path in pairs:
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"))
        reference = np.asarray(Image.open(reference_path).convert("L")) > 0
        mask = service.segment(rgb).full_mask

        assert mask.shape == reference.shape, image_path.name
        union = np.logical_or(mask, reference).sum()
        iou = np.logical_and(mask, reference).sum() / union if union else 1.0
        assert iou >= MIN_IOU, f"{image_path.name}: IoU {iou:.5f} below {MIN_IOU}"
