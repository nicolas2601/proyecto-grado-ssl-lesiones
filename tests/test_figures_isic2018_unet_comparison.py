"""Qualitative U-Net/ISIC2018 comparisons refuse mismatched IDs and masks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_isic2018_unet_comparison as figure  # noqa: E402


def test_verifies_case_id_and_dice_against_binary_masks():
    reference = np.zeros((8, 8), dtype=np.uint8)
    reference[2:6, 2:6] = 1
    case = {"dataset": "isic2018", "image_id": "ISIC_1234567", "role": "mediano",
            "dice": {"student_r3": 1.0}}
    base = "isic2018__ISIC_1234567"
    arrays = {f"{base}__image": np.zeros((8, 8, 3), dtype=np.uint8),
              f"{base}__reference": reference,
              f"{base}__student_r3": reference.copy()}
    image, truth, prediction = figure.verified_case(case, arrays)
    assert image.shape == (8, 8, 3) and np.array_equal(truth, prediction)
    arrays[f"{base}__student_r3"] = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="Dice mismatch"):
        figure.verified_case(case, arrays)


def test_missing_or_nonbinary_masks_are_rejected():
    case = {"dataset": "isic2018", "image_id": "ISIC_1234567", "role": "bueno",
            "dice": {"student_r3": 0.5}}
    base = "isic2018__ISIC_1234567"
    arrays = {f"{base}__image": np.zeros((8, 8, 3), dtype=np.uint8),
              f"{base}__reference": np.zeros((8, 8), dtype=np.uint8)}
    with pytest.raises(ValueError, match="Missing case array"):
        figure.verified_case(case, arrays)
    arrays[f"{base}__student_r3"] = np.full((8, 8), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="binary"):
        figure.verified_case(case, arrays)
