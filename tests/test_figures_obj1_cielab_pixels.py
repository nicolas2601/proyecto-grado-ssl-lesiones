"""Pixel-level CIELAB panels must count the same sampled pixels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_obj1_cielab_pixels as figure  # noqa: E402


def _images(tmp_path: Path) -> tuple[Path, Path]:
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for image_id in ("a", "b"):
        rgb = np.full((4, 4, 3), (160, 120, 95), dtype=np.uint8)
        rgb[0, 0] = 0
        Image.fromarray(rgb).save(images / f"{image_id}.png")
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 1] = 255
        Image.fromarray(mask).save(masks / f"{image_id}_segmentation.png")
    return images, masks


def test_same_sampled_pixels_feed_luminance_and_chrominance(tmp_path):
    images, masks = _images(tmp_path)
    data = figure.collect(images, "outside-lesion", 4, masks, 5, 42)
    assert data["n_images"] == 2
    assert data["n_pixels"] == 10
    assert data["l_hist"].sum() == data["ab_hist"].sum() == 10
    assert data["n_without_pixels"] == 0
    assert np.array_equal(data["l_hist"], figure.collect(images, "outside-lesion", 4, masks, 5, 42)["l_hist"])


def test_missing_mask_aborts_instead_of_silently_changing_population(tmp_path):
    images, masks = _images(tmp_path)
    (masks / "b_segmentation.png").unlink()
    with pytest.raises(FileNotFoundError, match="b"):
        figure.collect(images, "outside-lesion", 4, masks, 5, 42)


def test_population_requires_explicit_valid_mode_and_mask(tmp_path):
    images, _ = _images(tmp_path)
    with pytest.raises(ValueError, match="mask"):
        figure.collect(images, "outside-lesion", 4, None, 0, 42)
    with pytest.raises(ValueError, match="population"):
        figure.collect(images, "unknown", 4, None, 0, 42)


def test_build_exports_three_formats_with_reproducibility_metadata(tmp_path):
    images, masks = _images(tmp_path)
    paths = figure.build(images, "outside-lesion", 4, masks, 5, 42, tmp_path / "out")
    assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
    summary = json.loads((tmp_path / "out" / "fig_obj1_cielab_pixels.summary.json").read_text())
    assert summary["n_pixels"] == 10
    assert summary["population"] == "outside-lesion"
    assert summary["sample_per_image"] == 5
