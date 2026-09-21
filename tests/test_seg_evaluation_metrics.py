"""Tiny synthetic-array tests for the evaluation module (masks are read only here)."""

import csv
import json
import math

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import evaluate_segmentation as ev


def _square(size=8, top=2, left=2, side=4):
    mask = np.zeros((size, size), dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


def test_perfect_prediction_scores_one():
    gt = _square()
    metrics = ev.compute_binary_metrics(gt, gt)
    assert metrics["dice"] == pytest.approx(1.0)
    assert metrics["iou"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["sensitivity"] == pytest.approx(1.0)
    assert metrics["specificity"] == pytest.approx(1.0)


def test_half_overlap_metrics_match_hand_computation():
    gt = _square(top=2, left=2, side=4)  # 16 px
    pred = _square(top=2, left=4, side=4)  # 16 px, overlap 8
    metrics = ev.compute_binary_metrics(pred, gt)
    assert metrics["dice"] == pytest.approx(2 * 8 / 32)
    assert metrics["iou"] == pytest.approx(8 / 24)
    assert metrics["sensitivity"] == pytest.approx(8 / 16)
    assert metrics["specificity"] == pytest.approx((64 - 24) / (64 - 16))
    assert metrics["accuracy"] == pytest.approx((8 + 40) / 64)


def test_all_background_prediction_has_zero_dice_and_full_specificity():
    gt = _square()
    metrics = ev.compute_binary_metrics(np.zeros_like(gt), gt)
    assert metrics["dice"] == 0.0
    assert metrics["sensitivity"] == 0.0
    assert metrics["specificity"] == 1.0


def test_both_empty_counts_as_perfect_agreement_and_undefined_sensitivity():
    empty = np.zeros((4, 4), dtype=bool)
    metrics = ev.compute_binary_metrics(empty, empty)
    assert metrics["dice"] == 1.0 and metrics["iou"] == 1.0
    assert math.isnan(metrics["sensitivity"])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ev.compute_binary_metrics(np.zeros((4, 4), bool), np.zeros((5, 5), bool))


def test_hausdorff95_of_identical_masks_is_zero():
    gt = _square()
    assert ev.hausdorff95(gt, gt) == pytest.approx(0.0)


def test_hausdorff95_of_shifted_square_is_the_shift():
    gt = _square(size=16, top=4, left=4, side=6)
    pred = _square(size=16, top=4, left=6, side=6)
    assert ev.hausdorff95(pred, gt) == pytest.approx(2.0, abs=0.5)


def test_hausdorff95_is_nan_when_a_mask_is_empty():
    assert math.isnan(ev.hausdorff95(np.zeros((4, 4), bool), _square(4, 1, 1, 2)))


def test_threshold_is_fixed_at_half_and_strict():
    assert ev.DEFAULT_THRESHOLD == 0.5
    probs = np.array([[0.49, 0.5, 0.51]])
    assert ev.binarize(probs).tolist() == [[False, False, True]]


def test_summarize_ignores_nan_and_reports_counts():
    rows = [{"dice": 0.5, "sensitivity": float("nan")}, {"dice": 1.0, "sensitivity": 0.7}]
    summary = ev.summarize(rows, ["dice", "sensitivity"])
    assert summary["dice"]["mean"] == pytest.approx(0.75)
    assert summary["dice"]["n"] == 2
    assert summary["sensitivity"]["mean"] == pytest.approx(0.7)
    assert summary["sensitivity"]["n"] == 1


def test_pair_images_and_masks_uses_isic_suffix_and_flags_missing(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    images.mkdir(), masks.mkdir()
    for name in ("ISIC_1", "ISIC_2"):
        Image.new("RGB", (4, 4)).save(images / f"{name}.jpg")
    Image.new("L", (4, 4)).save(masks / "ISIC_1_segmentation.png")
    with pytest.raises(FileNotFoundError):
        ev.pair_images_and_masks(images, masks)
    Image.new("L", (4, 4)).save(masks / "ISIC_2_segmentation.png")
    pairs = ev.pair_images_and_masks(images, masks)
    assert [p[0] for p in pairs] == ["ISIC_1", "ISIC_2"]


def test_load_mask_resizes_with_nearest_and_binarizes(tmp_path):
    path = tmp_path / "m.png"
    array = np.zeros((4, 4), np.uint8)
    array[:, 2:] = 255
    Image.fromarray(array).save(path)
    mask = ev.load_mask(path, 8)
    assert mask.shape == (8, 8) and mask.dtype == bool
    assert mask[:, 4:].all() and not mask[:, :4].any()


def test_report_writers_produce_csv_and_json(tmp_path):
    rows = [{"image_id": "a", "dice": 0.9}, {"image_id": "b", "dice": 0.7}]
    csv_path, json_path = tmp_path / "per_image.csv", tmp_path / "summary.json"
    ev.write_per_image_csv(csv_path, rows, ["dice"])
    ev.write_summary_json(json_path, {"dice": {"mean": 0.8}}, {"n_images": 2})
    with csv_path.open(encoding="utf-8") as handle:
        assert [r["image_id"] for r in csv.DictReader(handle)] == ["a", "b"]
    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"]["dice"]["mean"] == 0.8


def test_evaluation_arguments_parse_without_torch():
    args = ev.parse_args(
        ["--checkpoint", "c.pt", "--image-dir", "i", "--mask-dir", "m", "--output-dir", "o"]
    )
    assert args.mask_suffix == "_segmentation"
