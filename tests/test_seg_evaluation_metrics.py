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


def test_coverage_columns_report_the_lesion_fraction_of_each_mask():
    gt = _square(size=8, top=2, left=2, side=4)  # 16 of 64 pixels
    pred = np.zeros((8, 8), dtype=bool)
    pred[0:2, 0:4] = True  # 8 of 64 pixels
    metrics = ev.compute_binary_metrics(pred, gt)
    assert metrics["pred_coverage"] == pytest.approx(8 / 64)
    assert metrics["ref_coverage"] == pytest.approx(16 / 64)


def test_coverage_is_zero_for_an_empty_prediction():
    metrics = ev.compute_binary_metrics(np.zeros((4, 4), bool), np.zeros((4, 4), bool))
    assert metrics["pred_coverage"] == 0.0 and metrics["ref_coverage"] == 0.0


def test_evaluation_defaults_cover_strata_overlays_and_tracking():
    args = ev.parse_args(
        ["--checkpoint", "c.pt", "--image-dir", "i", "--mask-dir", "m", "--output-dir", "o"]
    )
    assert args.eval_name == "referencia"
    assert args.fst_manifest == "" and args.fst_min_n == 30
    assert args.overlay_dir == "" and args.overlay_count == 12
    assert args.wandb_project == "" and args.wandb_run_name == ""


def test_evaluation_accepts_the_strata_and_overlay_flags(tmp_path):
    manifest = tmp_path / "f.csv"
    manifest.write_text("image_id,fst_bucket\na,2\n", encoding="utf-8")
    args = ev.parse_args(
        ["--checkpoint", "c.pt", "--image-dir", "i", "--mask-dir", "m", "--output-dir", "o",
         "--fst-manifest", str(manifest), "--fst-min-n", "5", "--overlay-dir", "ov",
         "--overlay-count", "3", "--eval-name", "isic", "--wandb-project", "p"]
    )
    assert args.fst_manifest == str(manifest) and args.fst_min_n == 5
    assert args.overlay_dir == "ov" and args.overlay_count == 3
    assert args.eval_name == "isic" and args.wandb_project == "p"


def test_the_metric_columns_include_coverage_and_optional_hausdorff():
    assert ev.metric_columns(hausdorff=False)[-2:] == ["pred_coverage", "ref_coverage"]
    assert ev.metric_columns(hausdorff=True)[-1] == "hd95"


def test_duplicate_image_ids_fail_loudly(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    images.mkdir(), masks.mkdir()
    Image.new("RGB", (4, 4)).save(images / "ISIC_1.jpg")
    Image.new("RGB", (4, 4)).save(images / "ISIC_1.png")
    Image.new("L", (4, 4)).save(masks / "ISIC_1_segmentation.png")
    with pytest.raises(ValueError, match="ISIC_1"):
        ev.pair_images_and_masks(images, masks)


def test_unique_image_ids_are_accepted(tmp_path):
    images, masks = tmp_path / "i", tmp_path / "m"
    images.mkdir(), masks.mkdir()
    for name in ("ISIC_1", "ISIC_2"):
        Image.new("RGB", (4, 4)).save(images / f"{name}.jpg")
        Image.new("L", (4, 4)).save(masks / f"{name}_segmentation.png")
    assert [p[0] for p in ev.pair_images_and_masks(images, masks)] == ["ISIC_1", "ISIC_2"]


def _base_argv(tmp_path):
    return ["--checkpoint", "c.pt", "--image-dir", "i", "--mask-dir", "m",
            "--output-dir", str(tmp_path / "o")]


def test_a_missing_fst_manifest_is_rejected_before_any_inference(tmp_path):
    with pytest.raises(SystemExit):
        ev.parse_args([*_base_argv(tmp_path), "--fst-manifest", str(tmp_path / "nope.csv")])


def test_an_fst_manifest_without_the_bucket_column_is_rejected(tmp_path):
    path = tmp_path / "fst.csv"
    path.write_text("image_id,ITA\na,1.0\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        ev.parse_args([*_base_argv(tmp_path), "--fst-manifest", str(path)])


def test_a_valid_fst_manifest_passes_argument_parsing(tmp_path):
    path = tmp_path / "fst.csv"
    path.write_text("image_id,fst_bucket\na,2\n", encoding="utf-8")
    args = ev.parse_args([*_base_argv(tmp_path), "--fst-manifest", str(path)])
    assert args.fst_manifest == str(path)
