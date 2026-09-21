"""Evaluating third-party masks (baselines) with no checkpoint in the loop."""

import csv
import json

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import eval_inputs
import evaluate_segmentation as ev

SIZE = 16


def _square(size=SIZE, top=2, left=2, side=8):
    mask = np.zeros((size, size), dtype=bool)
    mask[top : top + side, left : left + side] = True
    return mask


def _save_mask(path, mask):
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def _write_case(tmp_path, *, shift=0, count=2, suffix=""):
    """Images, reference masks and predicted PNGs for ``count`` fake cases."""
    images, references, predictions = tmp_path / "img", tmp_path / "gt", tmp_path / "pred"
    for folder in (images, references, predictions):
        folder.mkdir()
    for index in range(count):
        image_id = f"CASE_{index}"
        Image.fromarray(np.full((SIZE, SIZE, 3), 180, np.uint8)).save(images / f"{image_id}.png")
        _save_mask(references / f"{image_id}_segmentation.png", _square())
        _save_mask(predictions / f"{image_id}{suffix}.png", _square(left=2 + shift))
    return images, references, predictions


def _argv(tmp_path, images, references, predictions, *extra):
    return [
        "--pred-mask-dir", str(predictions),
        "--image-dir", str(images),
        "--mask-dir", str(references),
        "--output-dir", str(tmp_path / "out"),
        "--eval-resolution", str(SIZE),
        *extra,
    ]


def test_external_masks_produce_the_same_report_as_a_checkpoint_run(tmp_path):
    images, references, predictions = _write_case(tmp_path)
    summary = ev.evaluate(ev.parse_args(_argv(tmp_path, images, references, predictions)))
    assert summary["dice"]["mean"] == pytest.approx(1.0)

    output = tmp_path / "out"
    with (output / "per_image.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["image_id"] for row in rows] == ["CASE_0", "CASE_1"]
    assert float(rows[0]["iou"]) == pytest.approx(1.0)

    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["meta"]["predictions"] == "external"
    assert payload["meta"]["n_images"] == 2
    assert payload["meta"]["resolution"] == SIZE
    assert payload["meta"]["threshold"] == 0.5


def test_a_shifted_external_mask_scores_the_hand_computed_dice(tmp_path):
    images, references, predictions = _write_case(tmp_path, shift=4)
    summary = ev.evaluate(ev.parse_args(_argv(tmp_path, images, references, predictions)))
    overlap = 8 * 4  # 8x8 squares, four columns apart
    assert summary["dice"]["mean"] == pytest.approx(2 * overlap / (64 + 64))


def test_external_predictions_accept_a_suffix_and_an_extension(tmp_path):
    images, references, predictions = _write_case(tmp_path, suffix="_pred")
    argv = _argv(
        tmp_path, images, references, predictions,
        "--pred-mask-suffix", "_pred", "--pred-mask-extension", ".png",
    )
    assert ev.evaluate(ev.parse_args(argv))["dice"]["mean"] == pytest.approx(1.0)


def test_a_missing_external_prediction_names_the_flag(tmp_path):
    images, references, predictions = _write_case(tmp_path)
    (predictions / "CASE_1.png").unlink()
    with pytest.raises(FileNotFoundError, match="--pred-mask-suffix"):
        ev.evaluate(ev.parse_args(_argv(tmp_path, images, references, predictions)))


def test_external_masks_still_produce_strata_and_overlays(tmp_path):
    images, references, predictions = _write_case(tmp_path)
    manifest = tmp_path / "fst.csv"
    manifest.write_text("image_id,fst_bucket\nCASE_0,2\nCASE_1,2\n", encoding="utf-8")
    argv = _argv(
        tmp_path, images, references, predictions,
        "--fst-manifest", str(manifest), "--fst-min-n", "1",
        "--overlay-dir", str(tmp_path / "ov"), "--overlay-count", "2",
    )
    ev.evaluate(ev.parse_args(argv))
    strata = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))["strata"]
    assert strata["buckets"]["2"]["status"] == "OK"
    assert sorted(p.name for p in (tmp_path / "ov").iterdir()) == [
        "best.png", "random.png", "worst.png"
    ]


def test_hausdorff_is_available_for_external_masks(tmp_path):
    images, references, predictions = _write_case(tmp_path, shift=2)
    argv = _argv(tmp_path, images, references, predictions, "--hausdorff")
    summary = ev.evaluate(ev.parse_args(argv))
    assert summary["hd95"]["mean"] == pytest.approx(2.0, abs=0.5)


def test_a_checkpoint_and_an_external_mask_directory_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        ev.parse_args(
            ["--checkpoint", "c.pt", "--pred-mask-dir", "p",
             "--image-dir", "i", "--mask-dir", "m", "--output-dir", "o"]
        )


def test_one_prediction_source_is_mandatory():
    with pytest.raises(SystemExit):
        ev.parse_args(["--image-dir", "i", "--mask-dir", "m", "--output-dir", "o"])


def test_the_external_mode_falls_back_to_the_default_resolution():
    args = ev.parse_args(
        ["--pred-mask-dir", "p", "--image-dir", "i", "--mask-dir", "m", "--output-dir", "o"]
    )
    assert args.eval_resolution == ev.EXTERNAL_RESOLUTION == 224
    assert args.pred_mask_suffix == "" and args.pred_mask_extension == ".png"


def test_external_predictions_are_binarized_above_127_and_resized_with_nearest(tmp_path):
    path = tmp_path / "p.png"
    array = np.zeros((4, 4), np.uint8)
    array[:, 2:] = 200
    array[0, 0] = 127  # exactly at the cut-off: still background
    Image.fromarray(array).save(path)
    mask = eval_inputs.load_binary_mask(path, 8)
    assert mask.shape == (8, 8) and mask.dtype == bool
    assert mask[:, 4:].all() and not mask[:, :4].any()


def test_pairing_predictions_keeps_the_manual_mask_pairs(tmp_path):
    images, references, predictions = _write_case(tmp_path)
    pairs = eval_inputs.pair_images_and_masks(images, references)
    found = eval_inputs.pair_predictions(pairs, predictions, "", ".png")
    assert sorted(found) == ["CASE_0", "CASE_1"]
    assert found["CASE_0"].name == "CASE_0.png"
