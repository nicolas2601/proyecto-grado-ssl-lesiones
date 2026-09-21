"""Pure tests for the evaluation report helpers (percentiles, strata, overlays)."""

import argparse
import csv
import json

import numpy as np
import pytest

import seg_paths  # noqa: F401
import eval_report as er
import eval_tracking as et


def _rows(dices, prefix="img"):
    return [{"image_id": f"{prefix}{i}", "dice": d, "iou": d / 2} for i, d in enumerate(dices)]


# ------------------------------------------------------------- percentiles
def test_summarize_reports_the_five_extra_percentiles():
    rows = _rows([i / 10 for i in range(11)])
    summary = er.summarize(rows, ["dice"])
    assert summary["dice"]["p5"] == pytest.approx(0.05)
    assert summary["dice"]["p10"] == pytest.approx(0.10)
    assert summary["dice"]["p25"] == pytest.approx(0.25)
    assert summary["dice"]["p75"] == pytest.approx(0.75)
    assert summary["dice"]["p90"] == pytest.approx(0.90)


def test_summarize_percentiles_are_nan_without_values():
    summary = er.summarize([{"dice": float("nan")}], ["dice"])
    assert np.isnan(summary["dice"]["p5"]) and summary["dice"]["n"] == 0


def test_summarize_lists_the_ten_lowest_dice_image_ids_in_order():
    rows = _rows([i / 100 for i in range(12)])
    summary = er.summarize(rows, ["dice"])
    assert summary["worst_ids"] == [f"img{i}" for i in range(10)]


def test_summarize_omits_worst_ids_when_rows_have_no_image_id():
    summary = er.summarize([{"dice": 0.5}], ["dice"])
    assert "worst_ids" not in summary


# ------------------------------------------------------------------ strata
def test_read_fst_manifest_keeps_buckets_and_the_unavailable_marker(tmp_path):
    path = tmp_path / "fst.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "fst_bucket"])
        writer.writerows([["a", "1"], ["b", "6"], ["c", "unavailable"]])
    assert er.read_fst_manifest(path) == {"a": "1", "b": "6", "c": "unavailable"}


def test_read_fst_manifest_rejects_a_bucket_outside_one_to_six(tmp_path):
    path = tmp_path / "fst.csv"
    path.write_text("image_id,fst_bucket\na,9\n", encoding="utf-8")
    with pytest.raises(ValueError):
        er.read_fst_manifest(path)


def test_strata_marks_small_buckets_as_skipped_and_ignores_them_in_the_gap():
    rows = _rows([0.9] * 30, prefix="a") + _rows([0.5] * 30, prefix="b") + _rows([0.1] * 5, prefix="c")
    buckets = {}
    buckets.update({f"a{i}": "1" for i in range(30)})
    buckets.update({f"b{i}": "5" for i in range(30)})
    buckets.update({f"c{i}": "6" for i in range(5)})
    strata = er.stratified_summary(rows, buckets, min_n=30)
    assert strata["buckets"]["1"]["status"] == "OK"
    assert strata["buckets"]["6"]["status"] == "SKIPPED"
    assert strata["buckets"]["1"]["dice"]["mean"] == pytest.approx(0.9)
    assert strata["gap_dice"] == pytest.approx(0.4)
    assert strata["worst_bucket"] == "5"


def test_strata_without_qualifying_buckets_reports_no_gap():
    rows = _rows([0.4] * 3)
    strata = er.stratified_summary(rows, {r["image_id"]: "2" for r in rows}, min_n=30)
    assert strata["gap_dice"] is None and strata["worst_bucket"] is None


def test_strata_sends_unknown_images_to_the_unavailable_bucket():
    rows = _rows([0.4, 0.6])
    strata = er.stratified_summary(rows, {"img0": "3"}, min_n=1)
    assert strata["buckets"]["unavailable"]["n"] == 1
    assert "unavailable" not in {strata["worst_bucket"]}


def test_strata_table_rows_are_flat_and_ordered():
    rows = _rows([0.4] * 2)
    strata = er.stratified_summary(rows, {r["image_id"]: "2" for r in rows}, min_n=1)
    table = et.strata_table_rows(strata)
    assert table[0]["bucket"] == "2" and table[0]["status"] == "OK"
    assert table[0]["dice_mean"] == pytest.approx(0.4)


# ------------------------------------------------------------- flattening
def test_flatten_summary_prefixes_every_metric_statistic():
    flat = et.flatten_summary({"dice": {"mean": 0.5, "n": 3}, "worst_ids": ["a"]}, "isic")
    assert flat == {"isic/dice_mean": 0.5, "isic/dice_n": 3}


# ----------------------------------------------------------------- writers
def test_report_writers_include_the_strata_block(tmp_path):
    path = tmp_path / "summary.json"
    er.write_summary_json(path, {"dice": {"mean": 0.8}}, {"n_images": 1}, strata={"gap_dice": 0.1})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["strata"]["gap_dice"] == 0.1
    assert payload["metrics"]["dice"]["mean"] == 0.8


def test_summary_json_has_no_strata_key_when_not_stratified(tmp_path):
    path = tmp_path / "summary.json"
    er.write_summary_json(path, {}, {})
    assert "strata" not in json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------- overlays
def _record(image_id, dice, side=8):
    image = np.full((side, side, 3), 30, dtype=np.uint8)
    reference = np.zeros((side, side), dtype=bool)
    reference[1:5, 1:5] = True
    prediction = np.zeros((side, side), dtype=bool)
    prediction[2:6, 2:6] = True
    return {
        "image_id": image_id,
        "dice": dice,
        "image": image,
        "reference": reference,
        "prediction": prediction,
    }


def test_select_overlay_groups_orders_worst_and_best_by_dice():
    rows = _rows([0.9, 0.1, 0.5, 0.7])
    groups = er.select_overlay_groups(rows, count=2, seed=0)
    assert groups["worst"] == ["img1", "img2"]
    assert groups["best"] == ["img0", "img3"]
    assert len(groups["random"]) == 2


def test_select_overlay_groups_is_reproducible_for_a_given_seed():
    rows = _rows([i / 20 for i in range(20)])
    first = er.select_overlay_groups(rows, count=5, seed=0)["random"]
    assert first == er.select_overlay_groups(rows, count=5, seed=0)["random"]


def test_select_overlay_groups_handles_fewer_images_than_requested():
    groups = er.select_overlay_groups(_rows([0.3]), count=12, seed=0)
    assert groups["worst"] == groups["best"] == groups["random"] == ["img0"]


def test_contour_keeps_only_the_border_of_a_filled_square():
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:5, 1:5] = True
    border = er.contour(mask)
    assert border.sum() == 12
    assert not border[2, 2] and border[1, 1]


def test_overlay_paints_reference_in_green_and_prediction_in_red():
    record = _record("a", 0.5)
    painted = er.overlay_image(record["image"], record["reference"], record["prediction"])
    assert tuple(painted[1, 1]) == (0, 255, 0)
    assert tuple(painted[5, 5]) == (255, 0, 0)
    assert tuple(painted[7, 7]) == (30, 30, 30)


def test_write_overlay_grid_produces_one_png_with_four_columns(tmp_path):
    from PIL import Image

    path = tmp_path / "worst.png"
    er.write_overlay_grid(path, [_record("a", 0.2), _record("b", 0.4)], cell_size=32)
    assert path.is_file()
    with Image.open(path) as rendered:
        assert rendered.width == 4 * 32
        assert rendered.height > 2 * 32  # room for the per-row titles


def test_write_overlay_grid_refuses_an_empty_group(tmp_path):
    with pytest.raises(ValueError):
        er.write_overlay_grid(tmp_path / "empty.png", [], cell_size=32)


# ----------------------------------------------------------------- tracking
def test_tracking_is_a_no_op_without_a_project(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    args = argparse.Namespace(wandb_project="", wandb_run_name="", eval_name="referencia")
    assert et.track_evaluation(args, {"dice": {"mean": 0.5}}, None, _rows([0.5]), []) is False


# ------------------------------------ the manifest written by the FST stage
def _fst_json(tmp_path, records):
    """Same shape as fst_pseudo_manifest.run: buckets are ints or "unavailable"."""
    path = tmp_path / "fst_manifest.json"
    payload = {
        "schema_version": 1,
        "field_used": True,
        "records": records,
        "summary": {"total": len(records)},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_read_fst_manifest_accepts_the_json_written_by_the_fst_stage(tmp_path):
    path = _fst_json(
        tmp_path,
        [
            {"image_id": "a", "dataset": "d", "ITA": 41.2, "FST_bucket": 2, "status": "ok"},
            {"image_id": "b", "dataset": "d", "ITA": None, "FST_bucket": "unavailable",
             "status": "quarantine"},
        ],
    )
    assert er.read_fst_manifest(path) == {"a": "2", "b": "unavailable"}


def test_read_fst_manifest_rejects_an_invalid_bucket_in_the_json(tmp_path):
    path = _fst_json(tmp_path, [{"image_id": "a", "FST_bucket": 7}])
    with pytest.raises(ValueError):
        er.read_fst_manifest(path)


def test_read_fst_manifest_accepts_the_uppercase_column_in_a_csv(tmp_path):
    path = tmp_path / "fst.csv"
    path.write_text("image_id,FST_bucket\na,3\nb,unavailable\n", encoding="utf-8")
    assert er.read_fst_manifest(path) == {"a": "3", "b": "unavailable"}


def test_read_fst_manifest_reports_a_row_without_any_bucket_column(tmp_path):
    path = _fst_json(tmp_path, [{"image_id": "a", "ITA": 1.0}])
    with pytest.raises(ValueError, match="fst_bucket"):
        er.read_fst_manifest(path)


def test_the_fst_json_feeds_the_stratified_summary_end_to_end(tmp_path):
    records = [{"image_id": f"img{i}", "FST_bucket": 1 if i < 2 else "unavailable"}
               for i in range(3)]
    buckets = er.read_fst_manifest(_fst_json(tmp_path, records))
    strata = er.stratified_summary(_rows([0.8, 0.6, 0.1]), buckets, min_n=2)
    assert strata["buckets"]["1"]["status"] == "OK"
    assert strata["buckets"]["1"]["dice"]["mean"] == pytest.approx(0.7)
    assert strata["buckets"]["unavailable"]["n"] == 1
    assert strata["worst_bucket"] == "1"


# --------------------------------------------- NaN is null, not bare NaN
def test_summary_json_writes_null_instead_of_a_bare_nan(tmp_path):
    path = tmp_path / "summary.json"
    er.write_summary_json(
        path,
        {"dice": {"mean": float("nan"), "n": 0}},
        {"n_images": 0},
        strata={"gap_dice": None, "buckets": {"1": {"dice": {"mean": float("nan")}}}},
    )
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    payload = json.loads(text)
    assert payload["metrics"]["dice"]["mean"] is None
    assert payload["strata"]["buckets"]["1"]["dice"]["mean"] is None


def test_summary_json_keeps_finite_numbers_and_lists(tmp_path):
    path = tmp_path / "summary.json"
    er.write_summary_json(path, {"dice": {"mean": 0.5}, "worst_ids": ["a", "b"]}, {"n": 2})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"]["dice"]["mean"] == 0.5
    assert payload["metrics"]["worst_ids"] == ["a", "b"]


# ------------------------------------------------ a missing metric is NaN
def test_summarize_treats_a_missing_metric_as_nan():
    rows = [{"image_id": "a", "dice": 0.5}, {"image_id": "b"}]
    summary = er.summarize(rows, ["dice"])
    assert summary["dice"]["n"] == 1 and summary["dice"]["mean"] == pytest.approx(0.5)


def test_strata_treats_a_missing_metric_as_nan():
    rows = [{"image_id": "a", "dice": 0.5, "iou": 0.4}, {"image_id": "b", "dice": 0.7}]
    strata = er.stratified_summary(rows, {"a": "2", "b": "2"}, min_n=1)
    assert strata["buckets"]["2"]["dice"]["n"] == 2
    assert strata["buckets"]["2"]["iou"]["n"] == 1


def test_worst_ids_ignores_rows_without_a_dice_value():
    assert er.worst_image_ids([{"image_id": "a"}, {"image_id": "b", "dice": 0.2}]) == ["b"]


# --------------------------------------------------- manifest validation
def test_validate_fst_manifest_rejects_a_missing_file(tmp_path):
    with pytest.raises(ValueError, match="no existe"):
        er.validate_fst_manifest(tmp_path / "nope.csv")


def test_validate_fst_manifest_rejects_a_file_without_the_bucket_column(tmp_path):
    path = tmp_path / "fst.csv"
    path.write_text("image_id,ITA\na,12.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fst_bucket"):
        er.validate_fst_manifest(path)


def test_validate_fst_manifest_rejects_a_file_without_image_id(tmp_path):
    path = tmp_path / "fst.csv"
    path.write_text("name,fst_bucket\na,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image_id"):
        er.validate_fst_manifest(path)


def test_validate_fst_manifest_rejects_an_empty_manifest(tmp_path):
    path = tmp_path / "fst.csv"
    path.write_text("image_id,fst_bucket\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sin filas"):
        er.validate_fst_manifest(path)


def test_validate_fst_manifest_accepts_both_formats(tmp_path):
    csv_path = tmp_path / "fst.csv"
    csv_path.write_text("image_id,fst_bucket\na,2\n", encoding="utf-8")
    json_path = _fst_json(tmp_path, [{"image_id": "a", "FST_bucket": 2}])
    assert er.validate_fst_manifest(csv_path) is None
    assert er.validate_fst_manifest(json_path) is None


def test_strata_from_args_returns_none_without_a_manifest():
    args = argparse.Namespace(fst_manifest="", fst_min_n=30)
    assert er.strata_from_args(args, _rows([0.5])) is None


def test_strata_from_args_reads_the_manifest_and_applies_min_n(tmp_path):
    path = tmp_path / "fst.csv"
    path.write_text("image_id,fst_bucket\nimg0,4\nimg1,4\n", encoding="utf-8")
    args = argparse.Namespace(fst_manifest=str(path), fst_min_n=2)
    strata = er.strata_from_args(args, _rows([0.2, 0.8]))
    assert strata["buckets"]["4"]["status"] == "OK"
    assert strata["buckets"]["4"]["dice"]["mean"] == pytest.approx(0.5)


def test_reject_duplicate_ids_names_the_repeated_ones():
    er.reject_duplicate_ids(["a", "b"])
    with pytest.raises(ValueError, match="b"):
        er.reject_duplicate_ids(["a", "b", "b"])
