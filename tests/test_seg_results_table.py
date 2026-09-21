"""Comparison table across evaluation runs, built from their summary.json."""

import json

import pytest

import seg_paths  # noqa: F401
import build_results_table as table


def _metric(mean, median=None, p10=None):
    return {"mean": mean, "median": median if median is not None else mean,
            "p10": p10 if p10 is not None else mean, "n": 20}


def _summary(path, *, dice=0.812345, hausdorff=True, strata=True):
    metrics = {
        "dice": _metric(dice, median=dice + 0.02, p10=dice - 0.15),
        "iou": _metric(0.71),
        "sensitivity": _metric(0.884),
        "specificity": _metric(0.9612),
        "pred_coverage": _metric(0.2345),
        "ref_coverage": _metric(0.24),
    }
    if hausdorff:
        metrics["hd95"] = _metric(12.3456)
    payload = {"meta": {"n_images": 20, "eval_name": path.parent.name}, "metrics": metrics}
    if strata:
        payload["strata"] = {
            "min_n": 5,
            "gap_dice": 0.0456,
            "worst_bucket": "5",
            "in_pool": {"n": 20, "status": "OK",
                        "dice": {"mean": 0.8301, "median": 0.84, "n": 20},
                        "iou": {"mean": 0.73, "median": 0.74, "n": 20}},
            "out_of_pool": {"n": 2, "status": "SKIPPED",
                            "dice": {"mean": 0.5, "median": 0.5, "n": 2},
                            "iou": {"mean": 0.4, "median": 0.4, "n": 2}},
            "buckets": {
                "2": {"n": 12, "status": "OK",
                      "dice": {"mean": 0.85, "median": 0.86, "n": 12},
                      "iou": {"mean": 0.75, "median": 0.76, "n": 12}},
                "5": {"n": 8, "status": "OK",
                      "dice": {"mean": 0.8044, "median": 0.81, "n": 8},
                      "iou": {"mean": 0.70, "median": 0.71, "n": 8}},
                "unavailable": {"n": 2, "status": "SKIPPED",
                                "dice": {"mean": 0.5, "median": 0.5, "n": 2},
                                "iou": {"mean": 0.4, "median": 0.4, "n": 2}},
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _run_paths(tmp_path, **kwargs):
    first = _summary(tmp_path / "ssl" / "summary.json", **kwargs)
    second = _summary(tmp_path / "base" / "summary.json", dice=0.6, hausdorff=False, strata=False)
    return first, second


def _build(tmp_path, **kwargs):
    first, second = _run_paths(tmp_path, **kwargs)
    output = tmp_path / "results.md"
    args = table.parse_args(
        ["--run", f"ssl={first}", "--run", f"baseline={second}", "--output", str(output)]
    )
    return output, table.build_table(args)


def test_a_record_carries_the_headline_metrics(tmp_path):
    path = _summary(tmp_path / "ssl" / "summary.json")
    record = table.load_run("ssl", path)
    assert record["run"] == "ssl" and record["n"] == 20
    assert record["dice_mean"] == pytest.approx(0.812345)
    assert record["dice_p10"] == pytest.approx(0.662345)
    assert record["hd95_mean"] == pytest.approx(12.3456)
    assert record["worst_bucket"] == "5" and record["gap_dice"] == pytest.approx(0.0456)


def test_only_qualifying_buckets_reach_the_record(tmp_path):
    path = _summary(tmp_path / "ssl" / "summary.json")
    record = table.load_run("ssl", path)
    assert sorted(record["buckets"]) == ["2", "5"]
    assert record["buckets"]["5"]["dice_mean"] == pytest.approx(0.8044)


def test_a_run_without_hausdorff_or_strata_leaves_those_cells_empty(tmp_path):
    path = _summary(tmp_path / "base" / "summary.json", hausdorff=False, strata=False)
    record = table.load_run("base", path)
    assert record["hd95_mean"] is None
    assert record["worst_bucket"] is None and record["gap_dice"] is None
    assert record["buckets"] == {}


def test_the_markdown_has_one_row_per_run_with_three_decimals(tmp_path):
    output, _ = _build(tmp_path)
    text = output.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.startswith("| ssl ")]
    assert len(lines) == 1
    assert "| 0.812 |" in lines[0] and "| 0.662 |" in lines[0] and "| 12.346 |" in lines[0]
    assert "| 20 |" in lines[0]


def test_a_missing_metric_renders_as_a_blank_cell(tmp_path):
    output, _ = _build(tmp_path)
    row = next(
        line for line in output.read_text(encoding="utf-8").splitlines()
        if line.startswith("| baseline ")
    )
    assert "|  |" in row


def test_the_strata_columns_appear_only_when_some_run_has_buckets(tmp_path):
    output, _ = _build(tmp_path)
    assert "worst_bucket" in output.read_text(encoding="utf-8")

    plain = tmp_path / "plain.md"
    only = _summary(tmp_path / "solo" / "summary.json", strata=False)
    table.build_table(table.parse_args(["--run", f"solo={only}", "--output", str(plain)]))
    assert "worst_bucket" not in plain.read_text(encoding="utf-8")


def test_a_second_table_reports_the_dice_of_each_bucket(tmp_path):
    output, _ = _build(tmp_path)
    text = output.read_text(encoding="utf-8")
    assert "### ssl" in text
    assert "| 5 | 8 | 0.804 | 0.810 |" in text
    assert "unavailable" not in text  # SKIPPED buckets never reach the table


def test_the_json_carries_the_same_records(tmp_path):
    output, payload = _build(tmp_path)
    saved = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved == payload
    assert [record["run"] for record in saved["runs"]] == ["ssl", "baseline"]


def test_a_run_specification_without_a_name_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        table.parse_args(["--run", "sinruta", "--output", str(tmp_path / "r.md")])


def test_at_least_one_run_is_required(tmp_path):
    with pytest.raises(SystemExit):
        table.parse_args(["--output", str(tmp_path / "r.md")])


def test_the_pool_aggregates_reach_the_record(tmp_path):
    record = table.load_run("ssl", _summary(tmp_path / "ssl" / "summary.json"))
    assert record["in_pool_dice_mean"] == pytest.approx(0.8301)
    assert record["out_of_pool_dice_mean"] == pytest.approx(0.5)


def test_a_run_without_strata_reports_no_pool_aggregates(tmp_path):
    record = table.load_run("base", _summary(tmp_path / "base" / "summary.json", strata=False))
    assert record["in_pool_dice_mean"] is None
    assert record["out_of_pool_dice_mean"] is None


def test_the_pool_columns_appear_only_when_a_run_reports_them(tmp_path):
    output, _ = _build(tmp_path)
    text = output.read_text(encoding="utf-8")
    assert "in_pool_dice_mean" in text and "out_of_pool_dice_mean" in text
    row = next(line for line in text.splitlines() if line.startswith("| ssl "))
    assert "| 0.830 |" in row and "| 0.500 |" in row

    plain = tmp_path / "plain.md"
    only = _summary(tmp_path / "solo" / "summary.json", strata=False)
    table.build_table(table.parse_args(["--run", f"solo={only}", "--output", str(plain)]))
    assert "in_pool" not in plain.read_text(encoding="utf-8")


def test_the_dice_confidence_interval_is_rendered_when_present(tmp_path):
    path = _summary(tmp_path / "ci" / "summary.json", dice=0.65)
    payload = json.loads(path.read_text())
    payload["metrics"]["dice"]["ci95_low"] = 0.64
    payload["metrics"]["dice"]["ci95_high"] = 0.66
    path.write_text(json.dumps(payload))
    record = table.load_run("r", path)
    assert record["dice_ci95"] == "0.640-0.660"
    assert "0.640-0.660" in table.build_markdown([record])


def test_the_dice_confidence_interval_is_blank_without_bootstrap(tmp_path):
    record = table.load_run("r", _summary(tmp_path / "noci" / "summary.json"))
    assert record["dice_ci95"] is None
