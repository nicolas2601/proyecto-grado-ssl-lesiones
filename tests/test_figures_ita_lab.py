"""CIELAB ITA plots: reconstructed b* coordinates and source thresholds stay auditable."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig2_ita_lab as fig2  # noqa: E402

DATASETS = ("ham10000", "isic2019", "isic2020")
E5 = REPO / "resultados" / "banco_completo" / "e5"


def _rows(dataset: str) -> list[dict]:
    raw = json.loads((E5 / f"e5_fitzpatrick_{dataset}.json").read_text())
    columns = raw["per_image"]["columns"]
    return [dict(zip(columns, row)) for row in raw["per_image"]["rows"]]


def test_reconstruct_b_star_from_reported_ita_and_luminance():
    assert fig2.reconstruct_b_star(70.0, 45.0) == pytest.approx(20.0)
    assert fig2.reconstruct_b_star(50.0, 45.0) is None
    assert fig2.reconstruct_b_star(70.0, 0.0) is None
    assert fig2.reconstruct_b_star(None, 45.0) is None


@pytest.mark.parametrize("dataset", DATASETS)
def test_points_use_only_valid_ita_rows_and_reconstructable_values(dataset):
    rows = _rows(dataset)
    expected = [
        row for row in rows
        if row["roi_status"] == "ok"
        and fig2.reconstruct_b_star(row["roi_L"], row["roi_ita"]) is not None
    ]
    got = fig2.load_dataset_points(REPO, dataset)
    assert got["n_total"] == len(rows)
    assert got["n_ita_valid"] == sum(row["roi_status"] == "ok" for row in rows)
    assert got["n_points"] == len(expected)
    assert got["n_singular"] == got["n_ita_valid"] - got["n_points"]
    assert all(point["b_star"] > 0 for point in got["points"])
    assert all(point["l_star"] == row["roi_L"] for point, row in zip(got["points"], expected, strict=True))


def test_thresholds_are_read_from_each_committed_dataset():
    for dataset in DATASETS:
        raw = json.loads((E5 / f"e5_fitzpatrick_{dataset}.json").read_text())
        assert fig2.thresholds_for(REPO, dataset) == raw["fst_thresholds"]
    assert fig2.thresholds_for(REPO, "ham10000") == [40.0, 23.0, 12.0, 0.0, -25.0]


def test_build_all_writes_one_plot_per_dataset_and_one_combined(tmp_path):
    outputs = fig2.build_all(REPO, tmp_path)
    assert set(outputs) == {*DATASETS, "combined"}
    assert all(len(paths) == 3 for paths in outputs.values())
    assert all(path.is_file() and path.stat().st_size > 0 for paths in outputs.values() for path in paths)
    assert {path.stem for paths in outputs.values() for path in paths} == {
        f"{fig2.STEM}_{name}{suffix}" for name in (*DATASETS, "combined")
        for suffix in ("",)
    }
