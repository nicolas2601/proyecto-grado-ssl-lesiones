"""Individual ROI-ITA figures retain abstention and never imply validated FST."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_obj1_ita_estimates as figure  # noqa: E402


def test_class_counts_and_ita_values_share_only_valid_rows(tmp_path):
    source = tmp_path / "resultados/banco_completo/e5"
    source.mkdir(parents=True)
    (source / "e5_fitzpatrick_ham10000.json").write_text(json.dumps({
        "fst_thresholds": [40, 23, 12, 0, -25],
        "per_image": {"columns": ["image_id", "roi_status", "roi_bucket", "roi_ita"], "rows": [
            ["a", "ok", "I", 52.0], ["b", "ok", "VI", -27.0],
            ["c", "ita_indeterminate", None, None],
        ]},
    }), encoding="utf-8")
    data = figure.load_dataset(tmp_path, "ham10000")
    assert data["n_total"] == 3 and data["n_valid"] == 2 and data["n_abstention"] == 1
    assert data["counts"] == {"I": 1, "II": 0, "III": 0, "IV": 0, "V": 0, "VI": 1}
    assert data["values"]["I"] == [52.0] and data["values"]["VI"] == [-27.0]


def test_duplicate_or_inconsistent_bucket_aborts(tmp_path):
    source = tmp_path / "resultados/banco_completo/e5"
    source.mkdir(parents=True)
    target = source / "e5_fitzpatrick_isic2019.json"
    for rows in ([ ["a", "ok", "I", 42.0], ["a", "ok", "II", 35.0] ],
                 [ ["a", "ok", None, 55.0] ]):
        target.write_text(json.dumps({"fst_thresholds": [40, 23, 12, 0, -25], "per_image": {
            "columns": ["image_id", "roi_status", "roi_bucket", "roi_ita"], "rows": rows,
        }}), encoding="utf-8")
        with pytest.raises(ValueError):
            figure.load_dataset(tmp_path, "isic2019")


@pytest.mark.parametrize("dataset,total,valid,missing", [
    ("ham10000", 10015, 4539, 5476),
    ("isic2019", 25331, 12558, 12773),
    ("isic2020", 33126, 21973, 11153),
])
def test_versioned_e5_totals(dataset, total, valid, missing):
    data = figure.load_dataset(REPO, dataset)
    assert (data["n_total"], data["n_valid"], data["n_abstention"]) == (total, valid, missing)
    assert sum(data["counts"].values()) == valid


def test_export_three_separate_svg_figures_with_matching_counts(tmp_path):
    written = figure.build_all(REPO, tmp_path)
    assert set(written) == {"ham10000", "isic2019", "isic2020"}
    assert all({p.suffix for p in paths} == {".png", ".pdf", ".svg"} for paths in written.values())
    for dataset, paths in written.items():
        assert all(path.is_file() and path.stat().st_size for path in paths)
        svg = next(path.read_text(encoding="utf-8") for path in paths if path.suffix == ".svg")
        assert "ITA estimado" in svg and "no validado" in svg
        assert dataset in next(path.name for path in paths if path.suffix == ".svg")
