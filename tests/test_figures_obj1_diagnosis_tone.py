"""Diagnosis × ITA/FST figures keep clinical and estimated variables separate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_obj1_diagnosis_tone as figure  # noqa: E402


def test_estimated_ita_counts_are_derived_by_diagnosis_and_include_abstention(tmp_path):
    source = tmp_path / "resultados" / "banco_completo" / "e5"
    source.mkdir(parents=True)
    (source / "e5_fitzpatrick_ham10000.json").write_text(json.dumps({
        "per_image": {"columns": ["label", "roi_bucket", "roi_status"], "rows": [
            ["nv", "I", "ok"], ["nv", "VI", "ok"], ["nv", None, "ita_indeterminate"],
            ["mel", None, "ring_too_small"], ["mel", "II", "ok"],
        ]},
    }))
    counts = figure.load_e5_counts(tmp_path, "ham10000")
    assert counts["NV"] == {"I": 1, "II": 0, "III": 0, "IV": 0, "V": 0, "VI": 1, "missing": 1}
    assert counts["MEL"]["II"] == 1 and counts["MEL"]["missing"] == 1
    assert sum(sum(row.values()) for row in counts.values()) == 5


def test_committed_e5_panels_reconcile_to_dataset_totals():
    expected = {"ham10000": 10015, "isic2019": 25331, "isic2020": 33126}
    for dataset, total in expected.items():
        counts = figure.load_e5_counts(REPO, dataset)
        assert sum(sum(row.values()) for row in counts.values()) == total
        assert set(counts) <= set(figure.DIAGNOSIS_ORDER)


def test_approved_skin_scale_is_independent_of_uncommitted_figure_one():
    assert figure.SKIN_FILLS == (
        "#FEF3E0", "#F2C598", "#CE8B51", "#B6723F", "#91552D", "#53311B",
    )


def test_pad_counts_use_recorded_fst_not_the_estimated_bucket():
    records = [SimpleNamespace(diagnostic="BCC", fst=1), SimpleNamespace(diagnostic="BCC", fst=None),
               SimpleNamespace(diagnostic="NEV", fst=6)]
    counts = figure.count_pad(records)
    assert counts["BCC"]["I"] == 1 and counts["BCC"]["missing"] == 1
    assert counts["NV"]["VI"] == 1


def test_diagnosis_styles_respect_provided_aliases():
    assert figure.diagnosis_style("nv") == figure.diagnosis_style("NEV")
    assert figure.diagnosis_style("akiec") != figure.diagnosis_style("ACK")
    assert figure.diagnosis_style("benign")[1:] == figure.diagnosis_style("malignant")[1:]
    assert figure.diagnosis_style("benign")[0] != figure.diagnosis_style("malignant")[0]
    with pytest.raises(ValueError, match="Unknown diagnosis"):
        figure.diagnosis_style("invented")
