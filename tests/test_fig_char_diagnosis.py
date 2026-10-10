"""Characterisation: diagnostic counts must reconcile with the documented dataset totals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))
sys.path.insert(0, str(REPO / "src"))

import fig_char_diagnosis as figure  # noqa: E402


@pytest.mark.parametrize("dataset,_label", figure.PANELS)
def test_counts_sum_to_documented_total(dataset, _label):
    assert sum(figure.counts(REPO, dataset).values()) == figure.source.TOTALS[dataset]


def test_wrong_total_aborts(monkeypatch):
    monkeypatch.setitem(figure.source.TOTALS, "ham10000", 1)
    with pytest.raises(ValueError, match="documented total"):
        figure.counts(REPO, "ham10000")


def test_build_exports_all_formats(tmp_path):
    assert {p.suffix for p in figure.build(REPO, tmp_path)} == {".pdf", ".svg", ".png"}
    trace = json.loads((tmp_path / "fig_char_diagnosis_distribution.traceability.json").read_text(encoding="utf-8"))
    assert trace["totals"]["pad_ufes_20"] == 2298
