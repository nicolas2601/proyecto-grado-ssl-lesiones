"""T3: ITA-vs-Fitzpatrick figure must follow the committed PAD validation and never beat-flag wrongly."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_t3_ita_vs_fitzpatrick as figure  # noqa: E402


def test_committed_numbers_reconcile():
    doc = figure.load(REPO)["doc"]
    for key, _ in figure.CONDITIONS:
        m = doc["metrics"][key]
        assert m["accuracy"] < m["majority_baseline"]["accuracy"] and not m["beats_majority"]


def test_inconsistent_beats_flag_aborts(tmp_path, monkeypatch):
    doc = copy.deepcopy(figure.load(REPO)["doc"])
    doc["metrics"]["masked"]["beats_majority"] = True
    path = tmp_path / figure.SOURCE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="beats_majority"):
        figure.load(tmp_path)


def test_build_exports_all_formats(tmp_path):
    assert {p.suffix for p in figure.build(REPO, tmp_path)} == {".pdf", ".svg", ".png"}
    trace = json.loads((tmp_path / "fig_t3_ita_vs_fitzpatrick.traceability.json").read_text(encoding="utf-8"))
    assert trace["exploratory"] is True and trace["discrepancy"]
