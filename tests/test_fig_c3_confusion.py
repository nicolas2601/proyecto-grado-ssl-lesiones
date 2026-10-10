"""C3 confusion matrices: committed JSON must reconcile and the export must produce all formats."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_c3_confusion as figure  # noqa: E402


@pytest.mark.parametrize("key", sorted(figure.CELLS))
def test_committed_matrix_reconciles(key):
    cell = figure.load_cell(REPO, key)
    assert cell["counts"].sum() == cell["doc"]["n"]
    assert cell["support"].sum() == cell["doc"]["n"]


def test_inconsistent_counts_abort(tmp_path):
    source = tmp_path / figure.CONFUSION_DIR / "x"
    source.mkdir(parents=True)
    bad = {"clases": ["a", "b"], "n": 5, "conteos": [[1, 1], [1, 1]], "normalizada_por_fila": [[0.5, 0.5], [0.5, 0.5]]}
    (source / "m.json").write_text(json.dumps(bad), encoding="utf-8")
    figure.CELLS["bad"] = ("x/m.json", "Bad")
    try:
        with pytest.raises(ValueError, match="counts sum"):
            figure.load_cell(tmp_path, "bad")
    finally:
        del figure.CELLS["bad"]


def test_build_exports_three_formats_and_traceability(tmp_path):
    paths = figure.build(REPO, "ham10000", tmp_path)
    assert {p.suffix for p in paths} == {".pdf", ".svg", ".png"}
    trace = json.loads((tmp_path / "fig_c3_confusion_ham10000.traceability.json").read_text(encoding="utf-8"))
    assert trace["n"] == 2004 and trace["status"] == "Artefactos verificados"
    assert 'font-family: \'' in (tmp_path / "fig_c3_confusion_ham10000.svg").read_text(encoding="utf-8")
