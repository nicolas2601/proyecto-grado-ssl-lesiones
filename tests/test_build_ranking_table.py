"""The macro-F1 ranking is generated from committed JSON and checked for freshness."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_ranking_table as br  # noqa: E402


def test_committed_ranking_is_fresh():
    assert br.TARGET.read_text(encoding="utf-8") == br.render()


def test_ranking_is_sorted_best_to_worst():
    scores = [e.macro_f1 for e in br.ranked()]
    assert scores == sorted(scores, reverse=True)


def test_every_source_family_is_ranked():
    families = {e.family for e in br.ranked()}
    assert families == {br.OURS, "congelado (pesos públicos)", "fine-tuning supervisado"}


def test_lr_probe_runs_stay_out_of_the_main_ranking():
    main = {e.source for e in br.ranked()}
    assert all(e.source not in main for e in br.lrprobe_entries())


def test_overlap_is_symmetric_and_needs_both_intervals():
    a = br.Entry("a", br.OURS, 0.7, (0.65, 0.75), "x", Path("a"))
    b = br.Entry("b", br.OURS, 0.6, (0.55, 0.66), "x", Path("b"))
    c = br.Entry("c", br.OURS, 0.5, None, "x", Path("c"))
    assert br._overlaps(a, b) and br._overlaps(b, a)
    assert not br._overlaps(a, c)
    assert br._versus(c, a) == "sin IC"
