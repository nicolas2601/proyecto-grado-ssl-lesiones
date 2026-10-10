from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))
sys.path.insert(0, str(REPO / "src"))

import fig_t2_ita_intervals as table  # noqa: E402
from dermssl.fitzpatrick import FST_THRESHOLDS, fst_bucket  # noqa: E402


def test_t2_intervals_follow_source_cuts_and_exact_boundary_rule():
    intervals = table.intervals()

    assert [row["group"] for row in intervals] == ["I", "II", "III", "IV", "V", "VI"]
    assert [row["condition"] for row in intervals] == [
        "ITA > 40°",
        "23° < ITA ≤ 40°",
        "12° < ITA ≤ 23°",
        "0° < ITA ≤ 12°",
        "−25° < ITA ≤ 0°",
        "ITA ≤ −25°",
    ]
    for index, threshold in enumerate(FST_THRESHOLDS):
        assert fst_bucket(threshold) == index + 2
        assert fst_bucket(math.nextafter(threshold, math.inf)) == index + 1


def test_t2_table_states_nonclinical_interpretation_and_abstention():
    rendered = table.render_table()

    assert r"\mathrm{ITA} > 40^\circ" in rendered
    assert r"23^\circ < \mathrm{ITA} \le 40^\circ" in rendered
    assert r"\mathrm{ITA} \le -25^\circ" in rendered
    assert "no equivalen a fototipos clínicos validados" in rendered
    assert "b* ≥ 5" in rendered
    assert r"\caption{" in rendered


def test_t2_build_writes_latex_and_traceable_sidecars(tmp_path):
    paths = table.build(REPO, tmp_path)

    assert paths.name == "tab_t2_ita_intervals.tex"
    assert paths.is_file()
    trace = json.loads((tmp_path / "tab_t2_ita_intervals.traceability.json").read_text(encoding="utf-8"))
    source = REPO / "src/dermssl/fitzpatrick.py"
    assert trace["thresholds"] == list(FST_THRESHOLDS)
    assert trace["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (tmp_path / "tab_t2_ita_intervals.caption.txt").is_file()
    assert (tmp_path / "tab_t2_ita_intervals.note.txt").is_file()
    with pytest.raises(FileExistsError):
        table.build(REPO, tmp_path)
