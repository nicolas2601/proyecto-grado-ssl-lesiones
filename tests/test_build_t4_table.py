"""The T4 table is generated from ``t4_final_eval.json`` and checked for freshness."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_t4_table as bt  # noqa: E402

T4_DIR = REPO / "resultados" / "clasificacion" / "t4_sog"
DOC = T4_DIR / "t4_final_eval.json"

pytestmark = pytest.mark.skipif(not DOC.exists(), reason="T4 not read on val yet")


def _doc() -> dict:
    return json.loads(DOC.read_text(encoding="utf-8"))


def test_committed_t4_readme_matches_its_json():
    assert bt.main([str(DOC), "--out", str(T4_DIR / "README.md"), "--check"]) == 0


def test_verdicts_come_from_the_document():
    doc = _doc()
    text = bt.render(doc)
    for name, verdict in doc["verdicts"].items():
        expected = "se cumple" if verdict["holds"] else "no se cumple"
        assert f"**{name}: {expected}**" in text


def test_a_changed_primary_contrast_changes_the_table():
    doc = copy.deepcopy(_doc())
    row = next(r for r in doc["paired"] if r["arm"] == "B_sog" and r["metric"] == "macro_f1")
    before = bt.render(doc)
    row["point_delta"] = 0.123456
    assert bt.render(doc) != before
    assert "+0,1235" in bt.render(doc)


def test_check_detects_a_stale_table(tmp_path):
    out = tmp_path / "README.md"
    assert bt.main([str(DOC), "--out", str(out), "--check"]) == 1
    assert bt.main([str(DOC), "--out", str(out)]) == 0
    assert bt.main([str(DOC), "--out", str(out), "--check"]) == 0
