"""The T1 table (docs/30 5) is generated from ``pad_ufes_fst_validation.json`` and checked for freshness.

Every number, the verdicts and the phototype note come from the JSON; the
renderer is pinned on a small synthetic document, and the committed README
is compared against its committed JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_pad_ufes_fst_table as bt  # noqa: E402
import closing_table_rules as rules  # noqa: E402

T1_DIR = REPO / "resultados" / "fst" / "pad_ufes_t1"
NOTE = "El grupo III-VI es 84 % tipo III (392/465), así que NO representa piel oscura."


def _condition(acc: float, kappa: float | None, beats: bool | None) -> dict:
    return {"n_evaluable": 1265, "n_unavailable": 229, "accuracy": acc, "off_by_one_accuracy": 0.62,
            "confusion": {"1": {"1": 53, "6": 9}, "2": {"2": 174}, "6": {"5": 1}},
            "majority_baseline": {"label": 2, "accuracy": 0.59, "off_by_one_accuracy": 0.95},
            "linear_kappa": kappa, "ci95": {"accuracy": [0.19, 0.24], "linear_kappa": [-0.02, 0.04]},
            "beats_majority": beats}


def _doc() -> dict:
    return {"code_revision": "abc123", "registration_commit": "def456", "exploratory": True,
            "n_images": 2298, "n_without_fst": 804, "fst_note": NOTE,
            "metrics": {"whole_image": _condition(0.2174, 0.0105, False),
                        "masked": _condition(0.2288, 0.0256, False)},
            "verdicts": {"FST-P1": "cumplida", "FST-P2": "cumplida"}, "records": []}


def test_each_condition_row_reads_its_numbers_from_the_json():
    text = bt.render(_doc())
    assert ("| Imagen completa | 1.265 | 229 | 0,2174 [0,1900; 0,2400] | 0,6200 | 0,0105 [-0,0200; 0,0400] "
            "| II · 0,5900 · 0,9500 | no |") in text
    assert "| Fondo por Otsu (máscara) | 1.265 | 229 | 0,2288 [0,1900; 0,2400] |" in text


def test_the_confusion_matrix_fills_absent_cells_with_zero_and_totals_rows():
    text = bt.render(_doc())
    assert "| I | 53 | 0 | 0 | 0 | 0 | 9 | 62 |" in text
    assert "| II | 0 | 174 | 0 | 0 | 0 | 0 | 174 |" in text
    assert "| III | 0 | 0 | 0 | 0 | 0 | 0 | 0 |" in text
    assert "| VI | 0 | 0 | 0 | 0 | 1 | 0 | 1 |" in text
    assert text.count("| Real \\ Predicho | I | II | III | IV | V | VI | Total |") == 2


def test_verdicts_note_and_citations_come_through():
    text = bt.render(_doc())
    assert "FST-P1" in text and "FST-P2" in text and text.count("**cumplida**") == 2
    assert "0,20" in text  # the kappa threshold of FST-P2, read from the validation script
    assert NOTE in text
    assert rules.ITA_LABEL in text and rules.ITA_NOTE in text
    for fragment in ("Pacheco et al.", "Data in Brief", "10.17632/zr7vgbcyr2.1", "CC BY 4.0",
                     "Alipour, Burke y Courtney", "Skin Research and Technology", "PMC13459198"):
        assert fragment in text


def test_missing_values_render_as_not_available():
    doc = _doc()
    empty = {"n_evaluable": 0, "n_unavailable": 3, "accuracy": None, "off_by_one_accuracy": None,
             "confusion": {}, "majority_baseline": {"label": None, "accuracy": None, "off_by_one_accuracy": None},
             "linear_kappa": None, "ci95": {"accuracy": None, "linear_kappa": None}, "beats_majority": None}
    doc["metrics"]["masked"] = empty
    assert "| Fondo por Otsu (máscara) | 0 | 3 | n/d | n/d | n/d | n/d | n/d |" in bt.render(doc)


def test_check_detects_a_stale_table(tmp_path):
    doc_path, out = tmp_path / "pad_ufes_fst_validation.json", tmp_path / "README.md"
    doc_path.write_text(json.dumps(_doc()), encoding="utf-8")
    assert bt.main([str(doc_path), "--out", str(out), "--check"]) == 1
    assert bt.main([str(doc_path), "--out", str(out)]) == 0
    assert bt.main([str(doc_path), "--out", str(out), "--check"]) == 0


def test_committed_t1_readme_matches_its_json():
    assert bt.main([str(T1_DIR / "pad_ufes_fst_validation.json"), "--out", str(T1_DIR / "README.md"),
                    "--check"]) == 0
