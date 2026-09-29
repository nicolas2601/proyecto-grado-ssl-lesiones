"""T6 (luminance-adaptive mask threshold on A2) is registered in docs/27 before any computation.

The block fixes the source paper, the bins, the candidate grid, the fit split,
the operational prediction with its intervals and the follow-up rule. The
runner checks later that this block did not change after its registration.
"""

from __future__ import annotations

from pathlib import Path

from dermssl.pad_ufes_guard import registered_block

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/27-plan-v7-mejoras-ssl.md"
START = "<!-- registro-t6-umbral-luminancia:inicio -->"
END = "<!-- registro-t6-umbral-luminancia:fin -->"
MANIFEST_SHA = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"
VAL_IDS_SHA = "6ff4c17f04a9eae2f0d058aaa41084ea65b9aff91e4d5e5b05c924b4465a2fc6"


def _block() -> str:
    return registered_block(DOC.read_text(encoding="utf-8"), START, END)


def test_the_block_cites_the_verified_paper_and_says_this_is_not_skin_tone():
    block = _block()
    assert "2510.06655" in block and "Stothers" in block
    assert "tone_tercile" in block and MANIFEST_SHA in block
    assert "no son tono de piel" in block


def test_the_grid_and_the_fit_split_are_fixed_before_looking():
    block = _block()
    for fixed in ("0,30", "0,70", "0,05", "StratifiedGroupKFold", VAL_IDS_SHA, "pseudo-máscaras r0"):
        assert fixed in block, f"the registration does not fix {fixed!r}"


def test_the_prediction_is_operational_with_paired_intervals():
    block = _block()
    for fixed in ("20260923", "1.000 remuestreos", "+0,02", "−0,005", "pareado", "n < 100"):
        assert fixed in block, f"the registration does not fix {fixed!r}"


def test_the_block_declares_the_oracle_the_follow_up_and_what_is_not_blind():
    block = _block()
    assert "oráculo" in block and "C1-v7" in block
    assert "no es ciego" in block.lower()
    assert "diagnóstico" in block and "dermatoscopio" in block
