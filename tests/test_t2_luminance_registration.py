"""T2 (Dice by luminance tercile) is registered in docs/27 before any computation.

The block fixes what is computed, the bins and their source, the operational
prediction and the power rule. The script that reads the data checks later
that this block did not change after its registration commit.
"""

from __future__ import annotations

from pathlib import Path

from dermssl.pad_ufes_guard import registered_block

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/27-plan-v7-mejoras-ssl.md"
START = "<!-- registro-t2-luminancia:inicio -->"
END = "<!-- registro-t2-luminancia:fin -->"
MANIFEST_SHA = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"


def _block() -> str:
    return registered_block(DOC.read_text(encoding="utf-8"), START, END)


def test_the_block_names_the_bins_source_and_its_sha256():
    block = _block()
    assert "fst_pseudo_manifest.csv" in block and MANIFEST_SHA in block
    assert "tone_tercile" in block


def test_the_prediction_is_operational_before_looking():
    block = _block()
    for fixed in ("20260923", "1.000 remuestreos", "−0,02", "n < 100", "sin potencia"):
        assert fixed in block, f"the registration does not fix {fixed!r}"


def test_the_block_declares_the_confounding_and_what_is_not_blind():
    block = _block()
    assert "diagnóstico" in block and "dermatoscopio" in block
    assert "no es ciego" in block.lower() or "no es ciega" in block.lower()
