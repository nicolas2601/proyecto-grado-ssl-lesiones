"""T3 (C1-v7 by luminance tercile) is registered in docs/28 before any computation.

The block fixes the inputs, the bins and their source, the key cells, the
operational prediction, the power rule and the multiplicity. The script that
reads the data checks later that this block did not change after its
registration commit.
"""

from __future__ import annotations

from pathlib import Path

from dermssl.pad_ufes_guard import registered_block

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/28-rediseno-brazo-propuesto-v7.md"
START = "<!-- registro-t3-luminancia:inicio -->"
END = "<!-- registro-t3-luminancia:fin -->"
MANIFEST_SHA = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"


def _block() -> str:
    return registered_block(DOC.read_text(encoding="utf-8"), START, END)


def test_the_block_names_the_bins_source_and_its_sha256():
    block = _block()
    assert "fst_pseudo_manifest.csv" in block and MANIFEST_SHA in block
    assert "tone_tercile" in block and "no es FST" in block


def test_the_prediction_and_power_rule_are_fixed_before_looking():
    block = _block()
    for fixed in ("20260923", "1.000 remuestreos", "n < 100", "sin potencia", "12 contrastes",
                  "bbox_crop", "mask_pool", "EXPLORATORIO"):
        assert fixed in block, f"the registration does not fix {fixed!r}"


def test_the_block_declares_multiplicity_confounding_and_what_is_not_blind():
    block = _block()
    assert "270 contrastes" in block and "Multiplicidad" in block
    assert "diagnóstico" in block and "dermatoscopio" in block
    assert "no es ciego" in block.lower()


def test_the_old_fst_registration_is_marked_as_replaced_outside_the_block():
    text = DOC.read_text(encoding="utf-8")
    note = "§5.2.1 queda sustituido por T3-luminancia"
    assert note in text and note not in _block()
    assert "#### 5.2.1 EXPLORATORIO: C1-v7 estratificado por tono" in text
