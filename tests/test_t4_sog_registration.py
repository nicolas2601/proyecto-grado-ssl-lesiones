"""T4 (Shades of Gray before DINOv2) is registered in docs/31 before any code reads data.

The block fixes the transform and its p, the weights and the protocol of each
arm, the metrics, the numeric prediction, the selection rule and what counts
as an improvement. ``scripts/t4_sog_run.py`` later checks against git that the
block did not change after its registration commit.
"""

from __future__ import annotations

from pathlib import Path

from dermssl.pad_ufes_guard import registered_block

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/31-t4-shades-of-gray.md"
START = "<!-- registro-t4-sog:inicio -->"
END = "<!-- registro-t4-sog:fin -->"
MANIFEST_SHA = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"
FIXED_PROTOCOLS = (
    "A_soup_a1.0@224|cls_attn|la0.5|tta",
    "B_s42_a1.0@224|cls4|balanced|tta",
    "public@224|cls4|la0.5|tta",
)


def _block() -> str:
    return registered_block(DOC.read_text(encoding="utf-8"), START, END)


def test_the_transform_and_its_p_are_fixed_with_verified_citations():
    block = _block()
    for fixed in ("p = 6", "10.2352/CIC.2004.12.1.art00008", "10.1109/JBHI.2014.2336473", "MethodsX"):
        assert fixed in block, f"the registration does not fix {fixed!r}"


def test_each_arm_has_its_protocol_fixed_before_looking():
    block = _block()
    for cid in FIXED_PROTOCOLS:
        assert f"`{cid}`" in block


def test_the_prediction_is_numeric_and_falsifiable():
    block = _block()
    for fixed in ("0,01", "0,03", "20260923", "1.000 remuestras", "excluye el 0"):
        assert fixed in block, f"the registration does not fix {fixed!r}"


def test_the_strata_are_luminance_terciles_from_a_named_manifest():
    block = _block()
    assert MANIFEST_SHA in block and "tone_tercile" in block
    assert "no son tono de piel" in block and "no es FST" in block


def test_selection_is_on_the_holdout_and_val_is_read_once():
    block = _block()
    assert "--single-evaluation" in block and "1e-6" in block
    assert "docs/28 §4.1" in block
