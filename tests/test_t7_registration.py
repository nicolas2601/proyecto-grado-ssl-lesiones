"""T7 (retrain A2's decoder with darkening augmentation and dark oversampling) is registered in docs/27.

The block fixes the recipe kept from A2, the two training changes with their
values, the arms (a) and (b), the evaluation shared with T2, the operational
prediction with paired intervals, the seed-noise comparison, the r0 teacher
check with its conditional rule, the design limitations and the pre-mortem.
The runners check later that this block did not change after its registration.
"""

from __future__ import annotations

from pathlib import Path

from dermssl.pad_ufes_guard import registered_block

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs/27-plan-v7-mejoras-ssl.md"
START = "<!-- registro-t7-reentreno-luminancia:inicio -->"
END = "<!-- registro-t7-reentreno-luminancia:fin -->"
MANIFEST_SHA = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"
PSEUDO_R0_SHA = "8c9e47978070baa7bfa9f0174a80eb4e4236d45737801abf690804ae1eb0fc87"
CACHE_MANIFEST_SHA = "a83f927ab798b6f7a0614c3923fda77d082f887d10fdbe73416e34e9574ba71a"


def _block() -> str:
    return registered_block(DOC.read_text(encoding="utf-8"), START, END)


def _require(block: str, *fixed: str) -> None:
    for item in fixed:
        assert item in block, f"the registration does not fix {item!r}"


def test_the_bins_are_luminance_from_the_published_manifest():
    block = _block()
    _require(block, "tone_tercile", MANIFEST_SHA, "No son tono de piel", "no es FST")


def test_the_a2_recipe_is_kept_and_pinned():
    _require(_block(), PSEUDO_R0_SHA, CACHE_MANIFEST_SHA, "20 épocas", "Adam", "1e-4",
             "semilla 42", "best-val", "52.416")


def test_the_two_training_changes_have_fixed_values():
    _require(_block(), "k = 2", "γ ~ U[1,0; 2,2]", "U[0,85; 1,0]", "p = 0,5",
             "medio y claro", "brillo ×[0,6; 1,4]")


def test_the_arms_and_their_purpose_are_registered():
    _require(_block(), "Brazo (a)", "semilla 43", "Brazo (b)", "solo con el aumento", "ruido de semilla")


def test_the_prediction_is_operational_with_paired_intervals():
    _require(_block(), "20260923", "1.000 remuestreos", "+0,02", "−0,005", "pareado", "n < 100",
             "R(T7) − R(A2-s43)")


def test_the_r0_check_and_its_conditional_rule_precede_the_gpu():
    _require(_block(), "Chequeo del maestro", "techo del maestro", "adenda fechada", "antes de lanzar")


def test_limitations_replica_probability_and_pre_mortem_are_explicit():
    block = _block()
    _require(block, "Limitación del diseño: transductividad", "Réplica descriptiva en ISIC 2018",
             "722", "158", "20-30 %", "Pre-mortem", "no es ciego", "diagnóstico", "dermatoscopio")
