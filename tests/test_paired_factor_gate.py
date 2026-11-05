"""The gate must refuse the contrasts that would produce an unattributable delta."""

from __future__ import annotations

import pytest

from dermssl.pretrain.paired_factor_gate import (
    CONTROL_MODE,
    TREATED_MODE,
    PairedFactorError,
    validate_paired_factor,
)


def _arms(**treated_overrides):
    shared = {
        "method": "simsiam_dinov2",
        "backbone": "dinov2_vitb14",
        "seed": 42,
        "max_steps": 6000,
        "batch_size_per_gpu": 16,
        "freeze_blocks": 8,
        "resolution": 224,
        "precision": "bf16",
        "data_revision": "pool-v3",
    }
    control = {**shared, "adaptation_mode": CONTROL_MODE, "transformed_image_root": "",
               "tone_consistency_weight": 0.0, "output_dir": "/runs/control"}
    treated = {
        **shared,
        "adaptation_mode": TREATED_MODE,
        "transformed_image_root": "/data/tone",
        "transformed_manifest_path": "/data/tone/tone_transform_manifest.json",
        "transformed_data_revision": "rev-abc",
        "tone_consistency_weight": 0.194,
        "output_dir": "/runs/tone",
        **treated_overrides,
    }
    return control, treated


def test_accepts_a_clean_single_factor_contrast():
    control, treated = _arms()
    changed = validate_paired_factor(control, treated)
    assert changed == {
        "adaptation_mode",
        "transformed_image_root",
        "transformed_manifest_path",
        "transformed_data_revision",
        "tone_consistency_weight",
    }


def test_refuses_a_treated_arm_whose_tone_term_is_inert():
    """Con peso cero el brazo tratado paga las vistas transformadas y no las usa.

    Este es el agujero que abre admitir el peso en TREATMENT_FIELDS: sin esta
    comprobacion el contraste pasa todas las demas y no mide el tratamiento.
    """
    control, treated = _arms(tone_consistency_weight=0.0)
    with pytest.raises(PairedFactorError, match="inerte"):
        validate_paired_factor(control, treated)


def test_refuses_a_control_carrying_a_tone_weight():
    """En two_view_ssl no hay par, asi que un peso distinto de cero no describe lo que corre."""
    control, treated = _arms()
    control["tone_consistency_weight"] = 0.194
    with pytest.raises(PairedFactorError, match="debe ser cero"):
        validate_paired_factor(control, treated)


def test_a_missing_tone_weight_in_the_treated_arm_is_not_silently_accepted():
    """La ausencia del campo no puede leerse como tratamiento aplicado."""
    control, treated = _arms()
    del treated["tone_consistency_weight"]
    with pytest.raises(PairedFactorError, match="inerte"):
        validate_paired_factor(control, treated)


def test_output_dir_may_differ():
    """Donde escribe un brazo no cambia lo que computa."""
    control, treated = _arms()
    control["output_dir"] = "/runs/a"
    treated["output_dir"] = "/runs/b"
    validate_paired_factor(control, treated)


@pytest.mark.parametrize(
    "field,value",
    [
        ("freeze_blocks", 4),
        ("resolution", 336),
        ("batch_size_per_gpu", 32),
        ("max_steps", 3000),
        ("precision", "fp32"),
        ("backbone", "resnet50"),
    ],
)
def test_refuses_a_second_factor(field, value):
    """Este es el modo de falla que el contraste entero teme."""
    control, treated = _arms(**{field: value})
    with pytest.raises(PairedFactorError) as excinfo:
        validate_paired_factor(control, treated)
    assert field in str(excinfo.value)


def test_refuses_mismatched_seeds():
    """Una semilla distinta cambia orden de filas y fronteras de batch."""
    control, treated = _arms(seed=43)
    with pytest.raises(PairedFactorError):
        validate_paired_factor(control, treated)


def test_refuses_a_missing_seed():
    control, treated = _arms()
    del control["seed"]
    del treated["seed"]
    with pytest.raises(PairedFactorError):
        validate_paired_factor(control, treated)


def test_refuses_a_control_that_is_not_a_control():
    control, treated = _arms()
    control["adaptation_mode"] = TREATED_MODE
    with pytest.raises(PairedFactorError):
        validate_paired_factor(control, treated)


def test_refuses_a_control_carrying_transformed_images():
    control, treated = _arms()
    control["transformed_image_root"] = "/data/tone"
    with pytest.raises(PairedFactorError):
        validate_paired_factor(control, treated)


def test_refuses_a_treated_arm_without_transformed_root():
    control, treated = _arms(transformed_image_root="")
    with pytest.raises(PairedFactorError):
        validate_paired_factor(control, treated)


def test_refuses_two_controls():
    control, treated = _arms()
    treated["adaptation_mode"] = CONTROL_MODE
    with pytest.raises(PairedFactorError):
        validate_paired_factor(control, treated)
