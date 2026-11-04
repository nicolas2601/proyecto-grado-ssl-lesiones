"""Tests for the mask-anchored, eq.-7 CIELAB tone augmentation (v2).

v2 history: the first version sampled ``ita_objetivo`` from a fixed global
span, decoupled from the measured ITA. On realistic (high-median) ITA
distributions that collapsed 80%+ of deltas to the same capped constant --
see the module docstring's "REVISION". These tests cover the corrected
design: a window centered on the measured ITA, with no post-hoc clip.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl.tone_augment_lab import MAX_ABS_ITA_DELTA, summarize_augmentation_records
from dermssl.tone_augment_lab_v2 import (
    TARGET_WINDOW_DEGREES,
    augment_skin_tone_v2,
    compute_ita_delta,
    sample_apply_draw,
    sample_ita_objetivo,
)

DIGEST = "a" * 64


def _frame(value=(0.72, 0.58, 0.49), size=64):
    img = np.zeros((size, size, 3), dtype=np.float64)
    img[:] = np.asarray(value, dtype=np.float64)
    mask = np.zeros((size, size), dtype=bool)
    mask[size // 3 : 2 * size // 3, size // 3 : 2 * size // 3] = True
    img[mask] = (0.33, 0.21, 0.19)
    return img, mask


def _find_selected_step(seed=42, max_steps=200):
    """First step whose draw clears the mixing gate."""
    for step in range(max_steps):
        if sample_apply_draw(seed, step, DIGEST) > 0.5:
            return step
    pytest.fail("no step selected by the mixing gate in range")


def _find_unselected_step(seed=42, max_steps=200):
    for step in range(max_steps):
        if sample_apply_draw(seed, step, DIGEST) <= 0.5:
            return step
    pytest.fail("no step rejected by the mixing gate in range")


def test_window_reuses_the_frozen_cap_not_a_second_derivation():
    """La ventana no es un numero nuevo: es el mismo cap de tone_augment_lab."""
    assert TARGET_WINDOW_DEGREES == MAX_ABS_ITA_DELTA


def test_objetivo_is_centered_on_the_measured_value():
    """La correccion del bug: el objetivo depende de la imagen, no de un rango fijo."""
    low_anchor = sample_ita_objetivo(-30.0, 42, 7, DIGEST)
    high_anchor = sample_ita_objetivo(56.0, 42, 7, DIGEST)
    assert low_anchor != high_anchor
    assert low_anchor == pytest.approx(-30.0 + (high_anchor - 56.0))


def test_objetivo_sampling_is_deterministic_and_bounded_around_the_anchor():
    first = sample_ita_objetivo(56.0, 42, 7, DIGEST)
    assert first == sample_ita_objetivo(56.0, 42, 7, DIGEST)
    assert first != sample_ita_objetivo(56.0, 42, 8, DIGEST)
    assert first != sample_ita_objetivo(56.0, 43, 7, DIGEST)

    values = [sample_ita_objetivo(56.0, 42, step, DIGEST) for step in range(300)]
    assert all(56.0 - TARGET_WINDOW_DEGREES <= v <= 56.0 + TARGET_WINDOW_DEGREES for v in values)
    assert min(values) < max(values)


def test_objetivo_sampling_does_not_consume_global_rng():
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    sample_ita_objetivo(56.0, 42, 1, DIGEST)
    assert np.random.random() == before


def test_apply_draw_is_independent_of_the_objetivo_draw():
    objetivo_draws = [sample_ita_objetivo(56.0, 42, step, DIGEST) for step in range(50)]
    apply_draws = [sample_apply_draw(42, step, DIGEST) for step in range(50)]
    assert objetivo_draws != apply_draws


def test_delta_never_needs_truncation_regardless_of_the_measured_value():
    """Criterio 1 de maestro: la ventana hace el cap estructural, no un recorte posterior."""
    for measured in (-80.0, -30.0, 0.0, 56.0, 83.0, 500.0):
        record = compute_ita_delta(measured, 42, 3, DIGEST)
        assert record["ita_delta_raw"] == record["ita_delta"]
        assert record["ita_delta_capped"] is False
        assert abs(record["ita_delta"]) <= TARGET_WINDOW_DEGREES


def test_delta_tracks_the_measured_value_without_collapsing_to_a_constant():
    """Criterio 2 de maestro: a distribucion realista, el delta no colapsa en +25.

    Regresion directa del bug medido: con objetivo en un rango fijo global y
    mediana de ITA real ~56, el 79,9% de 5.000 draws caia a menos de 0,025
    grados del cap. Con la ventana centrada, ningun draw puede tocar el borde
    por recorte -- se verifica que la dispersion de deltas es real, no un
    monton de valores iguales.
    """
    rng = np.random.default_rng(1)
    ita_values = np.clip(rng.normal(56.2, 20, 2000), -58.3, 83.2)
    deltas = np.array(
        [
            compute_ita_delta(float(ita), 42, step, DIGEST)["ita_delta"]
            for step, ita in enumerate(ita_values)
        ]
    )
    at_boundary = np.isclose(np.abs(deltas), TARGET_WINDOW_DEGREES, atol=1e-6)
    assert at_boundary.mean() < 0.05
    assert deltas.std() > 1.0


def test_abstains_when_ita_is_indeterminate():
    """Criterio 6 original: piel con b* bajo MIN_B_FOR_ITA no transforma."""
    img, mask = _frame(value=(0.5, 0.5, 0.5))
    step = _find_selected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False
    assert record["fallback_reason"].startswith("ita_unmeasured_")
    assert record["ita_objetivo"] is None
    assert record["selected_for_transform"] is False


def test_not_selected_by_mix_is_identity_and_distinct_from_abstention():
    img, mask = _frame()
    step = _find_unselected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False
    assert record["fallback_reason"] == "not_selected_by_mix"
    assert record["ita_measurement_status"] == "ok"
    assert record["selected_for_transform"] is False


def test_selected_image_is_transformed_with_an_anchored_undapped_delta():
    img, mask = _frame()
    step = _find_selected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert record["applied"] is True
    assert record["selected_for_transform"] is True
    assert record["ita_measurement_status"] == "ok"
    assert record["ita_objetivo"] is not None
    assert record["ita_delta_raw"] == pytest.approx(
        record["ita_measured"] - record["ita_objetivo"]
    )
    assert record["ita_delta_raw"] == record["ita_delta_applied"]
    assert not np.array_equal(out, img)


def test_lesion_preserved_pixel_for_pixel_when_transformed():
    img, mask = _frame()
    step = _find_selected_step()
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    if record["applied"]:
        assert np.array_equal(out[mask], img[mask])


def test_degenerate_lesion_mask_falls_back_through_the_frozen_guard():
    img, _ = _frame()
    full = np.ones(img.shape[:2], dtype=bool)
    out, record = augment_skin_tone_v2(img, full, 42, 0, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False


def test_sampling_rejects_bad_key_or_anchor():
    with pytest.raises(ValueError):
        sample_ita_objetivo(56.0, 42, -1, DIGEST)
    with pytest.raises(ValueError):
        sample_ita_objetivo(float("nan"), 42, 1, DIGEST)
    with pytest.raises(ValueError):
        sample_apply_draw(42.0, 1, DIGEST)


def test_record_shape_is_compatible_with_the_frozen_summarizer():
    img, mask = _frame()
    selected_step = _find_selected_step()
    unselected_step = _find_unselected_step()
    grey, grey_mask = _frame(value=(0.5, 0.5, 0.5))
    abstain_step = _find_selected_step()

    records = [
        augment_skin_tone_v2(img, mask, 42, selected_step, DIGEST)[1],
        augment_skin_tone_v2(img, mask, 42, unselected_step, DIGEST)[1],
        augment_skin_tone_v2(grey, grey_mask, 42, abstain_step, DIGEST)[1],
    ]
    summary = summarize_augmentation_records(records)
    assert summary["images"] == 3
    assert summary["transformed"] == 1
    assert set(summary["fallbacks_by_reason"]) == {
        "not_selected_by_mix",
        "ita_unmeasured_ita_indeterminate",
    }
