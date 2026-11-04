"""Tests for the mask-anchored, eq.-7 CIELAB tone augmentation (v2)."""

from __future__ import annotations

import numpy as np
import pytest

from dermssl.fitzpatrick import FST_THRESHOLDS
from dermssl.tone_augment_lab import MAX_ABS_ITA_DELTA, summarize_augmentation_records
from dermssl.tone_augment_lab_v2 import (
    ITA_OBJETIVO_HIGH,
    ITA_OBJETIVO_LOW,
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


def _find_selected_step(img, mask, seed=42, max_steps=200):
    """First step whose draw clears the mixing gate, for a determinate image."""
    for step in range(max_steps):
        if sample_apply_draw(seed, step, DIGEST) > 0.5:
            return step
    pytest.fail("no step selected by the mixing gate in range")


def _find_unselected_step(img, mask, seed=42, max_steps=200):
    for step in range(max_steps):
        if sample_apply_draw(seed, step, DIGEST) <= 0.5:
            return step
    pytest.fail("no step rejected by the mixing gate in range")


def test_objetivo_span_matches_groh_calibration_points():
    """La cota no se inventa: son los dos extremos finitos ya usados en el repo."""
    assert ITA_OBJETIVO_HIGH == FST_THRESHOLDS[0]
    assert ITA_OBJETIVO_LOW == FST_THRESHOLDS[-1]
    assert ITA_OBJETIVO_HIGH > ITA_OBJETIVO_LOW


def test_objetivo_sampling_is_deterministic_and_bounded():
    first = sample_ita_objetivo(42, 7, DIGEST)
    assert first == sample_ita_objetivo(42, 7, DIGEST)
    assert first != sample_ita_objetivo(42, 8, DIGEST)
    assert first != sample_ita_objetivo(43, 7, DIGEST)

    values = [sample_ita_objetivo(42, step, DIGEST) for step in range(300)]
    assert all(ITA_OBJETIVO_LOW <= v <= ITA_OBJETIVO_HIGH for v in values)
    assert min(values) < max(values)


def test_objetivo_sampling_does_not_consume_global_rng():
    np.random.seed(0)
    before = np.random.random()
    np.random.seed(0)
    sample_ita_objetivo(42, 1, DIGEST)
    assert np.random.random() == before


def test_apply_draw_is_independent_of_the_objetivo_draw():
    """Las dos decisiones (que delta, si se aplica) no pueden ser el mismo numero."""
    objetivo_draws = [sample_ita_objetivo(42, step, DIGEST) for step in range(50)]
    apply_draws = [sample_apply_draw(42, step, DIGEST) for step in range(50)]
    assert objetivo_draws != apply_draws


def test_compute_ita_delta_is_anchored_to_the_measured_value():
    """Eq. 7: el delta depende de donde esta la imagen, no solo del paso."""
    low = compute_ita_delta(-30.0, 42, 3, DIGEST)
    high = compute_ita_delta(50.0, 42, 3, DIGEST)
    assert low["ita_objetivo"] == high["ita_objetivo"]
    assert low["ita_delta_raw"] != high["ita_delta_raw"]
    assert low["ita_delta_raw"] == pytest.approx(-30.0 - low["ita_objetivo"])


def test_compute_ita_delta_caps_and_records_it():
    record = compute_ita_delta(1000.0, 42, 3, DIGEST, max_abs=MAX_ABS_ITA_DELTA)
    assert record["ita_delta"] == MAX_ABS_ITA_DELTA
    assert record["ita_delta_capped_by_objetivo_draw"] is True


def test_abstains_when_ita_is_indeterminate():
    """Criterio 6: piel con b* bajo MIN_B_FOR_ITA no transforma, nunca inventa delta."""
    # Neutral grey skin: b* near zero under MIN_B_FOR_ITA = 5.0.
    img, mask = _frame(value=(0.5, 0.5, 0.5))
    step = _find_selected_step(img, mask)
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False
    assert record["fallback_reason"].startswith("ita_unmeasured_")
    assert record["ita_objetivo"] is None
    assert record["selected_for_transform"] is False


def test_not_selected_by_mix_is_identity_and_distinct_from_abstention():
    img, mask = _frame()
    step = _find_unselected_step(img, mask)
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False
    assert record["fallback_reason"] == "not_selected_by_mix"
    assert record["ita_measurement_status"] == "ok"
    assert record["selected_for_transform"] is False


def test_selected_image_is_transformed_with_an_anchored_delta():
    img, mask = _frame()
    step = _find_selected_step(img, mask)
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    assert record["applied"] is True
    assert record["selected_for_transform"] is True
    assert record["ita_measurement_status"] == "ok"
    assert record["ita_objetivo"] is not None
    assert record["ita_delta_raw"] == pytest.approx(
        record["ita_measured"] - record["ita_objetivo"]
    )
    assert not np.array_equal(out, img)


def test_lesion_preserved_pixel_for_pixel_when_transformed():
    """Criterio 4: la lesion no cambia, heredado sin tocar el modulo congelado."""
    img, mask = _frame()
    step = _find_selected_step(img, mask)
    out, record = augment_skin_tone_v2(img, mask, 42, step, DIGEST)
    if record["applied"]:
        assert np.array_equal(out[mask], img[mask])


def test_degenerate_lesion_mask_falls_back_through_the_frozen_guard():
    img, _ = _frame()
    full = np.ones(img.shape[:2], dtype=bool)
    out, record = augment_skin_tone_v2(img, full, 42, 0, DIGEST)
    assert np.array_equal(out, img)
    assert record["applied"] is False


def test_sampling_rejects_bad_key():
    with pytest.raises(ValueError):
        sample_ita_objetivo(42, -1, DIGEST)
    with pytest.raises(ValueError):
        sample_apply_draw(42.0, 1, DIGEST)


def test_record_shape_is_compatible_with_the_frozen_summarizer():
    """Los records de v2, selected o no, deben poder resumirse con la funcion existente."""
    img, mask = _frame()
    selected_step = _find_selected_step(img, mask)
    unselected_step = _find_unselected_step(img, mask)
    grey, grey_mask = _frame(value=(0.5, 0.5, 0.5))
    abstain_step = _find_selected_step(grey, grey_mask)

    records = [
        augment_skin_tone_v2(img, mask, 42, selected_step, DIGEST)[1],
        augment_skin_tone_v2(img, mask, 42, unselected_step, DIGEST)[1],
        augment_skin_tone_v2(grey, grey_mask, 42, abstain_step, DIGEST)[1],
    ]
    summary = summarize_augmentation_records(records)
    assert summary["images"] == 3
    assert summary["transformed"] == 1
    assert set(summary["fallbacks_by_reason"]) == {"not_selected_by_mix", "ita_unmeasured_ita_indeterminate"}
