"""T7 statistics (docs/27): gaps of several segmenters on the same resampled images."""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import luminance_retrain as lr
from dermssl import luminance_threshold as lt


def _groups(rng, shift_dark: float, shift_rest: float = 0.0, n_dark: int = 150, n_rest: int = 220):
    return (np.clip(rng.normal(0.70 + shift_dark, 0.1, n_dark), 0, 1),
            np.clip(rng.normal(0.78 + shift_rest, 0.1, n_rest), 0, 1))


def test_the_reduction_against_a2_is_t6s_paired_statistic_bit_for_bit():
    rng = np.random.default_rng(0)
    a2, t7 = _groups(rng, 0.0), _groups(rng, 0.03)
    gaps = lr.shared_gaps({"A2": a2, "T7": t7}, n_resamples=300, seed=20260923)
    reference = lt.gap_reduction(a2[0], t7[0], a2[1], t7[1], 300, 20260923)
    assert lr.reduction(gaps, "A2", "T7") == reference["reduction"]
    assert lr.gap(gaps, "T7") == reference["gap_adapted"]


def test_the_seed_contrast_is_the_difference_of_reductions_on_the_same_resamples():
    rng = np.random.default_rng(1)
    models = {"A2": _groups(rng, 0.0), "T7": _groups(rng, 0.04), "A2-s43": _groups(rng, 0.005)}
    gaps = lr.shared_gaps(models, n_resamples=300, seed=5)
    contrast = lr.reduction_difference(gaps, "T7", "A2-s43")
    expected = lr.reduction(gaps, "A2", "T7")["delta"] - lr.reduction(gaps, "A2", "A2-s43")["delta"]
    assert contrast["delta"] == pytest.approx(expected)
    assert contrast["ci95_low"] <= contrast["delta"] <= contrast["ci95_high"]


def test_models_must_share_the_images():
    with pytest.raises(ValueError, match="mismas imágenes"):
        lr.shared_gaps({"A2": (np.ones(3), np.ones(4)), "T7": (np.ones(3), np.ones(5))}, 10, 0)


@pytest.mark.parametrize(("low", "reading"), [(0.001, lr.SEED_SEPARATED), (0.0, lr.SEED_NOT_SEPARATED),
                                              (-0.01, lr.SEED_NOT_SEPARATED)])
def test_the_seed_reading(low, reading):
    assert lr.seed_reading(low) == reading


@pytest.mark.parametrize(("prediction", "seed", "conclusion"), [
    (lt.FULFILLED, lr.SEED_SEPARATED, lr.SHRINKS),
    (lt.FULFILLED, lr.SEED_NOT_SEPARATED, lr.FULFILLED_WITHIN_NOISE),
    (lt.NOT_FULFILLED, lr.SEED_SEPARATED, lt.NOT_FULFILLED),
    (lt.NO_POWER, lr.SEED_SEPARATED, lt.NO_POWER),
])
def test_the_registered_conclusion_needs_both(prediction, seed, conclusion):
    assert lr.conclusion(prediction, seed) == conclusion
