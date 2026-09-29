"""T6 (docs/27): one mask threshold per luminance tercile, chosen on the fit ids only.

Synthetic probability maps stand in for A2: the module never sees a model, a
file or a real mask, so every rule of the registration can be pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

import seg_paths  # noqa: F401
from dermssl import luminance_threshold as lt
from evaluate_segmentation import binarize, compute_binary_metrics

SIZE = 32


def _lesion(prob_inside: float, prob_outside: float = 0.1, size: int = SIZE):
    truth = np.zeros((size, size), bool)
    truth[8:24, 8:24] = True
    probability = np.where(truth, prob_inside, prob_outside).astype(np.float32)
    return probability, truth


def test_the_grid_is_the_registered_one_and_holds_the_fixed_threshold():
    assert lt.GRID == (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
    assert lt.FIXED == 0.5 and lt.FIXED in lt.GRID


def test_dice_is_the_evaluators_dice_bit_for_bit():
    rng = np.random.RandomState(0)
    for threshold in lt.GRID:
        probability = rng.rand(SIZE, SIZE).astype(np.float32)
        truth = rng.rand(SIZE, SIZE) > 0.6
        expected = compute_binary_metrics(binarize(probability, threshold), truth)["dice"]
        assert lt.dice_at(probability, truth, threshold) == expected


def test_a_pixel_exactly_at_the_threshold_is_background_and_two_empty_masks_agree():
    probability = np.full((4, 4), 0.5, np.float32)
    assert lt.dice_at(probability, np.zeros((4, 4), bool), 0.5) == 1.0
    assert lt.dice_at(probability, np.ones((4, 4), bool), 0.5) == 0.0


def test_the_curve_has_one_dice_per_grid_threshold():
    probability, truth = _lesion(0.42)
    curve = lt.dice_curve(probability, truth)
    assert curve.shape == (len(lt.GRID),)
    assert curve[lt.GRID.index(0.4)] == 1.0 and curve[lt.GRID.index(0.5)] == 0.0


def _curves_and_terciles(n_per_tercile: int = 3):
    probs = {"dark": 0.42, "medium": 0.62, "light": 0.9}
    curves, terciles = {}, {}
    for tercile, inside in probs.items():
        for k in range(n_per_tercile):
            image_id = f"{tercile}_{k}"
            curves[image_id] = lt.dice_curve(*_lesion(inside))
            terciles[image_id] = tercile
    return curves, terciles


def test_each_tercile_gets_the_threshold_that_maximises_its_fit_dice():
    curves, terciles = _curves_and_terciles()
    chosen = lt.select_thresholds(curves, terciles, list(curves), min_n=1)
    assert chosen["dark"]["threshold"] == 0.4  # 0.30-0.40 tie at Dice 1; 0.40 is closest to 0.5
    assert chosen["medium"]["threshold"] == 0.5  # 0.30-0.60 tie; 0.5 itself wins
    assert chosen["light"]["threshold"] == 0.5
    assert chosen["dark"]["fit_n"] == 3 and chosen["dark"]["fit_dice"] == 1.0
    assert chosen["dark"]["fit_dice_at_fixed"] == 0.0


def test_a_tie_at_equal_distance_from_the_fixed_threshold_goes_to_the_lower_one():
    curve = np.array([0.1, 0.1, 0.1, 0.9, 0.5, 0.9, 0.1, 0.1, 0.1])
    chosen = lt.select_thresholds({"a": curve}, {"a": "dark"}, ["a"], min_n=1, terciles_used=("dark",))
    assert chosen["dark"]["threshold"] == 0.45


def test_only_the_fit_ids_choose_the_threshold():
    curves, terciles = _curves_and_terciles()
    curves["dark_eval"] = lt.dice_curve(*_lesion(0.68))
    terciles["dark_eval"] = "dark"
    fit = [i for i in curves if i != "dark_eval"]
    assert lt.select_thresholds(curves, terciles, fit, min_n=1)["dark"]["threshold"] == 0.4


def test_a_tercile_with_too_few_fit_images_is_refused():
    curves, terciles = _curves_and_terciles(n_per_tercile=2)
    with pytest.raises(ValueError, match="dark"):
        lt.select_thresholds(curves, terciles, list(curves), min_n=3)


def test_the_adapted_rule_reads_each_images_own_tercile_threshold():
    curves, terciles = _curves_and_terciles()
    thresholds = {"dark": 0.4, "medium": 0.5, "light": 0.5}
    adapted = lt.adapted_dice(curves, terciles, thresholds, ["dark_0", "light_0"])
    assert adapted == {"dark_0": 1.0, "light_0": 1.0}


def test_no_change_means_no_reduction_and_a_zero_width_interval():
    fixed_dark, fixed_rest = np.array([0.6, 0.7, 0.5]), np.array([0.9, 0.8, 0.85, 0.95])
    out = lt.gap_reduction(fixed_dark, fixed_dark, fixed_rest, fixed_rest, 200, 1)
    assert out["reduction"]["delta"] == 0.0
    assert out["reduction"]["ci95_low"] == 0.0 and out["reduction"]["ci95_high"] == 0.0


def test_the_bootstrap_is_paired_so_a_constant_dark_gain_has_no_spread():
    rng = np.random.RandomState(3)
    fixed_dark, fixed_rest = rng.uniform(0.3, 0.6, 150), rng.uniform(0.8, 0.95, 300)
    out = lt.gap_reduction(fixed_dark, fixed_dark + 0.03, fixed_rest, fixed_rest, 300, 20260923)
    red = out["reduction"]
    assert red["delta"] == pytest.approx(0.03)
    assert red["ci95_low"] == pytest.approx(0.03) and red["ci95_high"] == pytest.approx(0.03)
    assert out["gap_fixed"]["delta"] < out["gap_adapted"]["delta"] < 0
    assert out["n_dark"] == 150 and out["n_rest"] == 300


def test_the_global_change_is_the_mean_paired_difference():
    fixed = np.array([0.8, 0.7, 0.9, 0.6])
    out = lt.global_change(fixed, fixed - 0.001, 200, 1)
    assert out["delta"] == pytest.approx(-0.001)
    assert out["ci95_low"] == pytest.approx(-0.001) and out["n"] == 4


@pytest.mark.parametrize("low, expected", [(0.02, lt.GAP_FULFILLED), (0.0199, lt.GAP_PARTIAL),
                                           (0.0, lt.GAP_NOT_FULFILLED), (-0.01, lt.GAP_NOT_FULFILLED)])
def test_the_gap_verdict_reads_the_lower_end_of_the_interval(low, expected):
    assert lt.gap_verdict(low) == expected


@pytest.mark.parametrize("low, expected", [(-0.005, lt.GLOBAL_HOLDS), (0.01, lt.GLOBAL_HOLDS),
                                           (-0.0051, lt.GLOBAL_FAILS)])
def test_the_global_verdict_tolerates_a_drop_of_half_a_hundredth(low, expected):
    assert lt.global_verdict(low) == expected


def test_the_prediction_needs_both_parts_and_enough_images():
    assert lt.prediction_verdict(lt.GAP_FULFILLED, lt.GLOBAL_HOLDS, 100, 100) == lt.FULFILLED
    assert lt.prediction_verdict(lt.GAP_FULFILLED, lt.GLOBAL_FAILS, 500, 500) == lt.NOT_FULFILLED
    assert lt.prediction_verdict(lt.GAP_PARTIAL, lt.GLOBAL_HOLDS, 500, 500) == lt.NOT_FULFILLED
    assert lt.prediction_verdict(lt.GAP_FULFILLED, lt.GLOBAL_HOLDS, 99, 500) == lt.NO_POWER


def _analysis_inputs():
    rng = np.random.RandomState(7)
    curves, terciles = {}, {}
    for tercile, inside, n in (("dark", 0.42, 120), ("medium", 0.7, 90), ("light", 0.9, 90)):
        for k in range(n):
            image_id = f"{tercile}_{k}"
            curves[image_id] = lt.dice_curve(*_lesion(inside + rng.uniform(-0.02, 0.02)))
            terciles[image_id] = tercile
    curves["no_bin"] = lt.dice_curve(*_lesion(0.9))
    terciles["no_bin"] = "unavailable"
    ids = sorted(curves)
    return curves, terciles, ids[::2], ids[1::2]


def test_the_analysis_fits_on_one_part_scores_the_other_and_counts_what_has_no_tercile():
    curves, terciles, fit, evaluation = _analysis_inputs()
    doc = lt.analyze(curves, terciles, fit, evaluation, n_resamples=100, seed=1, min_fit_n=10,
                     with_verdict=False)
    assert doc["thresholds"]["dark"]["threshold"] < 0.5
    ev = doc["evaluation"]
    assert ev["n_without_tercile"] + ev["global"]["n"] == len(evaluation)
    assert ev["gap"]["reduction"]["delta"] > 0.02
    assert "verdict" not in ev


def test_the_analysis_refuses_images_on_both_sides():
    curves, terciles, fit, evaluation = _analysis_inputs()
    with pytest.raises(ValueError, match="ajuste y evaluacion"):
        lt.analyze(curves, terciles, fit, [*evaluation, fit[0]], n_resamples=10, seed=1, min_fit_n=10)
