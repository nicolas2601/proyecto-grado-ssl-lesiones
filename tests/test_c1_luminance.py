"""T3 (docs/28): C1-v7 contrasts by luminance tercile, on synthetic predictions."""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import c1_luminance as cl

MEL = 4
N_CLASSES = 7


def _one_hot(pred: np.ndarray) -> np.ndarray:
    proba = np.full((len(pred), N_CLASSES), 0.01)
    proba[np.arange(len(pred)), pred] = 0.94
    return proba


def _labels(n: int, classes=(0, 4, 5), seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).choice(classes, size=n)


def test_macro_f1_averages_only_the_classes_present_in_the_bin():
    y = np.array([0, 0, 5, 5])
    fn = cl.metric_functions(y, n_classes=N_CLASSES, mel=MEL)["macro_f1"]
    assert fn(y, _one_hot(y)) == pytest.approx(1.0)


def test_balanced_accuracy_and_mel_recall_use_the_bin_classes():
    y = np.array([4, 4, 4, 4, 5, 5])
    pred = np.array([4, 4, 5, 5, 5, 5])
    fns = cl.metric_functions(y, n_classes=N_CLASSES, mel=MEL)
    assert fns["mel_recall"](y, _one_hot(pred)) == pytest.approx(0.5)
    assert fns["balanced_accuracy"](y, _one_hot(pred)) == pytest.approx(0.75)


def test_a_bin_without_melanoma_has_no_mel_recall():
    fns = cl.metric_functions(np.array([0, 5, 5]), n_classes=N_CLASSES, mel=MEL)
    assert "mel_recall" not in fns


def test_identical_conditions_are_indistinguishable_and_a_perfect_arm_improves():
    y = _labels(300)
    terciles = np.array(["dark"] * 300)
    wrong = np.where(y == 5, 0, 5)
    probas = {"none": _one_hot(wrong), "bbox_crop": _one_hot(wrong), "mask_pool": _one_hot(y)}
    out = cl.tercile_contrasts(y, terciles, probas, n_resamples=200, seed=20260923, mel=MEL,
                               n_classes=N_CLASSES)
    dark = out["dark"]
    assert dark["n"] == 300 and dark["n_mel"] == int((y == MEL).sum())
    assert dark["contrasts"]["bbox_crop"]["macro_f1"]["verdict"] == cl.INDISTINGUISHABLE
    assert dark["contrasts"]["mask_pool"]["macro_f1"]["verdict"] == cl.IMPROVES
    assert dark["contrasts"]["mask_pool"]["macro_f1"]["point_delta"] > 0


def test_a_worse_arm_is_reported_as_worse():
    y = _labels(300, seed=1)
    probas = {"none": _one_hot(y), "neutral_bg": _one_hot(np.where(y == 5, 0, 5))}
    out = cl.tercile_contrasts(y, np.array(["dark"] * 300), probas, n_resamples=200, seed=1,
                               mel=MEL, n_classes=N_CLASSES)
    assert out["dark"]["contrasts"]["neutral_bg"]["macro_f1"]["verdict"] == cl.WORSENS


def test_a_bin_under_100_images_has_no_verdict_but_keeps_its_point_delta():
    y = _labels(150, seed=2)
    terciles = np.array(["dark"] * 120 + ["light"] * 30)
    probas = {"none": _one_hot(np.where(y == 5, 0, 5)), "bbox_crop": _one_hot(y)}
    out = cl.tercile_contrasts(y, terciles, probas, n_resamples=100, seed=0, mel=MEL,
                               n_classes=N_CLASSES)
    light = out["light"]["contrasts"]["bbox_crop"]["macro_f1"]
    assert light["verdict"] == cl.NO_POWER and light["point_delta"] > 0
    assert "ci95_low" not in light
    assert out["dark"]["contrasts"]["bbox_crop"]["macro_f1"]["verdict"] == cl.IMPROVES


def test_rows_without_a_tercile_are_left_out_and_counted():
    y = _labels(210, seed=3)
    terciles = np.array(["dark"] * 200 + ["unavailable"] * 10)
    probas = {"none": _one_hot(y), "bbox_crop": _one_hot(y)}
    out = cl.tercile_contrasts(y, terciles, probas, n_resamples=50, seed=0, mel=MEL,
                               n_classes=N_CLASSES)
    assert out["dark"]["n"] == 200 and out["medium"]["n"] == 0
    assert "unavailable" not in out


def test_the_contrast_needs_the_none_baseline():
    with pytest.raises(ValueError, match="none"):
        cl.tercile_contrasts(np.zeros(3, int), np.array(["dark"] * 3), {"bbox_crop": np.eye(7)[:3]},
                             n_resamples=10, seed=0, mel=MEL, n_classes=N_CLASSES)


def _key_cells(verdicts: dict) -> dict:
    """{(segmenter, backbone, condition): verdict} for the 12 key cells."""
    cells = {}
    for seg in cl.SEGMENTERS:
        for bb in cl.KEY_BACKBONES:
            for cond in cl.KEY_CONDITIONS:
                cells[(seg, bb, cond)] = verdicts.get((seg, bb, cond), cl.INDISTINGUISHABLE)
    return cells


def test_the_prediction_holds_when_no_key_cell_improves():
    result = cl.prediction(_key_cells({("student_r3", "v7_astar_s42", "bbox_crop"): cl.WORSENS}))
    assert result["verdict"] == cl.PREDICTION_HOLDS and result["n_key_cells"] == 12
    assert result["improves"] == [] and result["n_worsens"] == 1


def test_one_improving_key_cell_refutes_the_prediction_and_robustness_is_counted():
    single = cl.prediction(_key_cells({("a2", "v7_astar_s42", "mask_pool"): cl.IMPROVES}))
    assert single["verdict"] == cl.PREDICTION_FAILS and single["robust"] == []
    both_seeds = cl.prediction(_key_cells({("a2", "v7_astar_s42", "mask_pool"): cl.IMPROVES,
                                           ("a2", "v7_astar_s43", "mask_pool"): cl.IMPROVES}))
    assert len(both_seeds["robust"]) == 2
    public = cl.prediction(_key_cells({("a2", "dinov2_public_vitb14", "bbox_crop"): cl.IMPROVES,
                                       ("student_r3", "dinov2_public_vitb14", "bbox_crop"): cl.IMPROVES}))
    assert len(public["robust"]) == 2


def test_no_power_in_the_dark_tercile_gives_no_prediction_verdict():
    result = cl.prediction(_key_cells({k: cl.NO_POWER for k in _key_cells({})}))
    assert result["verdict"] == cl.NO_POWER


def test_the_registered_multiplicity_is_what_the_grid_gives():
    assert cl.n_contrasts(n_backbones=5) == 270
    assert len(cl.SEGMENTERS) * len(cl.KEY_BACKBONES) * len(cl.KEY_CONDITIONS) == 12
