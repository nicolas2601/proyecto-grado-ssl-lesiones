"""Dice by luminance tercile and the dark-minus-rest contrast registered as T2 (docs/27)."""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import luminance_dice as ld


def _rows(dice, hd95=None, prefix="img"):
    hd95 = [10.0] * len(dice) if hd95 is None else hd95
    return {f"{prefix}{i}": {"dice": d, "hd95": h} for i, (d, h) in enumerate(zip(dice, hd95))}


def _terciles(labels, prefix="img"):
    return {f"{prefix}{i}": t for i, t in enumerate(labels)}


@pytest.mark.parametrize("ci_high, expected", [
    (-0.03, "cumplida"), (-0.0201, "cumplida"),
    (-0.02, "distinguible pero no clara"), (-0.001, "distinguible pero no clara"),
    (0.0, "no cumplida"), (0.05, "no cumplida"),
])
def test_the_verdict_follows_the_registered_upper_bound(ci_high, expected):
    assert ld.verdict(ci_high, n_dark=100, n_rest=100) == expected


@pytest.mark.parametrize("n_dark, n_rest", [(99, 500), (500, 99)])
def test_a_group_under_100_images_has_no_verdict(n_dark, n_rest):
    assert ld.verdict(-0.5, n_dark=n_dark, n_rest=n_rest) == "sin potencia"


def test_the_contrast_is_unpaired_and_stratified_by_group():
    dark, rest = np.full(150, 0.6), np.full(400, 0.8)
    out = ld.unpaired_contrast(dark, rest, n_resamples=200, seed=1)
    # constant groups: resampling within each group cannot move either mean
    assert out["delta"] == pytest.approx(-0.2)
    assert out["ci95_low"] == pytest.approx(-0.2) and out["ci95_high"] == pytest.approx(-0.2)
    assert out["n_a"] == 150 and out["n_b"] == 400


def test_the_contrast_is_reproducible_with_the_seed_and_moves_with_it():
    rng = np.random.default_rng(0)
    dark, rest = rng.uniform(0, 1, 200), rng.uniform(0, 1, 300)
    first = ld.unpaired_contrast(dark, rest, n_resamples=300, seed=20260923)
    assert first == ld.unpaired_contrast(dark, rest, n_resamples=300, seed=20260923)
    assert first["ci95_low"] != ld.unpaired_contrast(dark, rest, n_resamples=300, seed=7)["ci95_low"]


def test_coverage_counts_every_id_once_including_the_ones_without_tercile():
    rows = _rows([0.5] * 6)
    terciles = {"img0": "dark", "img1": "medium", "img2": "light", "img3": "unavailable",
                "img4": "dark", "other": "light"}
    assert ld.coverage(rows, terciles) == {"dark": 2, "medium": 1, "light": 1, "unavailable": 1,
                                           "not_in_manifest": 1, "total": 6}


def test_an_unknown_tercile_label_is_refused():
    with pytest.raises(ValueError, match="tercil"):
        ld.coverage(_rows([0.5]), {"img0": "oscuro"})


def test_a_clear_gap_is_read_as_fulfilled_with_the_rest_pooled():
    rng = np.random.default_rng(3)
    dice = np.concatenate([rng.normal(0.70, 0.05, 200), rng.normal(0.80, 0.05, 200),
                           rng.normal(0.84, 0.05, 200)])
    labels = ["dark"] * 200 + ["medium"] * 200 + ["light"] * 200
    out = ld.analyze(_rows(dice), _terciles(labels), n_resamples=500, seed=20260923)
    contrast = out["dark_minus_rest"]["dice"]
    assert contrast["delta"] == pytest.approx(dice[:200].mean() - dice[200:].mean())
    assert contrast["verdict"] == "cumplida"
    assert out["bins"]["dark"]["n"] == 200 and out["rest_n"] == 400
    assert out["dark_minus_global"]["dice"]["delta"] == pytest.approx(dice[:200].mean() - dice.mean())


def test_no_gap_is_not_fulfilled():
    rng = np.random.default_rng(4)
    dice = rng.normal(0.8, 0.05, 600)
    labels = ["dark", "medium", "light"] * 200
    out = ld.analyze(_rows(dice), _terciles(labels), n_resamples=500, seed=20260923)
    assert out["dark_minus_rest"]["dice"]["verdict"] in ("no cumplida", "distinguible pero no clara")
    assert out["dark_minus_rest"]["dice"]["ci95_high"] > -0.02


def test_hd95_drops_non_finite_values_and_counts_them_per_bin():
    dice = [0.5] * 6
    hd95 = [10.0, float("nan"), 30.0, 20.0, float("nan"), 40.0]
    labels = ["dark", "dark", "dark", "medium", "light", "light"]
    out = ld.analyze(_rows(dice, hd95), _terciles(labels), n_resamples=50, seed=1)
    assert out["bins"]["dark"]["hd95"]["n"] == 2 and out["bins"]["dark"]["hd95"]["n_dropped_non_finite"] == 1
    assert out["bins"]["light"]["hd95"]["n_dropped_non_finite"] == 1
    assert out["dark_minus_rest"]["hd95"]["delta"] == pytest.approx(20.0 - 30.0)
    assert "verdict" not in out["dark_minus_rest"]["hd95"]


def test_images_without_tercile_stay_out_of_the_contrast():
    dice = [0.2, 0.9, 0.9, 0.1]
    labels = {"img0": "dark", "img1": "medium", "img2": "light", "img3": "unavailable"}
    out = ld.analyze(_rows(dice), labels, n_resamples=50, seed=1)
    assert out["bins"]["dark"]["n"] == 1 and out["rest_n"] == 2
    assert out["dark_minus_rest"]["dice"]["verdict"] == "sin potencia"
