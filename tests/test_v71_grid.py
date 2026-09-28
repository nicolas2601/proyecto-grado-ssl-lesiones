"""The probe half of the v7.1 grid: features, class weighting and TTA per unit.

A unit is one set of weights at one resolution. Its primary cell (CLS, flat
logreg, no TTA) must be the C1 ``logreg`` probe exactly, or the v7.1 reading
is not on the same footing as the v7 primary.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import probes
from dermssl import v71_grid as vg
from dermssl import v71_protocol as vp

N_CLASSES, MEL, DIM = 7, 4, 6


def _parts(n: int, views: int, seed: int):
    rng = np.random.RandomState(seed)
    y = np.tile(np.arange(N_CLASSES), n // N_CLASSES + 1)[:n]
    signal = np.eye(N_CLASSES, DIM)[y] * 3
    parts = {
        name: np.stack([signal + rng.normal(size=(n, DIM)) for _ in range(views)]).astype(np.float32)
        for name in ("cls", "mean", "gem", "attn")
    }
    parts["cls4"] = np.concatenate([parts["cls"]] * 4, axis=2)
    return parts, y


@pytest.fixture(scope="module")
def unit():
    fit_parts, y_fit = _parts(140, 1, 0)
    eval_parts, y_eval = _parts(70, 8, 1)
    probas = vg.unit_probas(fit_parts, y_fit, eval_parts, N_CLASSES, n_jobs=1)
    return fit_parts, y_fit, eval_parts, y_eval, probas


def test_every_registered_cell_is_present(unit):
    *_, probas = unit
    expected = {(f, w, t) for f in vp.FEATURES for w in vp.WEIGHTINGS for t in vp.TTA_OPTIONS}
    assert set(probas) == expected


def test_primary_cell_is_the_c1_logreg_probe(unit):
    fit_parts, y_fit, eval_parts, _, probas = unit
    ref = probes.logreg_predict_proba(fit_parts["cls"][0], y_fit, eval_parts["cls"][0], N_CLASSES)
    assert np.array_equal(probas[("cls", "none", False)], ref)


def test_balanced_cell_is_the_balanced_probe(unit):
    fit_parts, y_fit, eval_parts, _, probas = unit
    x_fit = np.concatenate([fit_parts["cls"][0], fit_parts["gem"][0]], axis=1)
    x_eval = np.concatenate([eval_parts["cls"][0], eval_parts["gem"][0]], axis=1)
    ref = probes.logreg_balanced_predict_proba(x_fit, y_fit, x_eval, N_CLASSES)
    assert np.allclose(probas[("cls_gem", "balanced", False)], ref)


def test_tta_averages_the_eight_views(unit):
    fit_parts, y_fit, eval_parts, _, probas = unit
    per_view = [probes.logreg_predict_proba(fit_parts["cls"][0], y_fit, eval_parts["cls"][v], N_CLASSES)
                for v in range(8)]
    assert np.allclose(probas[("cls", "none", True)], np.mean(per_view, axis=0))


def test_logit_adjustment_uses_the_fit_prior(unit):
    _, y_fit, _, _, probas = unit
    prior = vp.class_prior(y_fit, N_CLASSES)
    assert np.allclose(probas[("cls", "la1.5", False)],
                       vp.logit_adjust(probas[("cls", "none", False)], prior, 1.5))


def test_composed_features_are_cls_first():
    parts, _ = _parts(14, 1, 0)
    x = vg.compose(parts, "cls_attn")
    assert x.shape == (1, 14, 2 * DIM)
    assert np.array_equal(x[..., :DIM], parts["cls"])


def test_without_views_tta_cells_are_absent():
    fit_parts, y_fit = _parts(70, 1, 0)
    eval_parts, _ = _parts(35, 1, 1)
    probas = vg.unit_probas(fit_parts, y_fit, eval_parts, N_CLASSES, features=("cls",), n_jobs=1)
    assert {t for _, _, t in probas} == {False}


def test_scored_unit_keys_are_config_ids(unit):
    *_, y_eval, probas = unit
    subset = {k: probas[k] for k in [("cls", "none", False), ("cls4", "la2.0", True)]}
    rows = vg.score_unit("A_s42_a1.0", 224, y_eval, subset, mel=MEL, n_resamples=20)
    cid = vp.config_id("A_s42_a1.0", 224, "cls4", "la2.0", True)
    assert set(rows) == {vp.config_id("A_s42_a1.0", 224, "cls", "none", False), cid}
    assert rows[cid]["ci95"]["macro_f1"][0] <= rows[cid]["macro_f1"] <= rows[cid]["ci95"]["macro_f1"][1]
    assert set(rows[cid]["thresholds"]) == {"spec80", "spec90", "sens95"}
