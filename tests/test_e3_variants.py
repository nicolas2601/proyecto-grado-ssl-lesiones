"""Building the registered E3 variants from per-condition features (synthetic, no extractor)."""

import numpy as np
import pytest

from dermssl import e3_variants as v
from dermssl import v71_grid as vg

N_VIEWS, N_CLASSES, DIM = 2, 3, 6


def make_feats(n, seed, conditions=None, views=N_VIEWS, y=None):
    """Class-dependent features per condition so a probe has something to learn."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n) if y is None else y
    out = {}
    for name in conditions or v.ALL_CONDITIONS:
        base = np.eye(N_CLASSES)[y] @ rng.normal(size=(N_CLASSES, DIM))
        out[name] = base[None] + 0.3 * rng.normal(size=(views, n, DIM))
    return out, y


def test_each_variant_declares_the_conditions_it_needs():
    assert v.required_conditions("B71") == ("full",)
    assert v.required_conditions("F1") == ("full", "crop_view")
    assert v.required_conditions("F2") == ("full", "crop_orig224")
    assert v.required_conditions("F2_448") == ("full", "crop_orig448")
    assert v.required_conditions("F3") == ("full", "shift0", "shift1", "shift2", "shift3")
    assert v.required_conditions("F4") == ("full", "crop_view")
    assert v.required_conditions("F5") == ("full", "crop_view", "ring")
    assert v.VARIANTS == ("B71", "F1", "F2", "F2_448", "F3", "F4", "F5")


def test_unknown_variant_is_refused():
    with pytest.raises(v.UnknownVariantError):
        v.required_conditions("F9")


def test_concat_is_full_then_crop_and_f4_appends_the_descriptors_to_every_view():
    feats, _ = make_feats(5, 0)
    f1 = v.concat_features("F1", feats)
    assert f1.shape == (N_VIEWS, 5, 2 * DIM)
    assert (f1[..., :DIM] == feats["full"]).all() and (f1[..., DIM:] == feats["crop_view"]).all()
    desc = np.arange(5 * 7, dtype=float).reshape(5, 7)
    f4 = v.concat_features("F4", feats, desc)
    assert f4.shape == (N_VIEWS, 5, 2 * DIM + 7)
    for view in range(N_VIEWS):
        assert (f4[view, :, -7:] == desc).all()
    assert v.concat_features("F5", feats).shape[-1] == 3 * DIM


def test_f4_without_descriptors_and_missing_conditions_are_refused():
    feats, _ = make_feats(4, 1)
    with pytest.raises(v.MissingInputError):
        v.concat_features("F4", feats)
    del feats["crop_view"]
    with pytest.raises(v.MissingInputError):
        v.concat_features("F1", feats)


def test_a_different_number_of_images_between_conditions_is_refused():
    feats, _ = make_feats(4, 2)
    feats["crop_view"] = feats["crop_view"][:, :3]
    with pytest.raises(v.MissingInputError):
        v.concat_features("F1", feats)


def run(name, fit, y_fit, ev, desc=(None, None)):
    return v.variant_proba(name, fit, y_fit, ev, N_CLASSES, desc_fit=desc[0], desc_eval=desc[1])


def test_b71_is_the_tta_average_of_the_balanced_probe_on_the_full_features():
    fit, y_fit = make_feats(60, 3, ("full",), views=1)
    ev, _ = make_feats(20, 4, ("full",))
    proba = run("B71", fit, y_fit, ev)
    direct = vg.fit_predict_views(fit["full"][0], y_fit, ev["full"], N_CLASSES, "balanced").mean(axis=0)
    assert np.allclose(proba, direct) and np.allclose(proba.sum(axis=1), 1.0)


def test_f1_scores_a_probe_fitted_on_the_identity_view_of_the_concatenation():
    fit, y_fit = make_feats(60, 5, views=1)
    ev, _ = make_feats(20, 6)
    proba = run("F1", fit, y_fit, ev)
    x_fit = v.concat_features("F1", fit)[0]
    direct = vg.fit_predict_views(x_fit, y_fit, v.concat_features("F1", ev), N_CLASSES, "balanced")
    assert np.allclose(proba, direct.mean(axis=0))


def test_f3_averages_the_full_image_and_the_four_shifted_crops_with_one_full_image_probe():
    fit, y_fit = make_feats(60, 7, ("full",), views=1)  # no crop features at fit time at all
    ev, _ = make_feats(20, 8)
    proba = run("F3", fit, y_fit, ev)
    members = v.f3_members(fit, y_fit, ev, N_CLASSES)
    assert members.shape == (5, 20, N_CLASSES)
    assert np.allclose(proba, members.mean(axis=0))
    assert np.allclose(members[0], run("B71", fit, y_fit, ev))  # member 0 is exactly B71


def test_f4_and_f5_run_and_return_distributions():
    fit, y_fit = make_feats(60, 9, views=1)
    ev, _ = make_feats(20, 10)
    rng = np.random.default_rng(0)
    desc = (rng.normal(size=(60, 7)), rng.normal(size=(20, 7)))
    for name in ("F4", "F5", "F2", "F2_448"):
        proba = run(name, fit, y_fit, ev, desc)
        assert proba.shape == (20, N_CLASSES) and np.allclose(proba.sum(axis=1), 1.0), name
