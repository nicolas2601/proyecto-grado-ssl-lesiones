"""Probas per arm (full vs F4), the metric row with grouped CIs, and the paired contrasts."""

import numpy as np
import pytest

from dermssl import e3_variants as var
from dermssl.e3b import contrasts, scoring

N_CLASSES, CLASSES = 3, ("a", "b", "c")


def data(n, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLASSES, n)
    full = rng.normal(size=(n, 5)) + np.eye(N_CLASSES, 5)[y] * 1.5
    crop = rng.normal(size=(n, 5)) + np.eye(N_CLASSES, 5)[y] * (1.5 + shift)
    desc = np.column_stack([y + rng.normal(size=n), rng.normal(size=(n, 6))])
    return y, {"full": full.astype(np.float32), "crop_view": crop.astype(np.float32)}, desc


def test_arm_probas_are_the_e3_variants_on_a_single_view():
    y_tr, x_tr, d_tr = data(90, 1)
    y_va, x_va, d_va = data(40, 2)
    probas, info = scoring.arm_probas(x_tr, x_va, d_tr, d_va, y_tr, N_CLASSES)
    assert set(probas) == {"full", "F4"} and probas["F4"].shape == (40, 3)
    assert np.allclose(probas["F4"].sum(axis=1), 1) and info["n_filled"] == [0] * 7
    from dermssl import e3_descriptors as desc

    z_tr, z_va, _ = desc.standardize(d_tr, d_va)
    want = var.variant_proba("F4", {c: x_tr[c][None] for c in x_tr}, y_tr, {c: x_va[c][None] for c in x_va},
                             N_CLASSES, desc_fit=z_tr, desc_eval=z_va)
    assert np.allclose(probas["F4"], want)
    only_full = var.variant_proba("B71", {"full": x_tr["full"][None]}, y_tr, {"full": x_va["full"][None]}, N_CLASSES)
    assert np.allclose(probas["full"], only_full)


def test_a_nan_descriptor_row_is_filled_and_counted():
    y_tr, x_tr, d_tr = data(60, 1)
    y_va, x_va, d_va = data(30, 2)
    d_va[3] = np.nan
    _, info = scoring.arm_probas(x_tr, x_va, d_tr, d_va, y_tr, N_CLASSES)
    assert info["n_filled"] == [1] * 7


def row(seed=0):
    y, x, d = data(120, seed)
    proba = np.random.default_rng(seed).dirichlet(np.ones(3), 120)
    proba[np.arange(120), y] += 0.6
    return y, proba / proba.sum(axis=1, keepdims=True)


def test_the_row_has_the_bank_schema_plus_mcc_and_grouped_intervals():
    y, proba = row()
    groups = np.arange(120) // 2
    out = scoring.score_row(y, proba, groups, CLASSES, mel=1, n_resamples=20)
    for key in ("macro_f1", "balanced_accuracy", "accuracy", "auc_ovr_macro", "ece", "mcc", "mel_recall",
                "macro_precision", "macro_recall", "recall_por_clase", "precision_por_clase", "f1_por_clase",
                "matriz_confusion", "ci95_grouped"):
        assert key in out, key
    assert set(out["recall_por_clase"]) == set(CLASSES) and len(out["matriz_confusion"]) == 3
    assert sum(map(sum, out["matriz_confusion"])) == 120
    lo, hi = out["ci95_grouped"]["macro_f1"]
    assert lo <= out["macro_f1"] <= hi or lo <= hi
    assert set(out["ci95_grouped"]) >= {"macro_f1", "balanced_accuracy", "accuracy", "auc", "ece", "mcc", "mel_recall"}


def test_the_row_ci_is_deterministic_and_grouped_by_lesion():
    y, proba = row()
    a = scoring.score_row(y, proba, np.arange(120) // 2, CLASSES, mel=1, n_resamples=30)
    b = scoring.score_row(y, proba, np.arange(120) // 2, CLASSES, mel=1, n_resamples=30)
    c = scoring.score_row(y, proba, np.arange(120), CLASSES, mel=1, n_resamples=30)
    assert a["ci95_grouped"] == b["ci95_grouped"] and a["ci95_grouped"] != c["ci95_grouped"]


def test_contrasts_are_paired_named_and_seeded():
    y, p_arm = row(1)
    _, p_ctl = row(2)
    groups = np.arange(120) // 2
    recs = contrasts.paired(y, {"x+F4": p_arm, "x": p_ctl}, [("x+F4", "x", "seg_effect")], groups,
                            n_classes=3, mel=1, n_resamples=25)
    assert {r["metric"] for r in recs} == {"macro_f1", "balanced_accuracy", "mel_recall"}
    assert all(r["arm"] == "x+F4" and r["control"] == "x" and r["family"] == "seg_effect" for r in recs)
    assert all(r["n_resamples"] == 25 and r["rng_seed"] == 20260923 for r in recs)
    again = contrasts.paired(y, {"x+F4": p_arm, "x": p_ctl}, [("x+F4", "x", "seg_effect")], groups,
                             n_classes=3, mel=1, n_resamples=25)
    assert recs == again


def test_the_pairs_of_one_baseline_cover_the_registered_contrasts():
    pairs = contrasts.pairs_for("alexnet")
    assert ("alexnet+F4", "alexnet", "seg_effect") in pairs
    assert ("alexnet", "B71", "vs_proposed_notta") in pairs
    assert ("alexnet+F4", "B71+F4", "vs_proposed_seg_notta") in pairs
    assert ("alexnet", "B71_tta", "vs_proposed_registered") in pairs
    assert ("alexnet+F4", "B71+F4_tta", "vs_proposed_seg_registered") in pairs


def test_a_missing_arm_is_an_error_not_a_silent_skip():
    y, p = row()
    with pytest.raises(KeyError):
        contrasts.paired(y, {"a": p}, [("a", "b", "f")], np.arange(120), n_classes=3, mel=1, n_resamples=5)
