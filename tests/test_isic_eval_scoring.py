"""Protocol registry, MCC, grouped bootstrap and paired contrasts of the ISIC evaluation."""

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

from dermssl import v71_grid as vg
from dermssl.isic_eval import protocol as pr
from dermssl.isic_eval import scoring as sc


def _proba(y, n, acc, rng):
    p = rng.dirichlet(np.ones(n), len(y)) * 0.2
    pred = np.where(rng.rand(len(y)) < acc, y, (y + 1) % n)
    p[np.arange(len(y)), pred] += 1.0
    return p / p.sum(1, keepdims=True)


def test_registered_config_ids_are_the_v71_strings():
    assert pr.CONFIGS["B71"] == "B_s42_a1.0@224|cls4|balanced|tta"
    assert pr.CONFIGS["A71"] == "A_soup_a1.0@224|cls_attn|la0.5|tta"
    assert pr.CONFIGS["PUBLIC_B71"] == "public@224|cls4|balanced|tta"
    assert pr.CONFIGS["PUBLIC_A71"] == "public@224|cls_attn|la0.5|tta"


def test_parts_needed_compose_into_the_registered_feature():
    for config in pr.CONFIGS:
        feature = pr.parse(config)["feature"]
        parts = {p: np.zeros((1, 2, 3)) for p in pr.parts_needed(config)}
        assert vg.compose(parts, feature).shape[-1] == 3 * len(parts)


def test_transductive_label_follows_the_pool_flag_and_never_marks_public():
    assert pr.transductive("B71", False) and not pr.transductive("B71", True)
    assert not pr.transductive("PUBLIC_A71", False)
    assert pr.cell_name("A71", True) == "A71__clean" and pr.cell_name("PUBLIC_B71", True) == "PUBLIC_B71"


def test_contrast_pairs_cover_public_other_config_and_clean_vs_original():
    pairs = pr.contrast_pairs()
    assert ("B71__clean", "B71__original") in pairs and ("B71__clean", "A71__clean") in pairs
    assert ("A71__original", "PUBLIC_A71") in pairs and len(pairs) == len(set(pairs)) == 8


def test_mcc_equals_scikit_learn_on_a_multiclass_sample():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 5, 400)
    proba = _proba(y, 5, 0.6, rng)
    assert sc.mcc(y, proba) == pytest.approx(matthews_corrcoef(y, proba.argmax(1)))


def test_mcc_from_confusion_matrix_of_the_score_block_is_consistent():
    from dermssl.bench import metrics as bm

    rng = np.random.RandomState(1)
    y = rng.randint(0, 3, 200)
    row = sc.score(y, _proba(y, 3, 0.7, rng), ("a", "b", "c"), n_resamples=20)
    assert bm.mcc_from_confusion(row["matriz_confusion"]) == pytest.approx(row["mcc"])
    assert [row["per_class"][c]["support"] for c in "abc"] == np.bincount(y, minlength=3).tolist()


def test_grouped_bootstrap_by_patient_is_wider_and_deterministic():
    y = np.zeros(100, dtype=int)
    proba = np.tile([1.0, 0.0], (100, 1))
    proba[np.repeat(np.arange(20) % 2 == 1, 5)] = [0.0, 1.0]  # odd patients all wrong
    groups = np.repeat(np.arange(20), 5)
    grouped = sc.bootstrap_ci(y, proba, groups=groups, n_resamples=500)["accuracy"]
    by_image = sc.bootstrap_ci(y, proba, n_resamples=500)["accuracy"]
    assert grouped == sc.bootstrap_ci(y, proba, groups=groups, n_resamples=500)["accuracy"]
    assert grouped[1] - grouped[0] > 1.5 * (by_image[1] - by_image[0])
    assert all(round(v * 20, 9) == round(v * 20) for v in grouped)  # whole patients (5 rows each) only


def test_score_declares_the_resampling_unit():
    rng = np.random.RandomState(2)
    y = np.array([0, 1] * 30)
    proba = _proba(y, 2, 0.8, rng)
    assert sc.score(y, proba, ("n", "p"), n_resamples=10)["bootstrap"]["unit"] == "image"
    assert sc.score(y, proba, ("n", "p"), groups=np.repeat(np.arange(30), 2), n_resamples=10)["bootstrap"]["unit"] == "patient_id"


def test_paired_contrast_flags_a_better_arm_and_ties_identical_arms():
    rng = np.random.RandomState(3)
    y = rng.randint(0, 2, 600)
    good, bad = _proba(y, 2, 0.95, rng), _proba(y, 2, 0.6, rng)
    rows = {r["metric"]: r for r in sc.contrasts("good", "bad", y, good, bad, n_resamples=200)}
    assert rows["macro_f1"]["excludes_zero"] and rows["macro_f1"]["direction"] == "arm_better"
    same = sc.contrasts("a", "b", y, good, good, groups=np.arange(600) // 3, n_resamples=50)
    assert all(r["point_delta"] == 0 and r["direction"] == "indistinguishable" for r in same)
