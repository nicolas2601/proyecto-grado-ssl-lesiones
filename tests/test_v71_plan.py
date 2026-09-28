"""The run plan of v7.1: holdout identity, pre-test, selection, the val read, cost.

These are the decisions that must hold before any GPU time: the holdout is the
docs/28 one and nothing else, the pre-test aborts on a real difference, each
arm is chosen among its own configurations, and the single val read contains
exactly the configurations the selection named.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import c1_split_identity as si
from dermssl import selection_holdout as sh
from dermssl import v71_plan as plan
from dermssl import v71_protocol as vp


def _split(n: int = 200, seed: int = 0):
    rng = np.random.RandomState(seed)
    ids = [f"ISIC_{i:05d}" for i in range(n)]
    labels = list(rng.randint(0, 7, n))
    groups = [f"L{i // 2}" for i in range(n)]
    return ids, labels, groups


def _holdout_doc(ids, labels, groups):
    fit, hold = sh.grouped_holdout(ids, labels, groups)
    return {"fit_ids": fit, "holdout_ids": hold, "fit_ids_sha256": si.ids_sha256(fit),
            "holdout_ids_sha256": si.ids_sha256(hold), "seed": 42, "n_splits": 5}


class TestHoldout:
    def test_the_rebuilt_holdout_passes(self):
        ids, labels, groups = _split()
        doc = _holdout_doc(ids, labels, groups)
        assert plan.check_holdout(doc, ids, labels, groups, val_ids=["X"]) == doc["holdout_ids_sha256"]

    def test_a_different_holdout_is_refused(self):
        ids, labels, groups = _split()
        doc = _holdout_doc(ids, labels, groups)
        doc["holdout_ids"] = doc["holdout_ids"][1:] + [doc["fit_ids"][0]]
        doc["holdout_ids_sha256"] = si.ids_sha256(doc["holdout_ids"])
        with pytest.raises(ValueError, match="no es el de docs/28"):
            plan.check_holdout(doc, ids, labels, groups, val_ids=[])

    def test_a_stale_digest_is_refused(self):
        ids, labels, groups = _split()
        doc = _holdout_doc(ids, labels, groups)
        doc["fit_ids_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="digest"):
            plan.check_holdout(doc, ids, labels, groups, val_ids=[])

    def test_val_ids_in_the_holdout_are_refused(self):
        ids, labels, groups = _split()
        doc = _holdout_doc(ids, labels, groups)
        with pytest.raises(ValueError, match="validacion"):
            plan.check_holdout(doc, ids, labels, groups, val_ids=[doc["holdout_ids"][0]])


class TestLimit:
    def test_limit_keeps_every_class_and_is_deterministic(self):
        ids, labels, _ = _split(300)
        picked = plan.limited_ids(ids, labels, 21)
        assert len(picked) == 21
        assert {labels[ids.index(i)] for i in picked} == set(range(7))
        assert picked == plan.limited_ids(ids, labels, 21)

    def test_no_limit_returns_everything(self):
        ids, labels, _ = _split(30)
        assert plan.limited_ids(ids, labels, None) == ids


class TestPretest:
    def test_identical_metrics_pass(self):
        m = {"macro_f1": 0.61, "ece": 0.02}
        verdict = plan.pretest_check("A", 0.0, m, dict(m), weights_diff=0.0)
        assert verdict["passed"] is True

    def test_a_metric_off_by_more_than_the_tolerance_fails(self):
        verdict = plan.pretest_check("A", 1.0, {"macro_f1": 0.61}, {"macro_f1": 0.61 + 2e-6},
                                     weights_diff=0.0)
        assert verdict["passed"] is False

    def test_a_weight_difference_fails_even_with_equal_metrics(self):
        verdict = plan.pretest_check("B", 0.0, {"macro_f1": 0.6}, {"macro_f1": 0.6}, weights_diff=1e-7)
        assert verdict["passed"] is False


def _row(f1, ece, mel80=0.6):
    return {"macro_f1": f1, "ece": ece, "mel_sens_at_spec80": mel80}


def _grid():
    return {
        vp.config_id("public", 224, "cls", "none", False): _row(0.61, 0.02),
        vp.config_id("public", 448, "cls_gem", "balanced", True): _row(0.66, 0.03),
        vp.config_id("A_s42_a1.0", 224, "cls", "none", False): _row(0.62, 0.02),
        vp.config_id("A_soup_a0.6", 448, "cls4", "la1.0", True): _row(0.68, 0.05, 0.72),
        vp.config_id("B_s42_a1.0", 224, "cls", "none", False): _row(0.63, 0.02),
        vp.config_id("B_s42_a0.4", 224, "cls_attn", "balanced", False): _row(0.66, 0.01, 0.65),
    }


class TestSelection:
    def test_winners_per_arm_and_public_matched_protocol(self):
        doc = plan.build_selection(_grid())
        assert doc["winners"]["A"] == vp.config_id("A_soup_a0.6", 448, "cls4", "la1.0", True)
        assert doc["winners"]["B"] == vp.config_id("B_s42_a0.4", 224, "cls_attn", "balanced", False)
        assert doc["winners"]["public"] == vp.config_id("public", 448, "cls_gem", "balanced", True)
        assert doc["phase2"]["enabled"] is False  # A passes 0.70 at spec 80 on the holdout

    def test_val_entries_are_the_named_configurations(self):
        entries = plan.val_entries(plan.build_selection(_grid()))
        assert entries["A71"] == vp.config_id("A_soup_a0.6", 448, "cls4", "la1.0", True)
        assert entries["public_A71"] == vp.config_id("public", 448, "cls4", "la1.0", True)
        assert entries["A_v7"] == vp.primary_config("A")
        assert entries["public_v7"] == vp.primary_config("public")
        assert set(entries) == set(plan.VAL_LABELS)

    def test_units_needed_for_val(self):
        units = plan.val_units(plan.val_entries(plan.build_selection(_grid())))
        assert units[("A_soup_a0.6", 448)] == {"features": {"cls4"}, "tta": True}
        assert units[("public", 224)]["tta"] is False

    def test_contrasts_only_name_known_entries(self):
        assert all(a in plan.VAL_LABELS and c in plan.VAL_LABELS for a, c in plan.VAL_CONTRASTS)


class TestCost:
    def test_projection_scales_with_units_and_resolution(self):
        small = plan.cost_projection(n_fit=100, n_hold=25, n_train=125, n_val=30,
                                     resolutions=(224,), n_weights=21)
        big = plan.cost_projection(n_fit=100, n_hold=25, n_train=125, n_val=30,
                                   resolutions=(224, 448), n_weights=21)
        assert big["gpu_hours"] > small["gpu_hours"] > 0
        assert big["cpu_hours"] > small["cpu_hours"] > 0
        assert big["n_units"] == 42 and big["n_configs"] == 42 * 60

    def test_the_forward_count_includes_tta_on_the_holdout_only(self):
        cost = plan.cost_projection(n_fit=100, n_hold=10, n_train=110, n_val=20,
                                    resolutions=(224,), n_weights=1)
        assert cost["forwards"]["grid"][224] == 100 + 8 * 10
