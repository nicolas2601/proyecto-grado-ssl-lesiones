"""docs/29 §2.1, §2.2, §2.4 and §2.7: the v7.1 grid, as registered.

The grid is data, so what is pinned here is that it matches the plan: the
alphas, the tau values, the feature list, and the selection rule reused from
docs/28 §4.1 without a second implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import v71_protocol as vp


class TestRegisteredGrid:
    def test_alphas_and_taus_are_the_registered_ones(self):
        assert vp.ALPHAS == (0.2, 0.4, 0.6, 0.8, 1.0)
        assert vp.TAUS == (0.5, 1.0, 1.5, 2.0)  # tau = 0 is the flat probe, not a copy of it

    def test_features_cover_section_2_2_and_2_7(self):
        assert vp.FEATURES == ("cls", "cls_mean", "cls4", "cls_gem", "cls_attn")

    def test_weight_configs_per_arm(self):
        names = vp.weight_configs("A")
        assert len(names) == 10
        assert "A_s42_a1.0" in names and "A_soup_a0.2" in names
        assert not any(n.startswith("B") for n in names)

    def test_weightings_are_none_balanced_and_each_tau(self):
        assert vp.WEIGHTINGS == ("none", "balanced", "la0.5", "la1.0", "la1.5", "la2.0")

    def test_config_id_round_trip(self):
        cid = vp.config_id("B_soup_a0.6", 448, "cls_gem", "la1.5", True)
        assert vp.parse_config_id(cid) == {
            "weights": "B_soup_a0.6", "resolution": 448, "feature": "cls_gem",
            "weighting": "la1.5", "tta": True,
        }

    def test_primary_protocol_is_the_v7_one(self):
        assert vp.parse_config_id(vp.primary_config("A")) == {
            "weights": "A_s42_a1.0", "resolution": 224, "feature": "cls",
            "weighting": "none", "tta": False,
        }

    def test_arm_of_a_config(self):
        assert vp.arm_of("A_soup_a0.4") == "A"
        assert vp.arm_of("public") == "public"


class TestLogitAdjustment:
    def test_tau_zero_is_the_identity(self):
        proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        assert np.allclose(vp.logit_adjust(proba, np.array([0.6, 0.3, 0.1]), 0.0), proba)

    def test_adjustment_moves_mass_to_rare_classes(self):
        proba = np.array([[0.5, 0.5]])
        prior = np.array([0.9, 0.1])
        adjusted = vp.logit_adjust(proba, prior, 1.0)
        # softmax(log p - log pi) is proportional to p / pi.
        assert adjusted[0] == pytest.approx(np.array([0.5 / 0.9, 0.5 / 0.1]) / (0.5 / 0.9 + 5.0))

    def test_prior_is_the_training_frequency(self):
        assert np.allclose(vp.class_prior(np.array([0, 0, 1, 2]), 3), [0.5, 0.25, 0.25])

    def test_absent_class_in_the_prior_is_refused(self):
        with pytest.raises(ValueError, match="sin ejemplos"):
            vp.class_prior(np.array([0, 0, 1]), 3)


class TestSelection:
    def test_rule_is_docs28_macro_f1_then_ece(self):
        rows = {"a": {"macro_f1": 0.700, "ece": 0.05},
                "b": {"macro_f1": 0.697, "ece": 0.02},
                "c": {"macro_f1": 0.690, "ece": 0.01}}
        assert vp.select_for_arm(rows, "A", candidates=["a", "b", "c"]) == "b"

    def test_candidates_are_the_arm_configs_only(self):
        grid = {
            vp.config_id("A_s42_a1.0", 224, "cls", "none", False): {"macro_f1": 0.6, "ece": 0.1},
            vp.config_id("B_s42_a1.0", 224, "cls", "none", False): {"macro_f1": 0.9, "ece": 0.1},
            vp.config_id("public", 224, "cls", "none", False): {"macro_f1": 0.95, "ece": 0.1},
        }
        assert vp.parse_config_id(vp.select_for_arm(grid, "A"))["weights"] == "A_s42_a1.0"

    def test_phase2_condition_is_no_arm_above_070(self):
        assert vp.phase2_enabled({"A": 0.70, "B": 0.65}) is True
        assert vp.phase2_enabled({"A": 0.71, "B": 0.65}) is False
