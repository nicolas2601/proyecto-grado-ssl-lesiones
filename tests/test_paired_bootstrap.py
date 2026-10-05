"""Paired bootstrap for arm-vs-control contrasts.

``resultados/clasificacion/bootstrap.json`` was committed as DATA -- there is
no generator in the repo. So this is written against that file's schema, and
validated by regenerating one of its contrasts from the bank features: the
point estimate does not depend on resampling, so it must come out identical.

The load-bearing property is that the two arms are resampled with the SAME
indices. Drawing independently would add variance that is not in the data and
widen every interval, which turns real differences into "indistinguishable"
-- a conclusion that looks conservative and is simply wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import paired_bootstrap as pb


def _proba(n_rows: int, n_classes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.random((n_rows, n_classes))
    return raw / raw.sum(axis=1, keepdims=True)


class TestTheResamplingIsShared:
    def test_both_arms_see_the_same_indices(self):
        seen: list[np.ndarray] = []

        def spy_metric(y_true, proba):
            seen.append(y_true.copy())
            return float(len(y_true))

        y_true = np.arange(50) % 3
        arm, control = _proba(50, 3, 1), _proba(50, 3, 2)

        pb.paired_contrast(
            y_true=y_true, arm_proba=arm, control_proba=control,
            metric_fn=spy_metric, n_resamples=5, seed=0,
        )

        # Point estimates first (2 calls), then 5 resamples x 2 arms.
        resample_pairs = seen[2:]
        for index in range(0, len(resample_pairs), 2):
            assert np.array_equal(resample_pairs[index], resample_pairs[index + 1]), (
                "los dos brazos vieron remuestras distintas"
            )

    def test_identical_arms_give_a_delta_of_exactly_zero(self):
        """The sharpest consequence of pairing: with the same predictions on
        both sides every resample cancels, so the interval collapses. Drawing
        independently would produce a spread around zero out of nothing."""
        y_true = np.arange(60) % 4
        proba = _proba(60, 4, 3)

        result = pb.paired_contrast(
            y_true=y_true, arm_proba=proba, control_proba=proba.copy(),
            metric_fn=pb.metric_macro_f1, n_resamples=50, seed=0,
        )

        assert result["point_delta"] == 0.0
        assert result["ci95_low"] == 0.0
        assert result["ci95_high"] == 0.0
        assert result["excludes_zero"] is False

    def test_the_seed_makes_it_reproducible(self):
        y_true = np.arange(60) % 4
        arm, control = _proba(60, 4, 5), _proba(60, 4, 6)
        kwargs = dict(
            y_true=y_true, arm_proba=arm, control_proba=control,
            metric_fn=pb.metric_macro_f1, n_resamples=40,
        )

        first = pb.paired_contrast(**kwargs, seed=7)
        again = pb.paired_contrast(**kwargs, seed=7)
        other = pb.paired_contrast(**kwargs, seed=8)

        assert first["ci95_low"] == again["ci95_low"]
        assert first["ci95_high"] == again["ci95_high"]
        assert (first["ci95_low"], first["ci95_high"]) != (other["ci95_low"], other["ci95_high"])


class TestThePointEstimateIsNotResampled:
    def test_the_point_is_the_metric_on_the_full_set(self):
        """It does not depend on the bootstrap at all, which is what makes it
        usable to validate this implementation against the v6 numbers."""
        y_true = np.arange(40) % 3
        arm, control = _proba(40, 3, 9), _proba(40, 3, 10)

        result = pb.paired_contrast(
            y_true=y_true, arm_proba=arm, control_proba=control,
            metric_fn=pb.metric_macro_f1, n_resamples=10, seed=0,
        )

        assert result["point_arm"] == pytest.approx(pb.metric_macro_f1(y_true, arm))
        assert result["point_control"] == pytest.approx(pb.metric_macro_f1(y_true, control))
        assert result["point_delta"] == pytest.approx(
            result["point_arm"] - result["point_control"]
        )

    def test_the_number_of_resamples_does_not_move_the_point(self):
        y_true = np.arange(40) % 3
        arm, control = _proba(40, 3, 11), _proba(40, 3, 12)
        kwargs = dict(
            y_true=y_true, arm_proba=arm, control_proba=control,
            metric_fn=pb.metric_macro_f1, seed=0,
        )

        assert pb.paired_contrast(**kwargs, n_resamples=10)["point_delta"] == pytest.approx(
            pb.paired_contrast(**kwargs, n_resamples=200)["point_delta"]
        )


class TestTheDirectionOfECE:
    """The inverted-label bug, prevented at the source.

    build_results_readme._verdict already refuses to infer a side from the
    sign, and its docstring says why: doing so "is exactly the inverted-label
    bug that hit HD95 and would flip every ECE row". But it READS `direction`
    from the JSON -- so whoever WRITES the JSON has to get it right, and that
    is now this module.
    """

    def test_a_lower_ece_favours_the_arm(self):
        result = pb.direction_for(metric="ece", point_delta=-0.02, excludes_zero=True)

        assert result == "arm_better"

    def test_a_higher_ece_favours_the_control(self):
        assert pb.direction_for(metric="ece", point_delta=0.02, excludes_zero=True) == (
            "control_better"
        )

    def test_a_higher_macro_f1_favours_the_arm(self):
        assert pb.direction_for(metric="macro_f1", point_delta=0.02, excludes_zero=True) == (
            "arm_better"
        )

    def test_a_lower_macro_f1_favours_the_control(self):
        assert pb.direction_for(metric="macro_f1", point_delta=-0.02, excludes_zero=True) == (
            "control_better"
        )

    @pytest.mark.parametrize("metric", ["ece", "macro_f1", "balanced_accuracy", "auc"])
    def test_an_interval_containing_zero_is_indistinguishable_whatever_the_sign(self, metric):
        """A delta with the 'good' sign and an interval spanning zero is not a
        win. Parametrised over ECE too, so the lower-is-better path cannot
        quietly skip this rule."""
        assert pb.direction_for(metric=metric, point_delta=0.05, excludes_zero=False) == (
            "indistinguishable"
        )

    def test_an_unknown_metric_is_refused_rather_than_assumed_higher_is_better(self):
        """Defaulting to higher-is-better is how a new lower-is-better metric
        (HD95, error rate, latency) gets its verdict printed backwards."""
        with pytest.raises(ValueError, match="direccion"):
            pb.direction_for(metric="hd95_inventada", point_delta=0.1, excludes_zero=True)

    def test_the_registry_says_which_metrics_are_lower_is_better(self):
        assert pb.LOWER_IS_BETTER == frozenset({"ece", "mel_to_nv_rate",
                                              "luminance_gap_dark_light", "luminance_gap_range"})

    def test_a_higher_mel_to_nv_rate_favours_the_control(self):
        """More melanomas sent to nevus is worse: a positive delta is the control's."""
        assert pb.direction_for(metric="mel_to_nv_rate", point_delta=0.03, excludes_zero=True) == (
            "control_better"
        )

    @pytest.mark.parametrize("metric", ["mel_sens_at_thr_spec80", "mel_sens_at_thr_spec90",
                                        "mel_spec_at_thr_sens95"])
    def test_the_fixed_threshold_operating_points_are_higher_is_better(self, metric):
        assert pb.direction_for(metric=metric, point_delta=0.02, excludes_zero=True) == "arm_better"

    @pytest.mark.parametrize("metric", ["accuracy", "mel_recall", "mel_auc"])
    def test_the_v71_secondaries_are_higher_is_better(self, metric):
        """docs/29 2.3b contrasts melanoma recall and AUC on val; they are
        registered, and on the higher-is-better side."""
        assert pb.direction_for(metric=metric, point_delta=0.02, excludes_zero=True) == "arm_better"


class TestTheContrastRecord:
    def test_it_carries_every_field_the_published_schema_has(self):
        y_true = np.arange(40) % 3
        arm, control = _proba(40, 3, 13), _proba(40, 3, 14)

        record = pb.contrast_record(
            arm="ssl_proposed_vitb14", control="dinov2_public_vitb14",
            metric="macro_f1", probe="logreg", fraction=1.0, seed=0,
            y_true=y_true, arm_proba=arm, control_proba=control,
            n_resamples=20, rng_seed=20260923,
        )

        assert set(record) == {
            "arm", "control", "metric", "probe", "fraction", "seed",
            "point_arm", "point_control", "point_delta",
            "ci95_low", "ci95_high", "excludes_zero", "direction",
        }

    def test_the_ece_record_gets_the_inverted_direction(self):
        y_true = np.arange(60) % 3
        confident = np.zeros((60, 3))
        confident[np.arange(60), y_true] = 1.0  # perfectly calibrated -> ECE 0
        vague = np.full((60, 3), 1 / 3)

        record = pb.contrast_record(
            arm="a", control="b", metric="ece", probe="logreg", fraction=1.0, seed=0,
            y_true=y_true, arm_proba=confident, control_proba=vague,
            n_resamples=20, rng_seed=0,
        )

        assert record["point_delta"] < 0, "el escenario no produjo una ECE menor en el brazo"
        assert record["direction"] == "arm_better"

    def test_it_is_json_serialisable(self):
        import json

        y_true = np.arange(40) % 3
        record = pb.contrast_record(
            arm="a", control="b", metric="macro_f1", probe="logreg", fraction=1.0, seed=0,
            y_true=y_true, arm_proba=_proba(40, 3, 15), control_proba=_proba(40, 3, 16),
            n_resamples=20, rng_seed=0,
        )

        json.dumps(record)


class TestGuards:
    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="largo"):
            pb.paired_contrast(
                y_true=np.arange(10) % 2, arm_proba=_proba(10, 2, 1),
                control_proba=_proba(9, 2, 2), metric_fn=pb.metric_macro_f1,
                n_resamples=5, seed=0,
            )

    def test_zero_resamples_is_refused(self):
        """An empty resample list makes np.percentile return nan, and
        `nan > 0` is False -- the interval would silently include zero."""
        with pytest.raises(ValueError, match="remuestras"):
            pb.paired_contrast(
                y_true=np.arange(10) % 2, arm_proba=_proba(10, 2, 1),
                control_proba=_proba(10, 2, 2), metric_fn=pb.metric_macro_f1,
                n_resamples=0, seed=0,
            )

    def test_a_nan_in_the_resamples_is_refused_not_ignored(self):
        """np.percentile over NaN returns NaN, and every comparison against it
        is False, so the contrast would report 'indistinguishable' for a
        metric that actually failed to compute."""
        def sometimes_nan(y_true, proba):
            return float("nan") if len(np.unique(y_true)) < 3 else 0.5

        with pytest.raises(ValueError, match="no finito"):
            pb.paired_contrast(
                y_true=np.array([0, 1, 2] * 4), arm_proba=_proba(12, 3, 1),
                control_proba=_proba(12, 3, 2), metric_fn=sometimes_nan,
                n_resamples=200, seed=0,
            )


class TestEachArmMayHaveItsOwnMetric:
    """Operating points at each entry's OWN holdout threshold: the two arms are
    scored by different functions, still on the same draw."""

    def test_the_control_function_scores_the_control_and_the_draw_is_shared(self):
        y_true = np.arange(40) % 3
        arm, control = _proba(40, 3, 5), _proba(40, 3, 6)
        arm_fn = lambda y, p: float(p[:, 0].mean())  # noqa: E731
        control_fn = lambda y, p: float(p[:, 1].mean())  # noqa: E731

        out = pb.paired_contrast(y_true=y_true, arm_proba=arm, control_proba=control,
                                 metric_fn=arm_fn, control_metric_fn=control_fn,
                                 n_resamples=50, seed=7)

        assert out["point_arm"] == arm_fn(y_true, arm)
        assert out["point_control"] == control_fn(y_true, control)
        rng = np.random.RandomState(7)
        deltas = []
        for _ in range(50):
            i = rng.randint(0, 40, 40)
            deltas.append(arm_fn(y_true[i], arm[i]) - control_fn(y_true[i], control[i]))
        assert out["ci95_low"] == float(np.percentile(deltas, 2.5))
        assert out["ci95_high"] == float(np.percentile(deltas, 97.5))

    def test_without_a_control_function_both_arms_use_the_same_one(self):
        y_true = np.arange(40) % 3
        arm, control = _proba(40, 3, 5), _proba(40, 3, 6)
        kwargs = dict(y_true=y_true, arm_proba=arm, control_proba=control,
                      metric_fn=pb.metric_macro_f1, n_resamples=30, seed=1)
        assert pb.paired_contrast(**kwargs) == pb.paired_contrast(**kwargs,
                                                                  control_metric_fn=pb.metric_macro_f1)

    def test_the_record_passes_the_control_function_through(self):
        y_true = np.arange(40) % 3
        arm, control = _proba(40, 3, 5), _proba(40, 3, 6)
        record = pb.contrast_record(arm="a", control="b", metric="mel_sens_at_thr_spec80",
                                    probe="logreg", fraction=1.0, seed=0, y_true=y_true,
                                    arm_proba=arm, control_proba=control, n_resamples=10,
                                    metric_fn=lambda y, p: 1.0, control_metric_fn=lambda y, p: 0.0)
        assert record["point_delta"] == 1.0
