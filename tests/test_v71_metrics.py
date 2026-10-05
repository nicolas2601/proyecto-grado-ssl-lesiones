"""docs/29 §2.3b and §2.7: melanoma operating points and per-config metrics.

The operating points are post-hoc thresholds on p(mel). The direction and the
tie handling are where a threshold silently lands on the wrong side, so they
are pinned on hand-made scores.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import paired_bootstrap as pb
from dermssl import selection_holdout as sh
from dermssl import v71_metrics as vm

MEL = 4


def _toy(n: int = 400, seed: int = 0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 7, n)
    logits = rng.normal(size=(n, 7))
    logits[np.arange(n), y] += 1.5
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y, proba


class TestThresholds:
    def test_specificity_threshold_keeps_at_least_the_target(self):
        y = np.array([0] * 10 + [1] * 5)
        s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.35, 0.75, 0.85, 0.95, 0.05])
        t = vm.threshold_for_specificity(y, s, 0.8)
        sens, spec = vm.sens_spec(y, s, t)
        assert spec == pytest.approx(0.8)
        # Lowest threshold with spec >= 0.8: 0.75, 0.85, 0.95 are the positives above it.
        assert sens == pytest.approx(3 / 5)

    def test_specificity_threshold_with_ties_never_undershoots(self):
        y = np.array([0, 0, 0, 0, 0, 1, 1])
        s = np.array([0.1, 0.5, 0.5, 0.5, 0.9, 0.6, 0.95])
        t = vm.threshold_for_specificity(y, s, 0.5)
        assert vm.sens_spec(y, s, t)[1] >= 0.5

    def test_sensitivity_threshold_is_the_highest_that_reaches_the_target(self):
        y = np.array([1] * 20 + [0] * 5)
        s = np.concatenate([np.linspace(0.05, 1.0, 20), np.full(5, 0.02)])
        t = vm.threshold_for_sensitivity(y, s, 0.95)
        sens, spec = vm.sens_spec(y, s, t)
        assert sens == pytest.approx(0.95)
        assert t == pytest.approx(np.linspace(0.05, 1.0, 20)[1])
        assert spec == 1.0

    def test_rounding_of_the_target_count_does_not_add_one(self):
        # 0.8 * 1400 is 1120.0000000000002 in floating point: ceil must not give 1121.
        assert vm.required_count(0.8, 1400) == 1120

    def test_one_class_only_is_refused(self):
        with pytest.raises(ValueError, match="positivos y negativos"):
            vm.threshold_for_specificity(np.zeros(5, int), np.arange(5.0), 0.8)


class TestMetricBundle:
    def test_bundle_reuses_the_published_metric_functions(self):
        y, proba = _toy()
        got = vm.metric_bundle(y, proba, mel=MEL)
        for name in ("macro_f1", "balanced_accuracy", "auc", "ece"):
            assert got[name] == pytest.approx(pb.METRIC_FUNCTIONS[name](y, proba))
        assert got["mel_recall"] == pytest.approx(
            np.mean(proba[y == MEL].argmax(axis=1) == MEL))

    def test_bundle_has_every_registered_metric(self):
        y, proba = _toy()
        assert set(vm.metric_bundle(y, proba, mel=MEL)) == set(vm.GRID_METRICS)

    def test_bootstrap_matches_the_selection_holdout_interval(self):
        y, proba = _toy(300)
        ci = vm.bootstrap_bundle(y, proba, mel=MEL, n_resamples=50)
        ref = sh.bootstrap_ci(y, proba, pb.metric_macro_f1, n_resamples=50)
        assert ci["macro_f1"] == pytest.approx(ref)

    def test_fixed_threshold_metrics_on_the_second_split(self):
        y, proba = _toy()
        thresholds = {"spec80": 0.3, "spec90": 0.5, "sens95": 0.1}
        got = vm.fixed_threshold_metrics(y, proba, mel=MEL, thresholds=thresholds)
        sens, spec = vm.sens_spec((y == MEL).astype(int), proba[:, MEL], 0.5)
        assert got["mel_sens_at_thr_spec90"] == pytest.approx(sens)
        assert got["mel_spec_at_thr_spec90"] == pytest.approx(spec)
        assert set(got) == {f"mel_{m}_at_thr_{k}" for m in ("sens", "spec") for k in thresholds}


class TestFastPaths:
    """The bundle runs ~2500 cells x 1000 resamples: it uses vectorised twins of
    the published metrics, which must agree with them, ties included."""

    def test_auc_matches_the_published_one_with_ties(self):
        from dermssl import metrics as _metrics

        y, proba = _toy(500, seed=3)
        proba = np.round(proba, 2)  # many ties, like a bootstrap resample's duplicates
        assert vm.auc_ovr_macro(y, proba) == pytest.approx(_metrics.auc_ovr_macro(y, proba), abs=1e-12)
        assert vm.binary_auc(y == MEL, proba[:, MEL]) == pytest.approx(
            _metrics.binary_auc((y == MEL).astype(int), proba[:, MEL]), abs=1e-12)

    def test_confusion_based_metrics_match(self):
        y, proba = _toy(500, seed=4)
        pred = proba.argmax(axis=1)
        pred[:40] = 3  # a class rarely predicted, to exercise the zero-denominator branch
        onehot = np.eye(7)[pred]
        assert vm.metric_bundle(y, onehot, mel=MEL)["macro_f1"] == pytest.approx(
            pb.metric_macro_f1(y, onehot), abs=1e-12)
        assert vm.metric_bundle(y, onehot, mel=MEL)["balanced_accuracy"] == pytest.approx(
            pb.metric_balanced_accuracy(y, onehot), abs=1e-12)


class TestValEntryProtocol:
    """Paula, 2026-09-28: the thresholds were fixed on the probe fitted on the 80 %
    fit split; a probe refitted on the full train has another score scale. On val
    the thresholded metrics therefore read the 80 % probe, the rest the full one."""

    def test_thresholded_metrics_read_the_fit80_probe_and_the_rest_the_full_one(self):
        y, full = _toy(400, seed=5)
        _, fit80 = _toy(400, seed=6)
        thresholds = vm.operating_thresholds(y, fit80, mel=MEL)
        got = vm.val_entry_metrics(y, proba_full=full, proba_threshold=fit80, mel=MEL,
                                   thresholds=thresholds, n_resamples=20)
        assert got["macro_f1"] == pytest.approx(pb.metric_macro_f1(y, full))
        assert got["mel_auc"] == pytest.approx(vm.metric_bundle(y, full, mel=MEL)["mel_auc"])
        expected = vm.fixed_threshold_metrics(y, fit80, mel=MEL, thresholds=thresholds)
        assert {k: got[k] for k in expected} == pytest.approx(expected)
        assert got["ci95"]["mel_sens_at_thr_spec80"] == pytest.approx(vm.bootstrap_bundle(
            y, fit80, mel=MEL, n_resamples=20,
            bundle=lambda yy, pp: vm.fixed_threshold_metrics(yy, pp, mel=MEL, thresholds=thresholds),
        )["mel_sens_at_thr_spec80"])

    def test_the_in_sample_operating_points_are_not_reported_on_val(self):
        """mel_sens_at_spec80 refits a threshold on the evaluated sample itself;
        on val that would be the very threshold transfer that was ruled out."""
        y, full = _toy(300, seed=7)
        got = vm.val_entry_metrics(y, proba_full=full, proba_threshold=full, mel=MEL,
                                   thresholds=vm.operating_thresholds(y, full, mel=MEL), n_resamples=5)
        assert not {"mel_sens_at_spec80", "mel_sens_at_spec90", "mel_spec_at_sens95"} & set(got)
