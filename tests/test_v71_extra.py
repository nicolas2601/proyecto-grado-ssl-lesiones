"""Extra v7.1 val contrasts: fair mel operating points, mel->nv confusion, s43.

Pinned on hand-made probabilities; the reproduction of the committed val
numbers lives in ``test_v71_extra_contrasts_script.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import paired_bootstrap as pb
from dermssl import v71_extra as vx
from dermssl import v71_metrics as vm

MEL, NV = 4, 5


def _onehot(pred: list[int], n_classes: int = 7) -> np.ndarray:
    out = np.full((len(pred), n_classes), 0.01)
    out[np.arange(len(pred)), pred] = 0.9
    return out / out.sum(axis=1, keepdims=True)


def _toy(n: int = 300, seed: int = 0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 7, n)
    logits = rng.normal(size=(n, 7))
    logits[np.arange(n), y] += 1.5
    return y, np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)


class TestMelToNv:
    def test_counts_melanomas_predicted_as_nevus_over_true_melanomas(self):
        y = np.array([MEL, MEL, MEL, MEL, NV, 0])
        proba = _onehot([NV, NV, MEL, 0, NV, NV])
        out = vx.mel_to_nv(y, proba, mel=MEL, nv=NV)
        assert out == {"n_mel": 4, "count": 2, "rate": 0.5}

    def test_the_rate_metric_matches_the_summary(self):
        y, proba = _toy()
        assert vx.mel_to_nv_rate_fn(MEL, NV)(y, proba) == vx.mel_to_nv(y, proba, mel=MEL, nv=NV)["rate"]

    def test_no_melanoma_is_nan_not_zero(self):
        # A rate over zero melanomas is undefined; 0 would read as "never confused".
        assert np.isnan(vx.mel_to_nv_rate_fn(MEL, NV)(np.array([0, 1]), _onehot([NV, NV])))


class TestFairOperatingPoints:
    def test_each_entry_is_read_at_its_own_threshold(self):
        y, arm = _toy(seed=1)
        _, control = _toy(seed=2)
        thr = {"arm": vm.operating_thresholds(y, arm, mel=MEL),
               "ctl": vm.operating_thresholds(y, control, mel=MEL)}
        thr["ctl"] = {k: v + 0.05 for k, v in thr["ctl"].items()}  # a different threshold per entry
        rows = vx.fair_operating_contrasts(y, {"arm": arm, "ctl": control}, thr, [("arm", "ctl")],
                                           mel=MEL, n_resamples=20)
        assert [r["metric"] for r in rows] == list(vx.FAIR_OPERATING_METRICS)
        for r in rows:
            assert r["point_arm"] == vm.fixed_threshold_metrics(y, arm, mel=MEL, thresholds=thr["arm"])[r["metric"]]
            assert r["point_control"] == vm.fixed_threshold_metrics(y, control, mel=MEL,
                                                                    thresholds=thr["ctl"])[r["metric"]]
            assert r["direction"] in {"arm_better", "control_better", "indistinguishable"}


class TestArgmaxContrasts:
    def test_macro_f1_is_the_published_contrast(self):
        y, arm = _toy(seed=3)
        _, control = _toy(seed=4)
        rows = vx.argmax_contrasts(y, {"a": arm, "c": control}, [("a", "c")], ("macro_f1",), n_resamples=25)
        expected = pb.contrast_record(arm="a", control="c", metric="macro_f1", probe="logreg", fraction=1.0,
                                      seed=0, y_true=y, arm_proba=arm, control_proba=control, n_resamples=25)
        assert rows == [expected]

    def test_mel_to_nv_rate_is_lower_is_better(self):
        y = np.array([MEL] * 10 + [0] * 10)
        good = _onehot([MEL] * 10 + [0] * 10)
        bad = _onehot([NV] * 10 + [0] * 10)
        (row,) = vx.argmax_contrasts(y, {"good": good, "bad": bad}, [("good", "bad")], ("mel_to_nv_rate",),
                                     n_resamples=20, mel=MEL, nv=NV)
        assert row["point_delta"] == -1.0
        assert row["direction"] == "arm_better"

    def test_an_unknown_metric_is_refused(self):
        y, p = _toy()
        with pytest.raises(ValueError, match="metrica"):
            vx.argmax_contrasts(y, {"a": p, "c": p}, [("a", "c")], ("hd95",), n_resamples=5)
