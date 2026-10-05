"""Optional seams of the v7.1 val stage used by the clean-weights sensitivity run; defaults stay as they were."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import v71_val  # noqa: E402

from dermssl import paired_bootstrap as pb  # noqa: E402


def _data(n=120, k=3, seed=0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, k, n)
    a, b = rng.dirichlet(np.ones(k), n), rng.dirichlet(np.ones(k), n)
    return y, {"arm": a, "ctl": b}


def test_mcc_is_a_contrast_metric_and_matches_the_paired_bootstrap():
    y, probas = _data()
    rows = v71_val._contrast("arm", "ctl", y, probas, 0, 20, metrics=("macro_f1", "mcc"))
    want = pb.contrast_record(arm="arm", control="ctl", metric="mcc", probe="logreg", fraction=1.0, seed=0, y_true=y,
                              arm_proba=probas["arm"], control_proba=probas["ctl"], n_resamples=20,
                              metric_fn=pb.metric_mcc)
    assert rows[1] == want


def test_appending_mcc_last_leaves_the_other_draws_unchanged():
    y, probas = _data()
    base = v71_val._contrast("arm", "ctl", y, probas, 0, 20)
    more = v71_val._contrast("arm", "ctl", y, probas, 0, 20, metrics=(*v71_val.CONTRAST_METRICS, "mcc"))
    assert more[:-1] == base and more[-1]["metric"] == "mcc"


def test_groups_resample_whole_groups_and_default_stays_by_row():
    y, probas = _data()
    groups = np.repeat(np.arange(40), 3)
    grouped = v71_val._contrast("arm", "ctl", y, probas, 0, 30, metrics=("macro_f1",), groups=groups)[0]
    by_row = v71_val._contrast("arm", "ctl", y, probas, 0, 30, metrics=("macro_f1",))[0]
    want = pb.contrast_record(arm="arm", control="ctl", metric="macro_f1", probe="logreg", fraction=1.0, seed=0, y_true=y,
                              arm_proba=probas["arm"], control_proba=probas["ctl"], n_resamples=30, groups=groups)
    assert grouped == want and grouped != by_row


def test_score_entries_takes_an_injected_bank(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    class Bank:
        def model(self, state):
            return state

        def state(self, name):
            seen["name"] = name
            raise Stop

    args = type("A", (), {"out_dir": Path("/nonexistent"), "decode_workers": 1, "weights_dir": Path("/x"),
                          "split": "holdout", "batch_size": 8, "probe_jobs": 1})()
    ctx = {"ids": {"fit": ["a"]}, "id2label": {"a": 0}, "image_index": {}, "classes": ["c"]}
    monkeypatch.setattr(v71_val.io, "ImageStore", lambda *a, **k: object())
    with pytest.raises(Stop):
        v71_val._score_entries(args, ctx, {"X": "B_s42_a1.0@224|cls4|balanced|tta"}, ["a"], ["a"], "cpu", bank=Bank())
    assert seen["name"] == "B_s42_a1.0"
