"""MCC joins the paired contrasts of the ISIC bank without moving any existing number."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from test_isic_eval_runner import FakeExtractor, make_ds, run

from dermssl import e3_store, metrics
from dermssl import paired_bootstrap as pb
from dermssl.bench import metrics as bm
from dermssl.isic_eval import runner as rn
from dermssl.isic_eval import scoring

#: sha256 of json.dumps of the rows of the five metrics the contrasts had before MCC (seeded data, 40 resamples).
GOLDEN = {"grouped": "5d017c19bab94d355ab5c515fe990f6c07278ec756b7cb724f3742db2dfdff16",
          "ungrouped": "b9e1603b6356560c0c10c732c756d40358cd9a73d401ee8fb3082387d11926d1"}
OLD = ("macro_f1", "balanced_accuracy", "accuracy", "auc", "ece")


def _data():
    rng = np.random.RandomState(5)
    y = np.arange(240) % 3
    groups = np.array([f"p{i // 3}" for i in range(240)])
    a = rng.dirichlet(np.ones(3), 240) + np.eye(3)[y] * 0.5
    return y, groups, a / a.sum(1, keepdims=True), rng.dirichlet(np.ones(3), 240)


@pytest.mark.parametrize("name", ["grouped", "ungrouped"])
def test_the_existing_metrics_keep_their_exact_values(name):
    y, groups, a, c = _data()
    rows = scoring.contrasts("A", "B", y, a, c, groups=groups if name == "grouped" else None, n_resamples=40)
    old = [r for r in rows if r["metric"] in OLD]
    assert [r["metric"] for r in old] == list(OLD)
    assert hashlib.sha256(json.dumps(old).encode()).hexdigest() == GOLDEN[name]


def test_mcc_is_added_last_with_the_exact_confusion_matrix_value():
    y, groups, a, c = _data()
    rows = scoring.contrasts("A", "B", y, a, c, groups=groups, n_resamples=40)
    assert [r["metric"] for r in rows] == [*OLD, "mcc"]
    mcc = rows[-1]
    expected = bm.mcc_from_confusion(metrics.confusion_matrix(y, a.argmax(1), 3).tolist())
    assert mcc["point_arm"] == pytest.approx(expected) and mcc["direction"] in {
        "arm_better", "control_better", "indistinguishable"}
    assert mcc["point_delta"] == pytest.approx(mcc["point_arm"] - mcc["point_control"])


def test_the_mcc_bootstrap_uses_the_same_resamples_as_the_other_metrics():
    y, groups, a, c = _data()
    draw, rng = pb.index_sampler(len(y), groups), np.random.RandomState(20260923)
    deltas = []
    for _ in range(40):
        i = draw(rng)
        deltas.append(pb.metric_mcc(y[i], a[i]) - pb.metric_mcc(y[i], c[i]))
    row = scoring.contrasts("A", "B", y, a, c, groups=groups, n_resamples=40)[-1]
    assert row["ci95_low"] == pytest.approx(np.percentile(deltas, 2.5))
    assert row["ci95_high"] == pytest.approx(np.percentile(deltas, 97.5))


def test_mcc_is_known_as_higher_is_better():
    assert pb.direction_for(metric="mcc", point_delta=0.1, excludes_zero=True) == "arm_better"
    assert pb.direction_for(metric="mcc", point_delta=-0.1, excludes_zero=True) == "control_better"


def test_a_contrast_computed_before_mcc_is_recomputed_not_skipped(tmp_path):
    ds = make_ds(grouped=False)
    store, cstore = e3_store.CellStore(tmp_path), e3_store.CellStore(tmp_path / "contrasts")
    run(store, ds, "B71", FakeExtractor(ds), pool=True)
    run(store, ds, "PUBLIC_B71", FakeExtractor(ds))
    assert rn.run_contrast(store, cstore, "B71__clean", "PUBLIC_B71", n_resamples=20) == "done"
    name = "B71__clean__vs__PUBLIC_B71"
    old = cstore.read(name)
    old["key"].pop("metrics")  # the key as it was written before MCC
    old["key_sha256"] = e3_store.canonical_sha256(old["key"])
    e3_store.write_json_atomic(cstore.path(name), old)
    assert rn.run_contrast(store, cstore, "B71__clean", "PUBLIC_B71", n_resamples=20) == "done"
    assert "mcc" in {r["metric"] for r in cstore.read(name)["result"]["paired"]}
    assert rn.run_contrast(store, cstore, "B71__clean", "PUBLIC_B71", n_resamples=20) == "skipped"
