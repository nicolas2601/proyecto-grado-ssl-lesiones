"""MCC joins the E3 final and E3b paired contrasts; the existing metrics keep their exact values."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from dermssl import e3_stages as stages
from dermssl import metrics
from dermssl.bench import metrics as bm
from dermssl.e3b import contrasts as c3

GOLDEN_E3B_OLD_METRICS = "301170d5a95bc2b684d1d61dbe7a4ae81a5ec5b81743ccb20844f85823194b65"
OLD = ("macro_f1", "balanced_accuracy", "mel_recall")


def _data():
    rng = np.random.RandomState(9)
    n = 300
    y = np.arange(n) % 4
    groups = np.array([f"L{i // 2}" for i in range(n)])

    def proba(signal):
        a = rng.dirichlet(np.ones(4), n) + np.eye(4)[y] * signal
        return a / a.sum(1, keepdims=True)

    return y, groups, {"m": proba(0.3), "m+F4": proba(0.6)}


def _rows():
    y, groups, probas = _data()
    return c3.paired(y, probas, [("m+F4", "m", "seg_effect")], groups, n_classes=4, mel=1, n_resamples=30)


def test_e3b_keeps_the_existing_metrics_byte_for_byte():
    old = [r for r in _rows() if r["metric"] in OLD]
    assert [r["metric"] for r in old] == list(OLD)
    assert hashlib.sha256(json.dumps(old).encode()).hexdigest() == GOLDEN_E3B_OLD_METRICS


def test_e3b_adds_an_exact_mcc_row_last():
    y, _, probas = _data()
    rows = _rows()
    assert [r["metric"] for r in rows] == [*OLD, "mcc"]
    expected = bm.mcc_from_confusion(metrics.confusion_matrix(y, probas["m+F4"].argmax(1), 4).tolist())
    assert rows[-1]["point_arm"] == pytest.approx(expected) and rows[-1]["direction"]


def test_the_e3_final_metric_set_ends_with_mcc_and_keeps_the_others_in_order():
    y = np.arange(40) % 4
    assert list(stages._metric_fns(y, 4, 1)) == [*OLD, "mcc"]


def test_the_e3_final_subset_contrasts_carry_mcc():
    y, groups, probas = _data()
    rows = stages._subset_records("m+F4", "m", "all", np.ones(len(y), bool), probas, y, groups, 1, 4, 20)
    assert [r["metric"] for r in rows] == [*OLD, "mcc"] and all(r["subset"] == "all" for r in rows)
