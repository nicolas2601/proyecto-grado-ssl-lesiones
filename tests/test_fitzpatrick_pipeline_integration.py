"""CPU integration for the manifest -> strata -> gate/report seam."""

import numpy as np

from dermssl import gates, metrics


def test_unavailable_rows_survive_stage1_and_gate_uses_only_measured_buckets():
    y = np.array([0, 1, 0, 1, 0])
    proba = np.array(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.8, 0.2], [0.7, 0.3]]
    )
    buckets = np.array([1, 1, 2, 2, "unavailable"], dtype=object)

    summary = metrics.stratified_summary(y, proba, buckets, n_classes=2, min_n=2)
    gate = gates.fairness_gate(summary, max_gap=0.15, min_bucket_n=2)

    assert summary["unavailable"]["count"] == 1
    assert gate.status == gates.FAIL
    assert gate.value > 0.15


def test_three_variant_report_keeps_uncertainty_explicit_without_raw_predictions():
    baseline = {
        1: {"status": "OK", "count": 30, "macro_f1": 0.8},
        2: {"status": "OK", "count": 30, "macro_f1": 0.5},
    }
    mitigation = {
        1: {"status": "OK", "count": 30, "macro_f1": 0.7},
        2: {"status": "OK", "count": 30, "macro_f1": 0.6},
    }

    report = metrics.comparison_report(baseline, mitigation, mitigation)

    assert report["regularization"]["relative_improvement"] > 0
    assert report["regularization"]["relative_improvement_ci95"] is None
