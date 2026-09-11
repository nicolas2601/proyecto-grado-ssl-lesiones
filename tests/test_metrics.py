"""Metrics are checked against values computed by hand, not against a library."""

import numpy as np
import pytest

from dermssl import metrics as m


def test_accuracy():
    assert m.accuracy([0, 1, 2, 2], [0, 1, 2, 0]) == pytest.approx(0.75)


def test_confusion_matrix_orientation():
    cm = m.confusion_matrix([0, 0, 1], [0, 1, 1], n_classes=2)
    # row = truth, column = prediction
    assert cm.tolist() == [[1, 1], [0, 1]]


def test_macro_f1_hand_computed():
    """y=[0,0,1,1], pred=[0,1,1,1] -> f1(class0)=2/3, f1(class1)=0.8."""
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    f1 = m.per_class_f1(y, p, 2)
    assert f1[0] == pytest.approx(2 / 3)
    assert f1[1] == pytest.approx(0.8)
    assert m.macro_f1(y, p, 2) == pytest.approx((2 / 3 + 0.8) / 2)


def test_class_never_predicted_scores_zero_not_nan():
    y = np.array([0, 0, 1])
    p = np.array([0, 0, 0])
    f1 = m.per_class_f1(y, p, 3)
    assert f1[1] == 0.0
    assert f1[2] == 0.0  # class absent from both -> 0, never NaN
    assert not np.isnan(m.macro_f1(y, p, 3))


def test_recall_per_class():
    r = m.recall_per_class([0, 0, 1, 1], [0, 1, 1, 1], n_classes=2)
    assert r.tolist() == pytest.approx([0.5, 1.0])


def test_auc_perfect_separation_is_one():
    assert m.binary_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(1.0)


def test_auc_inverted_separation_is_zero():
    assert m.binary_auc([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.0)


def test_auc_all_scores_tied_is_half():
    """Without average-rank tie handling this would come out wrongly optimistic."""
    assert m.binary_auc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auc_partial_hand_computed():
    """y=[0,1,0,1] with increasing scores -> positive ranks 2 and 4 -> 0.75."""
    assert m.binary_auc([0, 1, 0, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.75)


def test_auc_is_nan_for_single_class():
    assert np.isnan(m.binary_auc([1, 1, 1], [0.1, 0.5, 0.9]))


def test_auc_ovr_macro_skips_undefined_classes():
    y = np.array([0, 0, 1, 1])
    proba = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.2, 0.8, 0.0],
            [0.1, 0.9, 0.0],
        ]
    )
    # class 2 never appears -> undefined -> skipped, not counted as 0.5
    assert m.auc_ovr_macro(y, proba) == pytest.approx(1.0)


def test_ece_zero_when_perfectly_calibrated_and_certain():
    y = np.array([0, 1])
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert m.ece(y, proba) == pytest.approx(0.0)


def test_ece_is_large_when_confidently_wrong():
    y = np.array([1, 1, 1, 1])
    proba = np.tile([0.99, 0.01], (4, 1))  # 99 % confident, always wrong
    assert m.ece(y, proba) > 0.9


def test_summarize_returns_every_headline_metric():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1]])
    out = m.summarize(y, proba, n_classes=2)
    assert set(out) == {"macro_f1", "accuracy", "balanced_accuracy",
                        "macro_precision", "macro_recall", "auc", "ece"}
    assert out["accuracy"] == pytest.approx(1.0)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    proba = np.zeros((200, 2))
    # informative but imperfect scores
    proba[np.arange(200), y] = 0.7
    proba[np.arange(200), 1 - y] = 0.3
    noise = rng.rand(200) < 0.2
    proba[noise] = proba[noise][:, ::-1]

    point, lo, hi = m.bootstrap_ci(
        y, proba, lambda yy, pp: m.auc_ovr_macro(yy, pp), n_resamples=200, seed=1
    )
    assert lo <= point <= hi
    assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0


def test_mean_std_ignores_nan():
    out = m.mean_std([0.5, 0.7, float("nan")])
    assert out["n"] == 2
    assert out["mean"] == pytest.approx(0.6)
    assert out["std"] == pytest.approx(0.1)


def test_mean_std_all_nan_is_safe():
    out = m.mean_std([float("nan")])
    assert out["n"] == 0
    assert np.isnan(out["mean"])


def test_precision_per_class_hand_computed():
    """y=[0,0,1,1], pred=[0,1,1,1] -> prec(class0)=1.0, prec(class1)=2/3."""
    p = m.precision_per_class([0, 0, 1, 1], [0, 1, 1, 1], n_classes=2)
    assert p[0] == pytest.approx(1.0)
    assert p[1] == pytest.approx(2 / 3)


def test_precision_class_never_predicted_scores_zero_not_nan():
    p = m.precision_per_class([0, 0, 1], [0, 0, 0], n_classes=3)
    assert p[1] == 0.0
    assert p[2] == 0.0
    assert not np.isnan(p).any()


def test_macro_precision_and_recall():
    y = [0, 0, 1, 1]
    p = [0, 1, 1, 1]
    assert m.macro_precision(y, p, 2) == pytest.approx((1.0 + 2 / 3) / 2)
    assert m.macro_recall(y, p, 2) == pytest.approx((0.5 + 1.0) / 2)


def test_balanced_accuracy_is_mean_recall():
    """Predictor 'siempre clase 0' sobre clases desbalanceadas: acc alta,
    balanced accuracy baja - la trampa que la metrica corrige."""
    y = [0, 0, 0, 0, 0, 0, 1, 2]
    p = [0, 0, 0, 0, 0, 0, 0, 0]
    assert m.accuracy(y, p) == pytest.approx(0.75)
    assert m.balanced_accuracy(y, p, 3) == pytest.approx(1 / 3)


def test_summarize_includes_all_headline_metrics():
    y = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7]])
    out = m.summarize(y, proba, 2)
    for key in ("macro_f1", "accuracy", "balanced_accuracy", "auc",
                "macro_precision", "macro_recall", "ece"):
        assert key in out, f"missing {key}"
    assert out["balanced_accuracy"] == pytest.approx(
        m.balanced_accuracy(y, proba.argmax(axis=1), 2))


# --- stratified_summary: per-Fitzpatrick-bucket fairness evaluation --------
#
# Never fails on thin data (spec.md, fairness-evaluation req 2): buckets below
# ``min_n`` are SKIPPED, not silently dropped from the output structure - and
# an empty manifest (Fitzpatrick estimation not run yet) returns {}, matching
# design.md's error table ("Manifest missing -> stratified_summary returns {}").


def _multiclass_rows(n, n_classes, seed):
    """Deterministic, reasonably-separated fake predictions for n rows."""
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, n)
    proba = rng.dirichlet(np.ones(n_classes), size=n)
    return y, proba


def test_stratified_summary_returns_empty_dict_for_empty_input():
    """No fst_manifest.csv yet -> nothing to stratify. v5 behavior unchanged."""
    out = m.stratified_summary(np.array([]), np.zeros((0, 2)), np.array([]), n_classes=2)
    assert out == {}


def test_stratified_summary_always_reports_all_six_buckets_and_unavailable():
    """Every FST bucket appears in the output structure, even with zero rows."""
    y, proba = _multiclass_rows(40, 2, seed=0)
    buckets = np.array([1] * 40)
    out = m.stratified_summary(y, proba, buckets, n_classes=2, min_n=30)
    assert set(out) == {1, 2, 3, 4, 5, 6, "unavailable"}


@pytest.mark.parametrize(
    "count, expected_status",
    [
        (40, "OK"),       # populated: clears the default min_n=30
        (10, "SKIPPED"),  # sparse: below min_n, but not empty
        (0, "SKIPPED"),   # empty bucket: still present, never omitted
    ],
    ids=["populated", "sparse_skipped", "empty_skipped"],
)
def test_stratified_summary_bucket_status_by_population(count, expected_status):
    """Table-driven: bucket 1 gets `count` rows, the rest fall to 'unavailable'."""
    n_classes, total = 2, 100
    y, proba = _multiclass_rows(total, n_classes, seed=1)
    buckets = np.array([1] * count + [None] * (total - count), dtype=object)

    out = m.stratified_summary(y, proba, buckets, n_classes=n_classes)  # default min_n=30

    assert out[1]["status"] == expected_status
    assert out[1]["count"] == count
    if expected_status == "SKIPPED":
        assert "reason" in out[1]
        assert "macro_f1" not in out[1]
    else:
        assert out[1]["macro_f1"] == pytest.approx(
            m.summarize(y[:count], proba[:count], n_classes)["macro_f1"]
        )


def test_stratified_summary_unavailable_key_groups_none_buckets():
    """FSTEstimate.bucket=None (spec.md req 4) lands under the 'unavailable' key."""
    n_classes = 2
    y, proba = _multiclass_rows(50, n_classes, seed=3)
    buckets = np.array([None] * 35 + [1] * 15, dtype=object)

    out = m.stratified_summary(y, proba, buckets, n_classes=n_classes, min_n=30)

    assert out["unavailable"]["status"] == "OK"
    assert out["unavailable"]["count"] == 35
    assert out["unavailable"]["macro_f1"] == pytest.approx(
        m.summarize(y[:35], proba[:35], n_classes)["macro_f1"]
    )
    assert out[1]["status"] == "SKIPPED"
    assert out[1]["count"] == 15


def test_stratified_summary_min_n_is_configurable():
    n_classes = 2
    y, proba = _multiclass_rows(10, n_classes, seed=4)
    buckets = np.array([1] * 10)

    out = m.stratified_summary(y, proba, buckets, n_classes=n_classes, min_n=5)

    assert out[1]["status"] == "OK"
    assert out[1]["count"] == 10


def test_stratified_summary_ok_bucket_has_the_same_metric_keys_as_summarize():
    """'same metrics as non-stratified summary' (spec.md req 1)."""
    n_classes = 3
    y, proba = _multiclass_rows(40, n_classes, seed=5)
    buckets = np.array([2] * 40)

    out = m.stratified_summary(y, proba, buckets, n_classes=n_classes)

    expected_keys = set(m.summarize(y, proba, n_classes)) | {"status", "count"}
    assert set(out[2]) == expected_keys


# --- comparison_report: Stage 2 baseline vs. mitigation reporting ---------
#
# spec.md, circle-adaptation req 5: "MUST report Stage 1 stratified fairness
# metrics for three variants - baseline, regularization-only, and
# synthetic-blend - so the relative fairness improvement (primary thesis
# acceptance criterion) is directly computable between variants."
#
# Tested against synthetic, hand-built stratified_summary()-shaped fixtures -
# NOT against real data: no fst_manifest.csv exists yet, no real Stage 2
# Kaggle run has happened (see docs/21-checkpoint-manual-circle-stage2.md).
# This only checks that the report structure and arithmetic are correct given
# whatever three stratified_summary() outputs it is handed.


def qualifying_variant_bucket(macro_f1, count=40):
    return {"status": "OK", "count": count, "macro_f1": macro_f1}


def skipped_variant_bucket(count=5):
    return {"status": "SKIPPED", "count": count, "reason": f"n={count} < min_n=30"}


def test_comparison_report_lists_all_three_variants_with_a_gap():
    baseline = {1: qualifying_variant_bucket(0.70), 2: qualifying_variant_bucket(0.50)}
    regularization = {1: qualifying_variant_bucket(0.65), 2: qualifying_variant_bucket(0.55)}
    synthetic_blend = {1: qualifying_variant_bucket(0.62), 2: qualifying_variant_bucket(0.58)}

    report = m.comparison_report(baseline, regularization, synthetic_blend)

    assert set(report) == {"baseline", "regularization", "synthetic_blend"}
    for variant in report.values():
        assert "gap" in variant


def test_comparison_report_gap_is_max_minus_min_among_qualifying_buckets():
    baseline = {1: qualifying_variant_bucket(0.70), 2: qualifying_variant_bucket(0.50)}
    regularization = {1: qualifying_variant_bucket(0.65), 2: qualifying_variant_bucket(0.55)}
    synthetic_blend = {1: qualifying_variant_bucket(0.62), 2: qualifying_variant_bucket(0.58)}

    report = m.comparison_report(baseline, regularization, synthetic_blend)

    assert report["baseline"]["gap"] == pytest.approx(0.20)
    assert report["regularization"]["gap"] == pytest.approx(0.10)
    assert report["synthetic_blend"]["gap"] == pytest.approx(0.04)


def test_comparison_report_relative_improvement_vs_baseline():
    baseline = {1: qualifying_variant_bucket(0.70), 2: qualifying_variant_bucket(0.50)}  # gap 0.20
    regularization = {1: qualifying_variant_bucket(0.65), 2: qualifying_variant_bucket(0.55)}  # 0.10
    synthetic_blend = {1: qualifying_variant_bucket(0.61), 2: qualifying_variant_bucket(0.56)}  # 0.05

    report = m.comparison_report(baseline, regularization, synthetic_blend)

    assert report["regularization"]["relative_improvement"] == pytest.approx(0.5)
    assert report["synthetic_blend"]["relative_improvement"] == pytest.approx(0.75)
    assert "relative_improvement" not in report["baseline"]


def test_comparison_report_baseline_insufficient_data_disables_relative_improvement():
    """<2 qualifying buckets in the baseline: nothing to compare a relative
    improvement against, so it must not be silently reported as some number."""
    baseline = {1: qualifying_variant_bucket(0.70), 2: skipped_variant_bucket()}
    regularization = {1: qualifying_variant_bucket(0.65), 2: qualifying_variant_bucket(0.60)}
    synthetic_blend = {1: qualifying_variant_bucket(0.62), 2: qualifying_variant_bucket(0.61)}

    report = m.comparison_report(baseline, regularization, synthetic_blend)

    assert report["baseline"]["gap"] is None
    assert report["baseline"]["status"] == "insufficient_data"
    assert report["regularization"]["relative_improvement"] is None
    assert report["synthetic_blend"]["relative_improvement"] is None
    # the mitigation variants' own gaps are still reported even when the
    # baseline cannot anchor a relative improvement
    assert report["regularization"]["gap"] == pytest.approx(0.05)


def test_comparison_report_mitigation_variant_insufficient_data():
    """A mitigation variant with thin buckets reports gap=None and no
    relative_improvement, without breaking the other variants."""
    baseline = {1: qualifying_variant_bucket(0.70), 2: qualifying_variant_bucket(0.50)}
    regularization = {1: qualifying_variant_bucket(0.65), 2: skipped_variant_bucket()}
    synthetic_blend = {1: qualifying_variant_bucket(0.62), 2: qualifying_variant_bucket(0.58)}

    report = m.comparison_report(baseline, regularization, synthetic_blend)

    assert report["regularization"]["gap"] is None
    assert report["regularization"]["relative_improvement"] is None
    assert report["synthetic_blend"]["gap"] == pytest.approx(0.04)
    assert report["synthetic_blend"]["relative_improvement"] is not None


def test_comparison_report_zero_baseline_gap_avoids_division_by_zero():
    baseline = {1: qualifying_variant_bucket(0.70), 2: qualifying_variant_bucket(0.70)}  # gap 0.0
    regularization = {1: qualifying_variant_bucket(0.70), 2: qualifying_variant_bucket(0.70)}
    synthetic_blend = {1: qualifying_variant_bucket(0.65), 2: qualifying_variant_bucket(0.60)}

    report = m.comparison_report(baseline, regularization, synthetic_blend)

    assert report["baseline"]["gap"] == pytest.approx(0.0)
    assert report["regularization"]["relative_improvement"] is None
    assert report["synthetic_blend"]["relative_improvement"] is None


def test_comparison_report_respects_a_custom_metric_and_min_bucket_n():
    baseline = {
        1: {"status": "OK", "count": 12, "auc": 0.80},
        2: {"status": "OK", "count": 12, "auc": 0.60},
    }
    regularization = {
        1: {"status": "OK", "count": 12, "auc": 0.75},
        2: {"status": "OK", "count": 12, "auc": 0.68},
    }
    synthetic_blend = {
        1: {"status": "OK", "count": 12, "auc": 0.73},
        2: {"status": "OK", "count": 12, "auc": 0.70},
    }

    # default min_bucket_n=30 would make every bucket too thin to qualify
    default_report = m.comparison_report(
        baseline, regularization, synthetic_blend, metric="auc"
    )
    assert default_report["baseline"]["gap"] is None

    report = m.comparison_report(
        baseline, regularization, synthetic_blend, metric="auc", min_bucket_n=10
    )
    assert report["baseline"]["gap"] == pytest.approx(0.20)
    assert report["regularization"]["gap"] == pytest.approx(0.07)
