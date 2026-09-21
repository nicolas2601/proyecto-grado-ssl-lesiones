"""Evaluation metrics implemented with numpy only.

Why not scikit-learn
--------------------
scikit-learn is available on Kaggle but not on every machine, and these five
metrics are short enough to own. Owning them buys two things: the notebook can
be unit tested locally without extra dependencies, and the definition of every
reported number is visible in the repository instead of hidden behind a version
of a library.

Definitions used
----------------
``macro_f1``
    F1 computed per class, then averaged unweighted. Chosen over accuracy
    because HAM10000 is imbalanced ~58:1 - accuracy is dominated by ``nv``.
``auc_ovr_macro``
    One-vs-rest ROC AUC per class, averaged. Rank based (Mann-Whitney U) with
    proper handling of tied scores.
``ece``
    Expected Calibration Error: |confidence - accuracy| averaged over
    equal-width confidence bins, weighted by bin population. Answers "when the
    model says 90 %, is it right 90 % of the time?".
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are the true class, columns the predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """F1 per class; classes with no true and no predicted samples score 0."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
    return f1


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(per_class_f1(y_true, y_pred, n_classes).mean())


def recall_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    support = cm.sum(axis=1).astype(float)
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(support > 0, tp / support, 0.0)


def precision_per_class(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Precision per class; classes never predicted score 0, not NaN."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    predicted = cm.sum(axis=0).astype(float)
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(predicted > 0, tp / predicted, 0.0)


def macro_precision(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(precision_per_class(y_true, y_pred, n_classes).mean())


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(recall_per_class(y_true, y_pred, n_classes).mean())


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall: immune to class imbalance, unlike plain accuracy.

    Reference anchor: the official ISIC 2019 winner scored 63.6% on this
    metric; plain accuracy on HAM10000 can hit 67% by always predicting nv.
    """
    return macro_recall(y_true, y_pred, n_classes)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with tied values sharing their average rank.

    Tie handling is not cosmetic: a model that outputs the same probability for
    many samples would get a wrongly optimistic AUC with naive ranking.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def binary_auc(y_binary: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC for one class via the Mann-Whitney U statistic.

    Returns NaN when the class has only positives or only negatives, because
    AUC is undefined there (and silently returning 0.5 would hide the problem).
    """
    y_binary = np.asarray(y_binary).astype(int)
    n_pos = int(y_binary.sum())
    n_neg = int(len(y_binary) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(scores, dtype=float))
    rank_sum = ranks[y_binary == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_ovr_macro(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Macro-averaged one-vs-rest AUC. Undefined classes are skipped."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    aucs = [binary_auc((y_true == c).astype(int), proba[:, c]) for c in range(proba.shape[1])]
    valid = [a for a in aucs if a == a]  # drop NaN
    return float(np.mean(valid)) if valid else float("nan")


def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error over equal-width confidence bins."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        total += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(total)


def summarize(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> dict:
    """One call, every headline metric. Used by every probe."""
    pred = np.asarray(proba).argmax(axis=1)
    return {
        "macro_f1": macro_f1(y_true, pred, n_classes),
        "accuracy": accuracy(y_true, pred),
        "balanced_accuracy": balanced_accuracy(y_true, pred, n_classes),
        "macro_precision": macro_precision(y_true, pred, n_classes),
        "macro_recall": macro_recall(y_true, pred, n_classes),
        "auc": auc_ovr_macro(y_true, proba),
        "ece": ece(y_true, proba),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI: ``(point_estimate, low, high)``.

    Resamples the validation set with replacement. Reporting an interval instead
    of a bare number is the difference between "0.69" and "0.69, and with this
    sample size it could plausibly be 0.66 to 0.73".
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    point = metric_fn(y_true, proba)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        v = metric_fn(y_true[idx], proba[idx])
        if v == v:
            vals.append(v)
    if not vals:
        return float(point), float("nan"), float("nan")
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


FST_BUCKETS = (1, 2, 3, 4, 5, 6)
UNAVAILABLE_BUCKET = "unavailable"


def _is_unavailable_bucket(value) -> bool:
    """Accept the explicit CSV marker plus legacy empty/NaN representations."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", UNAVAILABLE_BUCKET, "nan"}
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def stratified_summary(
    y_true: np.ndarray,
    proba: np.ndarray,
    buckets: np.ndarray,
    n_classes: int,
    min_n: int = 30,
) -> dict:
    """``summarize()``, split per Fitzpatrick skin-type bucket (1-6, or
    ``"unavailable"`` for ``FSTEstimate.bucket is None`` rows - see
    ``dermssl.fitzpatrick``).

    Never fails on thin data (spec.md, fairness-evaluation req 2): every
    bucket in ``FST_BUCKETS + (UNAVAILABLE_BUCKET,)`` is always present in the
    output, even with zero matching rows - a bucket below ``min_n`` (0
    included) is marked ``"SKIPPED"`` with the observed count and a reason,
    never silently omitted from the structure.

    With no rows at all (``y_true`` empty - e.g. no ``fst_manifest.csv`` has
    been produced yet), returns ``{}``: nothing to stratify, matching
    design.md's error table ("Manifest missing -> stratified_summary returns
    {}; gate = SKIP").
    """
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        return {}

    proba = np.asarray(proba, dtype=float)
    buckets = np.asarray(buckets, dtype=object)

    result: dict = {}
    for key in (*FST_BUCKETS, UNAVAILABLE_BUCKET):
        if key == UNAVAILABLE_BUCKET:
            mask = np.array([_is_unavailable_bucket(b) for b in buckets])
        else:
            mask = np.array([b == key for b in buckets])
        count = int(mask.sum())
        if count < min_n:
            result[key] = {
                "status": "SKIPPED",
                "count": count,
                "reason": f"n={count} < min_n={min_n}",
            }
        else:
            result[key] = {
                "status": "OK",
                "count": count,
                **summarize(y_true[mask], proba[mask], n_classes),
            }
    return result


DEFAULT_COMPARISON_METRIC = "macro_f1"
DEFAULT_COMPARISON_MIN_BUCKET_N = 30
COMPARISON_VARIANTS = ("baseline", "regularization", "synthetic_blend")


def _variant_gap(summary: dict, *, metric: str, min_bucket_n: int) -> float | None:
    """Max-minus-min ``metric`` among buckets with ``status == "OK"`` and at
    least ``min_bucket_n`` samples - the same qualifying rule
    ``gates.fairness_gate`` applies, reimplemented here (not imported) so
    ``metrics.py`` stays independent of ``gates.py`` (mirrors D1: neither
    module depends on the other; the caller composes them).

    Returns ``None`` when fewer than two buckets qualify - "insufficient
    data", not zero: a report of ``0.0`` would misleadingly claim a perfect
    (measured) fairness gap where there is, in fact, nothing to measure.
    """
    qualifying = {
        bucket: entry[metric]
        for bucket, entry in summary.items()
        if entry.get("status") == "OK" and entry.get("count", 0) >= min_bucket_n
    }
    if len(qualifying) < 2:
        return None
    return max(qualifying.values()) - min(qualifying.values())


def _bucket_signature(values) -> np.ndarray:
    """Canonical representation used to verify paired validation cohorts."""
    return np.asarray(
        [UNAVAILABLE_BUCKET if _is_unavailable_bucket(v) else int(v) for v in values],
        dtype=object,
    )


def _paired_bootstrap_intervals(
    bootstrap_inputs: dict,
    *,
    metric: str,
    min_bucket_n: int,
    n_resamples: int,
    seed: int,
) -> dict:
    """Paired 95% bootstrap CIs for gaps and relative improvements.

    All variants must describe the same validation examples in the same order.
    One resampled index vector is shared across variants, preserving pairing
    instead of injecting extra variance through independent resampling.
    """
    missing = [name for name in COMPARISON_VARIANTS if name not in bootstrap_inputs]
    if missing:
        raise ValueError(f"bootstrap_inputs missing variants: {missing}")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    prepared = {}
    for name in COMPARISON_VARIANTS:
        raw = bootstrap_inputs[name]
        prepared[name] = {
            "y_true": np.asarray(raw["y_true"]),
            "proba": np.asarray(raw["proba"], dtype=float),
            "buckets": np.asarray(raw["buckets"], dtype=object),
            "n_classes": int(raw["n_classes"]),
        }

    reference = prepared["baseline"]
    n = len(reference["y_true"])
    reference_buckets = _bucket_signature(reference["buckets"])
    for name in COMPARISON_VARIANTS[1:]:
        candidate = prepared[name]
        same = (
            len(candidate["y_true"]) == n
            and np.array_equal(candidate["y_true"], reference["y_true"])
            and np.array_equal(_bucket_signature(candidate["buckets"]), reference_buckets)
            and candidate["n_classes"] == reference["n_classes"]
        )
        if not same:
            raise ValueError("bootstrap_inputs must use the same validation cohort")
    if n == 0:
        return {
            name: {"gap_ci95": None, "relative_improvement_ci95": None}
            for name in COMPARISON_VARIANTS
        }

    gap_samples = {name: [] for name in COMPARISON_VARIANTS}
    relative_samples = {name: [] for name in COMPARISON_VARIANTS[1:]}
    rng = np.random.RandomState(seed)
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        gaps = {}
        for name, raw in prepared.items():
            summary = stratified_summary(
                raw["y_true"][idx],
                raw["proba"][idx],
                raw["buckets"][idx],
                raw["n_classes"],
                min_n=min_bucket_n,
            )
            gap = _variant_gap(summary, metric=metric, min_bucket_n=min_bucket_n)
            gaps[name] = gap
            if gap is not None:
                gap_samples[name].append(gap)
        baseline_gap = gaps["baseline"]
        if baseline_gap is None or baseline_gap == 0:
            continue
        for name in COMPARISON_VARIANTS[1:]:
            if gaps[name] is not None:
                relative_samples[name].append((baseline_gap - gaps[name]) / baseline_gap)

    def ci(values) -> list[float] | None:
        if not values:
            return None
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    intervals = {}
    for name in COMPARISON_VARIANTS:
        intervals[name] = {
            "gap_ci95": ci(gap_samples[name]),
            "relative_improvement_ci95": (
                None if name == "baseline" else ci(relative_samples[name])
            ),
        }
    return intervals


def comparison_report(
    baseline: dict,
    regularization: dict,
    synthetic_blend: dict,
    *,
    metric: str = DEFAULT_COMPARISON_METRIC,
    min_bucket_n: int = DEFAULT_COMPARISON_MIN_BUCKET_N,
    bootstrap_inputs: dict | None = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Stage 2 baseline-vs-mitigation comparison (spec.md, circle-adaptation
    req 5): consumes three ``stratified_summary()`` outputs (baseline,
    regularization-only, synthetic-blend) and reports the per-FST-bucket gap
    for each, plus each mitigation variant's relative improvement over the
    baseline gap - the thesis's primary acceptance criterion
    (proposal.md, "Success Criteria": "measurable relative fairness
    improvement ... of Stage 2 over the Stage 1 baseline").

    This function computes arithmetic over whatever three summaries it is
    given - it does not run any training or produce any data itself. As of
    this writing no real Stage 1 or Stage 2 run has happened (see
    docs/21-checkpoint-manual-circle-stage2.md), so calling this with real
    data is future work; it is tested here against synthetic,
    hand-built ``stratified_summary()``-shaped fixtures only.

    ``relative_improvement`` is ``(baseline_gap - variant_gap) / baseline_gap``:
    positive means the variant's gap is smaller (fairer) than the baseline's.
    It is ``None`` - never a number - whenever it cannot be computed without
    guessing: the baseline gap is undefined (fewer than two qualifying
    buckets), the baseline gap is exactly zero (division by zero), or the
    variant's own gap is undefined.
    """
    intervals = (
        _paired_bootstrap_intervals(
            bootstrap_inputs,
            metric=metric,
            min_bucket_n=min_bucket_n,
            n_resamples=n_resamples,
            seed=seed,
        )
        if bootstrap_inputs is not None
        else {
            name: {"gap_ci95": None, "relative_improvement_ci95": None}
            for name in COMPARISON_VARIANTS
        }
    )
    baseline_gap = _variant_gap(baseline, metric=metric, min_bucket_n=min_bucket_n)
    report = {
        "baseline": {
            "gap": baseline_gap,
            "status": "ok" if baseline_gap is not None else "insufficient_data",
            "gap_ci95": intervals["baseline"]["gap_ci95"],
        }
    }
    for name, summary in (("regularization", regularization), ("synthetic_blend", synthetic_blend)):
        gap = _variant_gap(summary, metric=metric, min_bucket_n=min_bucket_n)
        relative_improvement = None
        if gap is not None and baseline_gap is not None and baseline_gap != 0:
            relative_improvement = (baseline_gap - gap) / baseline_gap
        report[name] = {
            "gap": gap,
            "status": "ok" if gap is not None else "insufficient_data",
            "relative_improvement": relative_improvement,
            "gap_ci95": intervals[name]["gap_ci95"],
            "relative_improvement_ci95": intervals[name]["relative_improvement_ci95"],
        }
    return report


def mean_std(values) -> dict:
    """Mean and population std over seeds, NaN-safe."""
    vals = [float(v) for v in values if v == v]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "n": len(vals),
    }
