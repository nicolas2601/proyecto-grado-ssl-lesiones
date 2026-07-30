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
