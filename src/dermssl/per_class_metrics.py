"""Per-class view of a classifier from its saved probabilities.

The headline tables report macro averages; this module opens them up per
class (confusion matrix, F1, recall) and for one clinically relevant class
(melanoma sensitivity and its one-vs-rest AUC). It composes the numpy
definitions in ``dermssl.metrics`` so every number here uses the same
definition as the published macro-F1 and AUC, ties included.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from dermssl import metrics

HAM10000_CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MELANOMA = "mel"
# Same scheme and seed as dermssl.paired_bootstrap (contrast_record's
# rng_seed): np.random.RandomState(seed), n image indices drawn with
# replacement per resample, percentile interval. Reused via metrics.bootstrap_ci.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20260923


def check_class_order(classes: Sequence[str], expected: Sequence[str] = HAM10000_CLASSES) -> None:
    """Fail if the probability columns would be read with the wrong labels."""
    if tuple(classes) != tuple(expected):
        raise ValueError(f"orden de clases inesperado: {list(classes)} != {list(expected)}")


def class_report(y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str],
                 positive: str = MELANOMA) -> dict:
    """Confusion matrix, per-class F1/recall and the positive-class metrics.

    Prediction is the argmax of ``proba``. ``positive_sensitivity`` is the
    recall of ``positive``; ``positive_auc`` scores its probability column
    against every other class (rank based, tied scores share their rank).
    """
    if positive not in classes:
        raise ValueError(f"clase positiva {positive!r} no esta en {list(classes)}")
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    n = len(classes)
    pred = proba.argmax(axis=1)
    f1 = metrics.per_class_f1(y_true, pred, n)
    recall = metrics.recall_per_class(y_true, pred, n)
    k = list(classes).index(positive)
    return {
        "confusion": metrics.confusion_matrix(y_true, pred, n),
        "f1": f1,
        "recall": recall,
        "macro_f1": float(f1.mean()),
        "positive_sensitivity": float(recall[k]),
        "positive_auc": metrics.binary_auc((y_true == k).astype(int), proba[:, k]),
    }


def positive_intervals(y_true: np.ndarray, proba: np.ndarray, classes: Sequence[str],
                       positive: str = MELANOMA, n_resamples: int = BOOTSTRAP_RESAMPLES,
                       seed: int = BOOTSTRAP_SEED) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap 95 % CI over images for the positive-class metrics."""
    if positive not in classes:
        raise ValueError(f"clase positiva {positive!r} no esta en {list(classes)}")
    k, n = list(classes).index(positive), len(classes)

    def sensitivity(y: np.ndarray, p: np.ndarray) -> float:
        return float(metrics.recall_per_class(y, p.argmax(axis=1), n)[k])

    def auc(y: np.ndarray, p: np.ndarray) -> float:
        return metrics.binary_auc((y == k).astype(int), p[:, k])

    out = {}
    for name, fn in (("sensitivity", sensitivity), ("auc", auc)):
        _, low, high = metrics.bootstrap_ci(y_true, proba, fn, n_resamples=n_resamples, seed=seed)
        out[name] = (low, high)
    return out


def accuracy_with_ci(y_true: np.ndarray, proba: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """``(accuracy, low, high)``: argmax accuracy with the same bootstrap as above."""

    def acc(y: np.ndarray, p: np.ndarray) -> float:
        return metrics.accuracy(y, p.argmax(axis=1))

    return metrics.bootstrap_ci(y_true, proba, acc, n_resamples=n_resamples, seed=seed)


def majority_accuracy(y_true: np.ndarray, classes: Sequence[str], label: str) -> float:
    """Accuracy of always predicting ``label``: its share of ``y_true``."""
    if label not in classes:
        raise ValueError(f"clase {label!r} no esta en {list(classes)}")
    return float((np.asarray(y_true) == list(classes).index(label)).mean())
