"""Exact metrics recomputed from a confusion matrix or from predicted probabilities.

The committed baseline JSON stores ``matriz_confusion`` (rows = true class, columns =
predicted class) but no MCC. The multiclass MCC (Gorodkin) is a function of that matrix
alone, so it is computed here in integer arithmetic and equals
``sklearn.metrics.matthews_corrcoef`` on the expanded label vectors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Matrix = Sequence[Sequence[int]]


def _check_square(cm: Matrix) -> None:
    size = len(cm)
    if size == 0 or any(len(row) != size for row in cm):
        raise ValueError("confusion matrix must be square and non-empty")


def mcc_from_confusion(cm: Matrix) -> float:
    """Multiclass Matthews correlation coefficient from a square count matrix.

    Returns 0.0 when the denominator vanishes (all predictions or all truths in one
    class), matching scikit-learn.
    """
    _check_square(cm)
    total = sum(sum(row) for row in cm)
    correct = sum(cm[k][k] for k in range(len(cm)))
    true_counts = [sum(row) for row in cm]
    pred_counts = [sum(row[k] for row in cm) for k in range(len(cm))]
    numerator = correct * total - sum(t * p for t, p in zip(true_counts, pred_counts))
    denominator = (total * total - sum(p * p for p in pred_counts)) * (
        total * total - sum(t * t for t in true_counts))
    return 0.0 if denominator == 0 else numerator / math.sqrt(denominator)


def macro_precision_recall(cm: Matrix) -> tuple[float, float]:
    """Macro precision and macro recall; a class never predicted contributes 0 precision."""
    _check_square(cm)
    size = len(cm)
    recalls, precisions = [], []
    for k in range(size):
        support = sum(cm[k])
        predicted = sum(cm[i][k] for i in range(size))
        recalls.append(cm[k][k] / support if support else 0.0)
        precisions.append(cm[k][k] / predicted if predicted else 0.0)
    return sum(precisions) / size, sum(recalls) / size


def confusion_from_probabilities(y_true: np.ndarray, proba: np.ndarray) -> list[list[int]]:
    """Count matrix of the argmax prediction (rows = true class)."""
    size = proba.shape[1]
    counts = np.zeros((size, size), dtype=np.int64)
    np.add.at(counts, (np.asarray(y_true, dtype=np.int64), proba.argmax(axis=1)), 1)
    return counts.tolist()


def row_normalized(cm: Matrix) -> list[list[float]]:
    """Each row divided by its support (recall per class on the diagonal); empty rows stay 0."""
    out = []
    for row in cm:
        support = sum(row)
        out.append([c / support if support else 0.0 for c in row])
    return out
