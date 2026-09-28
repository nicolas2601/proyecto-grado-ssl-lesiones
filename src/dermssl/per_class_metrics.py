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
