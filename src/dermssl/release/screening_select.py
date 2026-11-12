"""Melanoma-priority operating point for a frozen reader (rule: resultados/clasificacion/v10/screening/RULE.md).

The shipped argmax is never changed. A second prediction adds ``delta`` to the log score of one
target class; ``delta`` is chosen on development out-of-fold probabilities only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

DELTA_GRID = tuple(float(value) for value in np.round(np.arange(0.0, 3.0 + 1e-9, 0.25), 2))
MACRO_F1_TOLERANCE = 0.01


def screening_predictions(probabilities: np.ndarray, target: int, delta: float) -> np.ndarray:
    """Argmax of ``log p + delta * onehot(target)``; a zero probability never wins."""
    if delta < 0 or not np.isfinite(delta):
        raise ValueError("delta must be a finite non-negative number")
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or not 0 <= target < p.shape[1]:
        raise ValueError("probabilities must be (rows, classes) and target a valid column")
    with np.errstate(divide="ignore"):
        scores = np.log(p)
    scores[:, target] += delta
    return np.argmax(scores, axis=1)


def select_delta(macro_f1_by_delta: Mapping[float, float], tolerance: float = MACRO_F1_TOLERANCE) -> float:
    """Largest delta whose macro-F1 stays within ``tolerance`` of the shipped argmax (delta = 0)."""
    if 0.0 not in macro_f1_by_delta:
        raise ValueError("the grid must contain delta = 0 (the shipped argmax)")
    floor = macro_f1_by_delta[0.0] - tolerance
    passing = [delta for delta, score in macro_f1_by_delta.items() if score >= floor]
    return float(max(passing))


def operating_metrics(
    y: np.ndarray, pred: np.ndarray, classes: Sequence[str], target: str, nevus: str
) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y, pred = np.asarray(y), np.asarray(pred)
    t, n = classes.index(target), classes.index(nevus)
    is_target, flagged = y == t, pred == t
    return {
        "target_recall": float(np.mean(flagged[is_target])),
        "target_to_nevus": float(np.mean(pred[is_target] == n)),
        "nevus_recall": float(np.mean(pred[y == n] == n)),
        "target_precision": float(np.mean(is_target[flagged])) if flagged.any() else 0.0,
        "target_alert_rate": float(np.mean(flagged)),
        "macro_f1": float(f1_score(y, pred, labels=range(len(classes)), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }
