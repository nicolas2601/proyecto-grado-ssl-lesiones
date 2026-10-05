"""Fold StandardScaler + multinomial logistic regression into one linear layer.

The probe scores ``softmax(W @ ((x - mu) / sigma) + b)``. Expanding the
standardisation gives ``softmax(W' @ x + b')`` with, per class row,

    W' = W / sigma
    b' = b - W @ (mu / sigma)

``sigma`` is ``scaler.scale_``, which StandardScaler already sets to 1 for
features with (near) zero training variance, so the fold inherits that rule
instead of re-deriving it.
"""

from __future__ import annotations

import numpy as np


def fuse_probe(scaler, logreg, n_classes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(W', b')`` in float64 from a fitted scaler and multinomial logreg."""
    coef = np.asarray(logreg.coef_, dtype=np.float64)
    intercept = np.asarray(logreg.intercept_, dtype=np.float64)
    if coef.shape[0] == 1:
        raise ValueError("binary logreg has one row; the fold expects a multinomial probe")
    expected = coef.shape[0] if n_classes is None else n_classes
    if list(logreg.classes_) != list(range(expected)):
        raise ValueError(f"probe classes {list(logreg.classes_)} are not 0..{expected - 1}")
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    weight = coef / scale
    bias = intercept - coef @ (mean / scale)
    return weight, bias


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fused_predict_proba(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Class probabilities of the fused layer, in float64."""
    return softmax(np.asarray(x, dtype=np.float64) @ weight.T + bias)
