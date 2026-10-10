from __future__ import annotations

import numpy as np

N_BINS = 15
LOW_SUPPORT_THRESHOLD = 10
ECE_TOLERANCE = 1e-6


def reliability_data(labels: np.ndarray, probabilities: np.ndarray, published_ece: float,
                     bins: int = N_BINS) -> dict:
    y = np.asarray(labels, dtype=int)
    proba = np.asarray(probabilities, dtype=float)
    if (y.ndim != 1 or proba.ndim != 2 or len(y) != len(proba) or bins < 1
            or proba.shape[1] < 2 or not np.isfinite(proba).all()
            or np.any(proba < 0) or np.any(proba > 1)
            or not np.allclose(proba.sum(axis=1), 1, atol=1e-6)
            or np.any(y < 0) or np.any(y >= proba.shape[1])):
        raise ValueError("C6 requires aligned labels and valid class probabilities")
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    ece = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence > low) & (confidence <= high) if index else (confidence >= low) & (confidence <= high)
        support = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if support else None
        accuracy = float(correct[mask].mean()) if support else None
        if support:
            ece += support / len(y) * abs(mean_confidence - accuracy)
        rows.append({"index": index, "low": float(low), "high": float(high),
                     "confidence": mean_confidence, "accuracy": accuracy, "n": support,
                     "low_support": 0 < support < LOW_SUPPORT_THRESHOLD})
    matches = abs(ece - float(published_ece)) <= ECE_TOLERANCE
    return {"confidence_definition": "maximum class probability", "correctness": "argmax equals true class",
            "bin_count": bins, "bins": rows, "ece_recomputed": ece,
            "published_ece": float(published_ece), "ece_matches_published": matches,
            "ece": ece if matches else None}
