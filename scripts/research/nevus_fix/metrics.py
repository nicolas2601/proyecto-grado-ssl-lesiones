"""Metrics in the app ontology (HAM7), with focus on nevus collapse and melanoma misses."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from labels import HAM7, U8_TO_HAM7

MALIGNANT = ("akiec", "bcc", "mel")


def project_pad6(p6: np.ndarray) -> np.ndarray:
    """PAD6 (ACK, BCC, MEL, NEV, SCC, SEK) -> HAM7 probabilities; df and vasc get zero mass."""
    out = np.zeros((len(p6), len(HAM7)), dtype=np.float64)
    out[:, HAM7.index("akiec")] = p6[:, 0] + p6[:, 4]
    out[:, HAM7.index("bcc")] = p6[:, 1]
    out[:, HAM7.index("mel")] = p6[:, 2]
    out[:, HAM7.index("nv")] = p6[:, 3]
    out[:, HAM7.index("bkl")] = p6[:, 5]
    return out


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, s))


def evaluate(u8_labels: np.ndarray, p7: np.ndarray) -> dict:
    """All metrics on HAM7 ground truth; recall/bacc only over classes present."""
    y = np.asarray([HAM7.index(U8_TO_HAM7[u]) for u in u8_labels])
    pred = p7.argmax(1)
    present = sorted(set(y.tolist()))
    recalls = {HAM7[c]: float((pred[y == c] == c).mean()) for c in present}
    nv, mel = HAM7.index("nv"), HAM7.index("mel")
    non_nv = y != nv
    malignant = np.isin(y, [HAM7.index(c) for c in MALIGNANT])
    mal_score = p7[:, [HAM7.index(c) for c in MALIGNANT]].sum(1)
    melnv = np.isin(y, [mel, nv])
    out = {
        "n": int(len(y)),
        "counts": {HAM7[c]: int((y == c).sum()) for c in present},
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "macro_f1": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "accuracy": float((pred == y).mean()),
        "recall": recalls,
        "pred_nv_rate": float((pred == nv).mean()),
        "non_nv_predicted_nv": float((pred[non_nv] == nv).mean()) if non_nv.any() else None,
        "mel_to_nv": float((pred[y == mel] == nv).mean()) if (y == mel).any() else None,
        "malignant_recall_any": float(np.isin(pred[malignant], [HAM7.index(c) for c in MALIGNANT]).mean())
        if malignant.any() else None,
        "malignant_auc": _safe_auc(malignant.astype(int), mal_score),
        "mel_vs_nv_auc": _safe_auc((y[melnv] == mel).astype(int),
                                   p7[melnv, mel] / np.clip(p7[melnv, mel] + p7[melnv, nv], 1e-12, None)),
    }
    return out
