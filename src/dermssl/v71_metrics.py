"""Per-configuration metrics of the v7.1 grid (docs/29 §2.3b and §2.7).

Every configuration gets the same bundle: the four published metrics, plus
accuracy and the melanoma secondaries (recall of ``mel`` at the argmax, AUC of
mel against the rest, and the three post-hoc operating points of §2.7 item 1).
The operating points never change the multiclass prediction or the macro-F1.

Thresholds are on p(mel), positive when ``score >= threshold``. On the holdout
they are fitted on the same sample they are scored on (inside every bootstrap
resample too); on validation they are the holdout thresholds, fixed.

The resampling stream is the one of ``selection_holdout.bootstrap_ci`` and
``paired_bootstrap.paired_contrast`` (``RandomState(seed).randint(0, n, n)``
per resample), so an interval here matches one computed there.
"""

from __future__ import annotations

import math

import numpy as np

from dermssl import metrics as _metrics
from dermssl import paired_bootstrap as pb

BOOTSTRAP_SEED = 20260923
BOOTSTRAP_RESAMPLES = 1000
SPEC_TARGETS = {"spec80": 0.80, "spec90": 0.90}
SENS_TARGETS = {"sens95": 0.95}
GRID_METRICS = (
    "macro_f1", "balanced_accuracy", "accuracy", "auc", "ece",
    "mel_recall", "mel_auc", "mel_sens_at_spec80", "mel_sens_at_spec90", "mel_spec_at_sens95",
)
_ROUNDING_SLACK = 1e-9


def required_count(target: float, n: int) -> int:
    """Smallest count reaching ``target * n``, immune to 0.8 * 1400 = 1120.0000000000002."""
    return max(1, math.ceil(target * n - _ROUNDING_SLACK))


def _split_scores(y_binary, scores) -> tuple[np.ndarray, np.ndarray]:
    y_binary, scores = np.asarray(y_binary).astype(bool), np.asarray(scores, dtype=float)
    if y_binary.all() or not y_binary.any():
        raise ValueError("un punto de operacion necesita positivos y negativos de mel")
    return scores[y_binary], scores[~y_binary]


def threshold_for_specificity(y_binary, scores, target: float) -> float:
    """Lowest threshold whose specificity is at least ``target`` (most sensitive)."""
    _, negatives = _split_scores(y_binary, scores)
    ordered = np.sort(negatives)
    return float(np.nextafter(ordered[required_count(target, len(ordered)) - 1], np.inf))


def threshold_for_sensitivity(y_binary, scores, target: float) -> float:
    """Highest threshold whose sensitivity is at least ``target`` (most specific)."""
    positives, _ = _split_scores(y_binary, scores)
    ordered = np.sort(positives)[::-1]
    return float(ordered[required_count(target, len(ordered)) - 1])


def sens_spec(y_binary, scores, threshold: float) -> tuple[float, float]:
    positives, negatives = _split_scores(y_binary, scores)
    return float(np.mean(positives >= threshold)), float(np.mean(negatives < threshold))


def operating_thresholds(y_true, proba, *, mel: int) -> dict:
    """The three §2.7 thresholds on p(mel), fitted on this sample."""
    y_mel, score = np.asarray(y_true) == mel, np.asarray(proba)[:, mel]
    out = {k: threshold_for_specificity(y_mel, score, t) for k, t in SPEC_TARGETS.items()}
    out.update({k: threshold_for_sensitivity(y_mel, score, t) for k, t in SENS_TARGETS.items()})
    return out


def binary_auc(y_binary, scores) -> float:
    """``metrics.binary_auc`` with scipy's tie-averaged ranks instead of a Python loop."""
    from scipy.stats import rankdata  # noqa: PLC0415 - ships with scikit-learn

    y_binary = np.asarray(y_binary).astype(bool)
    n_pos = int(y_binary.sum())
    n_neg = len(y_binary) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = rankdata(np.asarray(scores, dtype=float))[y_binary].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_ovr_macro(y_true, proba) -> float:
    y_true = np.asarray(y_true)
    aucs = [binary_auc(y_true == c, proba[:, c]) for c in range(proba.shape[1])]
    valid = [a for a in aucs if a == a]
    return float(np.mean(valid)) if valid else float("nan")


def _confusion_metrics(y_true: np.ndarray, pred: np.ndarray, n_classes: int) -> dict:
    """Macro-F1 and balanced accuracy from one bincount, same zero conventions as ``metrics``."""
    cm = np.bincount(y_true * n_classes + pred, minlength=n_classes ** 2).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    denom = 2 * tp + (predicted - tp) + (support - tp)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
    return {"macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean())}


def _mel_block(y_true, proba, mel: int) -> dict:
    y_mel, score = np.asarray(y_true) == mel, np.asarray(proba)[:, mel]
    if y_mel.all() or not y_mel.any():
        nan = float("nan")
        return {"mel_recall": nan, "mel_auc": nan, "mel_sens_at_spec80": nan,
                "mel_sens_at_spec90": nan, "mel_spec_at_sens95": nan}
    thr = operating_thresholds(y_true, proba, mel=mel)
    return {
        "mel_recall": float(np.mean(np.asarray(proba)[y_mel].argmax(axis=1) == mel)),
        "mel_auc": binary_auc(y_mel, score),
        "mel_sens_at_spec80": sens_spec(y_mel, score, thr["spec80"])[0],
        "mel_sens_at_spec90": sens_spec(y_mel, score, thr["spec90"])[0],
        "mel_spec_at_sens95": sens_spec(y_mel, score, thr["sens95"])[1],
    }


def metric_bundle(y_true, proba, *, mel: int) -> dict:
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    pred = proba.argmax(axis=1)
    return {
        **_confusion_metrics(y_true.astype(np.int64), pred, proba.shape[1]),
        "accuracy": _metrics.accuracy(y_true, pred),
        "auc": auc_ovr_macro(y_true, proba),
        "ece": pb.metric_ece(y_true, proba),
        **_mel_block(y_true, proba, mel),
    }


def resample_indices(n: int, n_resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.stack([rng.randint(0, n, n) for _ in range(n_resamples)])


def percentile_interval(values, alpha: float = 0.05) -> list[float]:
    """Percentile interval over the finite values; NaN when there are none."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan"), float("nan")]
    return [float(np.percentile(finite, 100 * alpha / 2)),
            float(np.percentile(finite, 100 * (1 - alpha / 2)))]


def bootstrap_bundle(y_true, proba, *, mel: int, n_resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED, bundle=None) -> dict:
    """Percentile 95 % interval of every metric of ``bundle`` (default: the grid bundle)."""
    bundle = bundle or (lambda y, p: metric_bundle(y, p, mel=mel))
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    draws = [bundle(y_true[i], proba[i]) for i in resample_indices(len(y_true), n_resamples, seed)]
    return {name: percentile_interval([d[name] for d in draws]) for name in draws[0]}


def fixed_threshold_metrics(y_true, proba, *, mel: int, thresholds: dict) -> dict:
    """Sensitivity and specificity at thresholds fixed elsewhere (the holdout)."""
    y_mel, score = np.asarray(y_true) == mel, np.asarray(proba)[:, mel]
    out = {}
    for key, threshold in thresholds.items():
        sens, spec = sens_spec(y_mel, score, threshold)
        out[f"mel_sens_at_thr_{key}"], out[f"mel_spec_at_thr_{key}"] = sens, spec
    return out
