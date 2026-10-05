"""Probabilities of the two arms (full image, F4) and the metric row of one arm on validation."""

from __future__ import annotations

import numpy as np

from dermssl import e3_descriptors as desc
from dermssl import e3_stages as stages
from dermssl import e3_variants as var
from dermssl import metrics
from dermssl import paired_bootstrap as pb
from dermssl import v71_metrics as vm
from dermssl.bench.metrics import mcc_from_confusion


def arm_probas(x_fit: dict, x_eval: dict, desc_fit: np.ndarray, desc_eval: np.ndarray, y_fit,
               n_classes: int) -> tuple[dict, dict]:
    """``({"full", "F4"} -> (n, classes), descriptor info)``; one view, the probe of E3 (balanced logreg).

    ``x_*`` map ``full`` and ``crop_view`` to ``(n, dim)``. The descriptors are standardised on
    the fit split, NaN (empty mask) filled with the fit mean, exactly like ``e3_stages``.
    """
    fit = {c: np.asarray(x_fit[c])[None] for c in ("full", "crop_view")}
    evaluated = {c: np.asarray(x_eval[c])[None] for c in ("full", "crop_view")}
    z_fit, z_eval, info = desc.standardize(desc_fit, desc_eval)
    probas = {
        "full": var.variant_proba("B71", {"full": fit["full"]}, y_fit, {"full": evaluated["full"]}, n_classes),
        "F4": var.variant_proba("F4", fit, y_fit, evaluated, n_classes, desc_fit=z_fit, desc_eval=z_eval),
    }
    return probas, info


def _interval_functions(y, n_classes: int, mel: int) -> dict:
    def mcc(yt, proba):
        return mcc_from_confusion(metrics.confusion_matrix(yt, proba.argmax(axis=1), n_classes))

    fns = {**pb.METRIC_FUNCTIONS, "mcc": mcc}
    fns["accuracy"] = lambda yt, proba: metrics.accuracy(yt, proba.argmax(axis=1))
    fns["mel_recall"] = stages._metric_fns(y, n_classes, mel).get("mel_recall")
    return {name: fn for name, fn in fns.items() if fn is not None}


def _per_class(y, pred, classes) -> dict:
    n = len(classes)
    return {
        "recall_por_clase": dict(zip(classes, map(float, metrics.recall_per_class(y, pred, n)))),
        "precision_por_clase": dict(zip(classes, map(float, metrics.precision_per_class(y, pred, n)))),
        "f1_por_clase": dict(zip(classes, map(float, metrics.per_class_f1(y, pred, n)))),
    }


def score_row(y, proba, groups, classes, *, mel: int, n_resamples: int) -> dict:
    """Point metrics (bank schema + MCC) and 95 % percentile intervals grouped by ``lesion_id``."""
    y, proba, n = np.asarray(y), np.asarray(proba, dtype=float), len(classes)
    pred = proba.argmax(axis=1)
    matrix = metrics.confusion_matrix(y, pred, n)
    bundle = vm.metric_bundle(y, proba, mel=mel)
    return {
        "n": int(len(y)),
        **{k: bundle[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "ece", "mel_recall", "mel_auc")},
        "macro_precision": metrics.macro_precision(y, pred, n), "macro_recall": metrics.macro_recall(y, pred, n),
        "auc_ovr_macro": bundle["auc"], "mcc": mcc_from_confusion(matrix),
        **_per_class(y, pred, classes), "matriz_confusion": matrix.tolist(),
        "ci95_grouped": stages._grouped_ci(y, proba, groups, _interval_functions(y, n, mel), n_resamples),
    }
