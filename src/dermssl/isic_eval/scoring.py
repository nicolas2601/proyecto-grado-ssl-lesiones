"""Metrics, bootstrap intervals and paired contrasts of one evaluated cell (docs/33 section 5).

Resampling is by image, or by whole groups (``patient_id`` for ISIC 2020) through
``paired_bootstrap.index_sampler``; 1000 resamples, seed 20260923, percentile interval, the stream the
published intervals use. The MCC comes from the confusion matrix (``bench.metrics``).
"""

from __future__ import annotations

import numpy as np

from dermssl import metrics as m
from dermssl import paired_bootstrap as pb
from dermssl import v71_metrics as vm

N_RESAMPLES = vm.BOOTSTRAP_RESAMPLES
SEED = vm.BOOTSTRAP_SEED
CONTRAST_METRICS = ("macro_f1", "balanced_accuracy", "accuracy", "auc", "ece", "mcc")
HEADLINE = ("macro_f1", "balanced_accuracy", "accuracy", "auc", "ece", "mcc")


mcc = pb.metric_mcc


def accuracy(y_true, proba) -> float:
    return m.accuracy(np.asarray(y_true), np.asarray(proba).argmax(axis=1))


def bundle(y_true, proba) -> dict:
    """The headline metrics of one sample (argmax prediction; AUC macro one-vs-rest)."""
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    n, pred = proba.shape[1], proba.argmax(axis=1)
    return {"macro_f1": m.macro_f1(y_true, pred, n), "balanced_accuracy": m.balanced_accuracy(y_true, pred, n),
            "accuracy": m.accuracy(y_true, pred), "auc": vm.auc_ovr_macro(y_true, proba),
            "ece": pb.metric_ece(y_true, proba), "mcc": mcc(y_true, proba)}


def per_class(y_true, proba, classes) -> dict:
    y_true, n = np.asarray(y_true), len(classes)
    pred = np.asarray(proba).argmax(axis=1)
    cm = m.confusion_matrix(y_true, pred, n)
    prec, rec, f1 = (fn(y_true, pred, n) for fn in (m.precision_per_class, m.recall_per_class, m.per_class_f1))
    return {"per_class": {c: {"precision": float(prec[k]), "recall": float(rec[k]), "f1": float(f1[k]),
                              "support": int(cm[k].sum())} for k, c in enumerate(classes)},
            "matriz_confusion": cm.tolist(), "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean())}


def bootstrap_ci(y_true, proba, *, groups=None, n_resamples: int = N_RESAMPLES, seed: int = SEED) -> dict:
    """Percentile 95 % interval of each headline metric; whole groups are redrawn when ``groups`` is given."""
    y_true, proba = np.asarray(y_true), np.asarray(proba, dtype=float)
    draw, rng = pb.index_sampler(len(y_true), groups), np.random.RandomState(seed)
    draws = [bundle(y_true[i], proba[i]) for i in (draw(rng) for _ in range(n_resamples))]
    return {name: vm.percentile_interval([d[name] for d in draws]) for name in HEADLINE}


def score(y_true, proba, classes, *, groups=None, n_resamples: int = N_RESAMPLES) -> dict:
    """Point metrics, per-class report, confusion matrix and the bootstrap block of one cell."""
    return {**bundle(y_true, proba), **per_class(y_true, proba, classes),
            "ci95": bootstrap_ci(y_true, proba, groups=groups, n_resamples=n_resamples),
            "bootstrap": {"n_resamples": n_resamples, "seed": SEED, "unit": "patient_id" if groups is not None else "image"}}


def _metric_fn(metric: str):
    return {"accuracy": accuracy, "mcc": mcc}.get(metric) or pb.METRIC_FUNCTIONS[metric]


def contrasts(arm: str, control: str, y_true, arm_proba, control_proba, *, groups=None,
              n_resamples: int = N_RESAMPLES) -> list[dict]:
    """Paired bootstrap of ``arm - control`` for each contrast metric, both arms on the same draws."""
    return [pb.contrast_record(arm=arm, control=control, metric=name, probe="logreg", fraction=1.0, seed=0,
                               y_true=y_true, arm_proba=arm_proba, control_proba=control_proba,
                               n_resamples=n_resamples, metric_fn=_metric_fn(name), groups=groups)
            for name in CONTRAST_METRICS]
