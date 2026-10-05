"""Probe side of one v7.1 unit (one set of weights at one resolution).

For every feature of §2.2/§2.7 the C1 ``logreg`` probe (``probes.fit_logreg``,
C = 1, standardised) is fitted twice on the fit split's identity view, flat
and ``class_weight='balanced'``, and never on TTA views (§2.7 item 4). The
logit-adjusted cells are post-hoc on the flat fit with the fit split's class
prior (§2.7 item 2); the TTA cells average the probe's softmax over the eight
D4 views of the evaluated split.

``parts`` are ``name -> (n_views, n_images, dim)`` arrays from
``torch_v71.extract_split``.
"""

from __future__ import annotations

import numpy as np

from dermssl import probe_threads, probes
from dermssl import v71_metrics as vm
from dermssl import v71_protocol as vp

_COMPOSITION = {
    "cls": ("cls",),
    "cls_mean": ("cls", "mean"),
    "cls4": ("cls4",),
    "cls_gem": ("cls", "gem"),
    "cls_attn": ("cls", "attn"),
}


def compose(parts: dict, feature: str) -> np.ndarray:
    return np.concatenate([parts[name] for name in _COMPOSITION[feature]], axis=-1)


def fit_predict_views(x_fit: np.ndarray, y_fit: np.ndarray, x_views: np.ndarray,
                      n_classes: int, class_weight: str | None) -> np.ndarray:
    """``(n_views, n, n_classes)``: one fit, scored on every view (missing classes -> 0)."""
    scaler, model = probes.fit_logreg(x_fit, y_fit, class_weight=class_weight)
    out = np.zeros((len(x_views), x_views.shape[1], n_classes), dtype=np.float64)
    columns = [int(c) for c in model.classes_]
    for view, x in enumerate(x_views):
        out[view][:, columns] = model.predict_proba(scaler.transform(x))
    return out


def _fit_jobs(fit_parts, y_fit, eval_parts, n_classes, features):
    for feature in features:
        x_fit = compose(fit_parts, feature)[0]
        x_eval = compose(eval_parts, feature)
        for fitted in vp.FITTED_WEIGHTINGS:
            weight = "balanced" if fitted == "balanced" else None
            yield (feature, fitted), (x_fit, y_fit, x_eval, n_classes, weight)


def _cells(feature: str, fits: dict, prior: np.ndarray, n_views: int) -> dict:
    flat, balanced = fits[(feature, "none")], fits[(feature, "balanced")]
    by_weighting = {"none": flat, "balanced": balanced}
    for tau in vp.TAUS:
        by_weighting[f"la{tau}"] = np.stack([vp.logit_adjust(v, prior, tau) for v in flat])
    tta_options = vp.TTA_OPTIONS if n_views > 1 else (False,)
    return {(feature, w, tta): (views.mean(axis=0) if tta else views[0])
            for w, views in by_weighting.items() for tta in tta_options}


def unit_probas(fit_parts: dict, y_fit: np.ndarray, eval_parts: dict, n_classes: int, *,
                features=vp.FEATURES, n_jobs: int = 1) -> dict:
    """``(feature, weighting, tta) -> proba`` for every registered cell of the unit.

    ``n_jobs`` only sets how many fits run at once; each fit gets
    ``probe_threads.PROBE_BLAS_THREADS`` BLAS threads, so the cells do not depend on it.
    """
    jobs = list(_fit_jobs(fit_parts, y_fit, eval_parts, n_classes, features))
    results = probe_threads.map_pinned(fit_predict_views, [args for _, args in jobs], n_jobs=n_jobs)
    fits = {key: result for (key, _), result in zip(jobs, results)}
    prior = vp.class_prior(y_fit, n_classes)
    n_views = next(iter(eval_parts.values())).shape[0]
    cells: dict = {}
    for feature in features:
        cells.update(_cells(feature, fits, prior, n_views))
    return cells


def score_cell(y_eval, proba, *, mel: int, n_resamples: int) -> dict:
    """Point metrics, their 95 % intervals and the holdout operating thresholds."""
    row = vm.metric_bundle(y_eval, proba, mel=mel)
    row["ci95"] = vm.bootstrap_bundle(y_eval, proba, mel=mel, n_resamples=n_resamples)
    row["thresholds"] = vm.operating_thresholds(y_eval, proba, mel=mel)
    return row


def score_unit(weights: str, resolution: int, y_eval, probas: dict, *, mel: int,
               n_resamples: int = vm.BOOTSTRAP_RESAMPLES, n_jobs: int = 1) -> dict:
    """``config_id -> row`` for every cell of the unit."""
    from joblib import Parallel, delayed  # noqa: PLC0415

    keys = list(probas)
    rows = Parallel(n_jobs=n_jobs)(
        delayed(score_cell)(y_eval, probas[k], mel=mel, n_resamples=n_resamples) for k in keys)
    return {vp.config_id(weights, resolution, *key): row for key, row in zip(keys, rows)}
