"""Diagnostic: does the v7.1 logreg probe converge within its ``max_iter``?

Paula's hypothesis (2026-09-29): changing loky's inner BLAS threads (4 -> 8)
flips ~260 argmax per unit because lbfgs stops at ``max_iter`` before
converging, so the stopping point depends on the floating-point summation
order. This module refits EXACTLY the published fits (``probes.fit_logreg``
with its own defaults, through ``v71_grid``'s job list and cell builder),
with BLAS pinned to the published inner thread count, and records per fit
``n_iter_`` against ``max_iter`` and whether sklearn raised a
``ConvergenceWarning``. It changes no protocol and refits nothing published.

Fits happen on the identity view of the fit split only (§2.7 item 4), so a
fit key is (unit, feature, weighting); the eval views only get scored.
"""

from __future__ import annotations

import inspect
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from dermssl import probes
from dermssl import v71_grid as vg
from dermssl import v71_protocol as vp
from dermssl.probe_threads import PROBE_BLAS_THREADS

MAX_ITER = inspect.signature(probes.fit_logreg).parameters["max_iter"].default
# probe_jobs 10 on the 48-core pod -> loky gave each worker 48 // 10 = 4 BLAS
# threads: the setting that reproduced v7.1 bit for bit (t4/diagnostico_reproduccion).
BLAS_THREADS = PROBE_BLAS_THREADS


def fit_views_recorded(x_fit, y_fit, x_views, n_classes, class_weight, *, max_iter=MAX_ITER,
                       threads=BLAS_THREADS):
    """``vg.fit_predict_views`` plus its convergence record, BLAS pinned to ``threads``."""
    from sklearn.exceptions import ConvergenceWarning  # noqa: PLC0415
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    with threadpool_limits(limits=threads), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler, model = probes.fit_logreg(x_fit, y_fit, max_iter=max_iter, class_weight=class_weight)
        out = np.zeros((len(x_views), x_views.shape[1], n_classes), dtype=np.float64)
        columns = [int(c) for c in model.classes_]
        for view, x in enumerate(x_views):
            out[view][:, columns] = model.predict_proba(scaler.transform(x))
    n_iter = [int(n) for n in np.atleast_1d(model.n_iter_)]
    record = {"n_iter": n_iter, "max_iter": int(max_iter), "hit_max_iter": max(n_iter) >= max_iter,
              "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
              "n_views_scored": int(len(x_views)), "n_fit_rows": int(len(x_fit)), "dim": int(x_fit.shape[1])}
    return out, record


def unit_fits(unit: str, fit_parts, y_fit, eval_parts, n_classes, *, features=None,
              n_jobs: int = 1, threads: int = BLAS_THREADS):
    """``(cells, records)``: the unit's cells as ``vg.unit_probas`` builds them, one record per fit.

    ``features`` defaults to every registered feature, read at call time.
    """
    from joblib import Parallel, delayed  # noqa: PLC0415

    features = vp.FEATURES if features is None else features

    jobs = list(vg._fit_jobs(fit_parts, y_fit, eval_parts, n_classes, features))
    results = Parallel(n_jobs=n_jobs)(delayed(fit_views_recorded)(*args, threads=threads) for _, args in jobs)
    fits = {key: views for (key, _), (views, _) in zip(jobs, results)}
    records = [{"unit": unit, "feature": f, "weighting": w, **rec} for ((f, w), _), (_, rec) in zip(jobs, results)]
    prior = vp.class_prior(y_fit, n_classes)
    n_views = next(iter(eval_parts.values())).shape[0]
    cells: dict = {}
    for feature in features:
        cells.update(vg._cells(feature, fits, prior, n_views))
    return cells, records


def compare_probas(weights: str, resolution: int, cells: dict, reference: dict) -> dict:
    """Refit cells (cast to the stored float32) against the published ``probas.npz``, bit for bit."""
    differing, missing = [], []
    for key, proba in cells.items():
        cid = vp.config_id(weights, resolution, *key)
        if cid not in reference:
            missing.append(cid)
        elif not np.array_equal(proba.astype(np.float32), reference[cid]):
            differing.append(cid)
    return {"n_cells": len(cells), "differing_cells": differing, "missing_cells": missing,
            "passed": not differing and not missing}


def compare_metrics(weights: str, resolution: int, cells: dict, rows: dict, y, *, mel: int) -> dict:
    """Point metrics of the refit cells against the committed grid rows, exact equality."""
    from dermssl import v71_metrics as vm  # noqa: PLC0415

    differing, missing = [], []
    for key, proba in cells.items():
        cid = vp.config_id(weights, resolution, *key)
        if cid not in rows:
            missing.append(cid)
            continue
        bundle = vm.metric_bundle(y, proba, mel=mel)
        if any(rows[cid].get(m) != v for m, v in bundle.items()):
            differing.append(cid)
    return {"n_cells": len(cells), "differing_cells": differing, "missing_cells": missing,
            "passed": not differing and not missing}


def _counts(records) -> dict:
    return {"n_fits": len(records), "n_hit_max_iter": sum(r["hit_max_iter"] for r in records),
            "n_convergence_warning": sum(r["convergence_warning"] for r in records)}


def summarize(records) -> dict:
    out = _counts(records)
    for axis in ("unit", "feature", "weighting"):
        groups = defaultdict(list)
        for r in records:
            groups[r[axis]].append(r)
        out[f"by_{axis}"] = {name: _counts(group) for name, group in groups.items()}
    return out


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def render_readme(doc: dict) -> str:
    s = doc["summary"]
    lines = [
        "# Diagnostic: convergence of the v7.1 logreg probe",
        "",
        "Generated by `scripts/probe_convergence_check.py` from `probe_convergence.json`; do not edit.",
        "Diagnostic only: no protocol changes and nothing published is refitted or replaced.",
        "",
        f"Refit of every published holdout-grid fit at 224 (identity input), `max_iter={doc['max_iter']}`, "
        f"BLAS threads pinned to {doc['blas_threads']}, code revision `{doc['code_revision']}`.",
        f"Sanity check (refit probas bit-equal to the published ones and point metrics equal to the "
        f"committed grid): **{'passed' if doc['sanity']['passed'] else 'FAILED'}**.",
        "",
        f"- Fits: {s['n_fits']}",
        f"- Hit `max_iter`: {s['n_hit_max_iter']} of {s['n_fits']}",
        f"- Raised `ConvergenceWarning`: {s['n_convergence_warning']} of {s['n_fits']}",
        "",
        "| unit | feature | weighting | n_iter | max_iter | hit cap | ConvergenceWarning |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in doc["fits"]:
        lines.append(f"| {r['unit']} | {r['feature']} | {r['weighting']} | {max(r['n_iter'])} | {r['max_iter']} "
                     f"| {_yes(r['hit_max_iter'])} | {_yes(r['convergence_warning'])} |")
    for axis in ("unit", "feature", "weighting"):
        lines += ["", f"By {axis}:", "", f"| {axis} | fits | hit cap | ConvergenceWarning |", "|---|---|---|---|"]
        for name, c in s[f"by_{axis}"].items():
            lines.append(f"| {name} | {c['n_fits']} | {c['n_hit_max_iter']} | {c['n_convergence_warning']} |")
    return "\n".join(lines) + "\n"


def check_out_dir(out_dir: Path, *, protected) -> None:
    """The diagnostic never writes inside a published run or results directory."""
    target = Path(out_dir).expanduser().resolve()
    for root in protected:
        root = Path(root).expanduser().resolve()
        if target == root or root in target.parents:
            raise SystemExit(f"ABORT: {target} esta dentro de {root}, que es publicado; usar otro --out-dir")
