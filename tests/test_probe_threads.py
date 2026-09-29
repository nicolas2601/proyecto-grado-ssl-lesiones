"""Probe fits must not depend on how many workers run them.

Measured on the pod (resultados/clasificacion/t4/diagnostico_reproduccion/):
loky gives each worker ``cpu_count // n_jobs`` BLAS threads, and the lbfgs
logistic probe is not bit-stable across that count. ``--probe-jobs`` may only
change how many fits run at once, never the BLAS threads inside one fit.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import v71_grid as vg

N_CLASSES, DIM = 3, 4


def _blas_threads() -> int:
    from threadpoolctl import threadpool_info  # noqa: PLC0415

    blas = [pool["num_threads"] for pool in threadpool_info() if pool["user_api"] == "blas"]
    return blas[0] if blas else -1


def thread_spy(x_fit, y_fit, x_views, n_classes, class_weight):
    """Stands in for the fit: returns the worker's BLAS thread count as the 'proba'."""
    return np.full((len(x_views), x_views.shape[1], n_classes), float(_blas_threads()))


def _parts(n: int, views: int, seed: int):
    rng = np.random.RandomState(seed)
    y = np.arange(n) % N_CLASSES
    parts = {name: rng.normal(size=(views, n, DIM)).astype(np.float32)
             for name in ("cls", "mean", "gem", "attn")}
    parts["cls4"] = np.concatenate([parts["cls"]] * 4, axis=2)
    return parts, y


@pytest.mark.parametrize("n_jobs", [1, 2, 3])
def test_every_probe_fit_runs_under_the_pinned_blas_threads(monkeypatch, n_jobs):
    from dermssl.probe_threads import PROBE_BLAS_THREADS  # noqa: PLC0415

    monkeypatch.setattr(vg, "fit_predict_views", thread_spy)
    fit_parts, y_fit = _parts(12, 1, 0)
    eval_parts, _ = _parts(6, 2, 1)
    cells = vg.unit_probas(fit_parts, y_fit, eval_parts, N_CLASSES, features=("cls",), n_jobs=n_jobs)
    seen = {float(v) for key, proba in cells.items() if key[1] in ("none", "balanced") for v in np.unique(proba)}
    assert seen == {float(PROBE_BLAS_THREADS)}
