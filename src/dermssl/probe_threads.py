"""BLAS threads inside every probe fit, fixed so ``--probe-jobs`` cannot move a number.

loky gives each worker ``cpu_count() // n_jobs`` BLAS threads by default, and the
lbfgs logistic probe (``probes.fit_logreg``) is not bit-stable across that count.
Measured on the 48-CPU pod (resultados/clasificacion/t4/diagnostico_reproduccion/):
``probe_jobs=10`` (4 inner threads) reproduces the published v7.1 grid bit for bit
(180 cells, max delta 0), while ``probe_jobs=6`` (8 inner threads) on bit-equal
features flips about 260 argmaxes per unit and moves macro-F1 by up to 0.014.

So the published numbers are the 4-thread numbers, and every probe fit runs with
exactly ``PROBE_BLAS_THREADS`` whatever the worker count or the host.
"""

from __future__ import annotations

PROBE_BLAS_THREADS = 4


def call_pinned(fn, *args):
    """``fn(*args)`` with the loaded BLAS/OpenMP pools capped at ``PROBE_BLAS_THREADS``."""
    from threadpoolctl import threadpool_limits  # noqa: PLC0415 - ships with scikit-learn

    with threadpool_limits(limits=PROBE_BLAS_THREADS):
        return fn(*args)


def map_pinned(fn, arg_tuples, n_jobs: int = 1) -> list:
    """``[fn(*args) ...]`` on ``n_jobs`` loky workers, each fit under ``PROBE_BLAS_THREADS``.

    ``inner_max_num_threads`` sets the workers' thread env from start-up; the
    ``threadpool_limits`` in ``call_pinned`` also covers ``n_jobs=1``, where joblib
    runs in-process with every core.
    """
    from joblib import Parallel, delayed, parallel_config  # noqa: PLC0415

    with parallel_config(backend="loky", inner_max_num_threads=PROBE_BLAS_THREADS):
        return Parallel(n_jobs=n_jobs)(delayed(call_pinned)(fn, *args) for args in arg_tuples)
