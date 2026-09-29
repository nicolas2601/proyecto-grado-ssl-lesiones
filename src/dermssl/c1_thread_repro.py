"""Diagnostic: which BLAS thread count reproduces the published C1 probas bit for bit?

The lbfgs logreg probe (``probes.fit_logreg``) is not bit-stable across BLAS
thread counts: on the 48-CPU pod, 4 vs 8 threads moved v7.1 macro-F1 by up to
0.014 (resultados/clasificacion/t4/diagnostico_reproduccion/). v7.1 and T4 are
pinned by ``probe_threads``; C1 never was -- its grid fitted in-process with
whatever the library default resolved to. This module refits published C1
cells under explicit thread caps and compares each refit with the published
``.proba.npy``. It changes no protocol and rewrites nothing published.
"""

from __future__ import annotations

import numpy as np

DEFAULT = "default"
RECORDED = "recorded"
POOL_KEYS = ("user_api", "internal_api", "num_threads")


def parse_settings(raw, *, recorded: int | None) -> list[tuple[str, int | None]]:
    """``[(label, cap)]``: ``default`` is no cap, ``recorded`` the environment's value, else an int.

    ``recorded`` is dropped when nothing was recorded: a label with no value
    would read as a setting that was tried.
    """
    out = []
    for label in raw:
        if label == DEFAULT:
            out.append((DEFAULT, None))
        elif label == RECORDED:
            if recorded is not None:
                out.append((RECORDED, int(recorded)))
        else:
            try:
                count = int(label)
            except ValueError as exc:
                raise ValueError(f"thread setting {label!r}: expected default, recorded or a positive int") from exc
            if count < 1:
                raise ValueError(f"thread setting {label!r}: must be >= 1")
            out.append((str(count), count))
    return out


def _pools() -> list[dict]:
    from threadpoolctl import threadpool_info  # noqa: PLC0415 - ships with scikit-learn

    return [{k: pool.get(k) for k in POOL_KEYS} for pool in threadpool_info()]


def fit_with_threads(fn, *args, threads: int | None):
    """``(fn(*args), pools)`` with every BLAS/OpenMP pool capped at ``threads``; ``None`` = no cap.

    ``pools`` is what threadpoolctl saw INSIDE the context, so the default
    setting records the count it actually resolved to on this host.
    """
    if threads is None:
        return fn(*args), _pools()
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    with threadpool_limits(limits=threads):
        return fn(*args), _pools()


def compare_proba(refit, published, *, refit_macro_f1: float, published_macro_f1: float) -> dict:
    """Bit equality (values AND dtype), max |delta|, argmax flips and |delta macro-F1|."""
    refit, published = np.asarray(refit), np.asarray(published)
    if refit.shape != published.shape:
        raise ValueError(f"refit {refit.shape} and published {published.shape} are not the same cell")
    delta = np.abs(refit.astype(np.float64) - published.astype(np.float64))
    return {
        "bit_equal": bool(refit.dtype == published.dtype and np.array_equal(refit, published)),
        "max_abs_delta": float(delta.max()) if delta.size else 0.0,
        "argmax_flips": int((refit.argmax(axis=1) != published.argmax(axis=1)).sum()),
        "abs_delta_macro_f1": abs(float(refit_macro_f1) - float(published_macro_f1)),
        "refit_macro_f1": float(refit_macro_f1),
        "published_macro_f1": float(published_macro_f1),
    }
