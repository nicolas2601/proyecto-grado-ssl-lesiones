"""C1 probe fits run under ``probe_threads.PROBE_BLAS_THREADS``, whatever the host.

Decision rule, stated before the check (resultados/clasificacion/c1_v7/diagnostico_hilos/):
pin C1 to the smallest explicit thread count that reproduces every published C1-v7
cell bit for bit. Default, 4, 8, 12, 24 and 48 all did; only 1 differed. So C1 and
v7.1 share one constant, and pinning moves no published number.
"""

from __future__ import annotations

import numpy as np
import pytest
from blas_spy import blas_threads

from dermssl import c1_grid_cells as gc
from dermssl import probe_threads, probes

N_CLASSES = 3


class ThreadSpy:
    """Stands in for a probe: records the BLAS threads it ran under, returns a valid proba."""

    def __init__(self):
        self.seen: list[int] = []

    def __call__(self, x_train, y_train, x_val, n_classes, **kwargs):
        self.seen.append(blas_threads())
        rng = np.random.default_rng(len(self.seen))
        raw = rng.random((len(x_val), n_classes))
        return raw / raw.sum(axis=1, keepdims=True)


@pytest.fixture
def spy(monkeypatch):
    spy = ThreadSpy()
    for name in probes.PROBES:
        monkeypatch.setitem(probes.PROBES, name, spy)
    return spy


@pytest.fixture(params=[None, 1], ids=["host-default", "host-one-thread"])
def host(request):
    """The caller's BLAS setting: whatever the machine resolved to, or capped at one."""
    from threadpoolctl import threadpool_limits  # noqa: PLC0415

    if request.param is None:
        yield
        return
    with threadpool_limits(limits=request.param):
        yield


def _features(n: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 4)).astype(np.float32), np.arange(n) % N_CLASSES


@pytest.mark.parametrize("pinned", [2, 3])
@pytest.mark.parametrize("probe", ["logreg", "knn"])
def test_a_c1_grid_cell_fits_its_probe_under_the_pinned_threads(monkeypatch, spy, host, pinned, probe):
    monkeypatch.setattr(probe_threads, "PROBE_BLAS_THREADS", pinned)
    x_train, y_train = _features(12, 0)
    x_val, y_val = _features(9, 1)
    gc.evaluate_cell(x_train, y_train, x_val, y_val, probe, N_CLASSES)
    assert spy.seen == [pinned]


def test_c1_and_v71_share_the_published_constant():
    assert probe_threads.PROBE_BLAS_THREADS == 4
