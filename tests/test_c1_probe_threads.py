"""C1 probe fits run under ``probe_threads.PROBE_BLAS_THREADS``, whatever the host.

Decision rule, stated before the check (resultados/clasificacion/c1_v7/diagnostico_hilos/):
pin C1 to the smallest explicit thread count that reproduces every published C1-v7
cell bit for bit. Default, 4, 8, 12, 24 and 48 all did; only 1 differed. So C1 and
v7.1 share one constant, and pinning moves no published number.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from blas_spy import blas_threads
from env_sidecar import sidecar_env

from dermssl import c1_grid_cells as gc
from dermssl import c1_split_identity as si
from dermssl import probe_threads, probes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1_final_eval as final_eval  # noqa: E402
import c1_select_variant as select_variant  # noqa: E402

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


# ------------------------------------------- the single evaluations (select, final)

FIT_IDS = [f"ISIC_{i:07d}" for i in range(24)]
HOLDOUT_IDS = [f"ISIC_{i:07d}" for i in range(100, 115)]


def _fake_torch() -> types.ModuleType:
    ns = types.SimpleNamespace
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-fake"
    torch.version = ns(cuda=None)
    torch.backends = ns(cudnn=ns(is_available=lambda: False, version=lambda: None,
                                 deterministic=False, benchmark=False))
    torch.are_deterministic_algorithms_enabled = lambda: False
    torch.get_num_threads = lambda: 1
    torch.cuda = ns(is_available=lambda: False, is_initialized=lambda: False, device_count=lambda: 0)
    torch.device = lambda name: f"device:{name}"
    torch.Tensor = type("Tensor", (), {})  # scipy/sklearn probe it for array-API dispatch
    return torch


def _extract(model, ids, *rest):
    rng = np.random.default_rng(len(ids))
    return rng.normal(size=(len(ids), 4)).astype(np.float32), np.arange(len(ids)) % N_CLASSES


@pytest.fixture
def evaluation_world(tmp_path, monkeypatch):
    """Everything around the probe faked: no torch, no images, no git."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    bank = select_variant.bank
    monkeypatch.setattr(bank, "index_images", lambda roots: {})
    monkeypatch.setattr(bank, "build_frozen_backbone", lambda *a: {"model": None})
    monkeypatch.setattr(bank, "extract_features", _extract)
    monkeypatch.setattr(select_variant.rp, "clean_revision", lambda repo: "f" * 40)
    split = ({"train_ids": FIT_IDS, "val_ids": [], "id2label": {}}, None)
    monkeypatch.setattr(select_variant, "rebuild_split", lambda root: split)
    monkeypatch.setattr(final_eval, "rebuild_split", lambda root: split)
    holdout = {"fit_ids": FIT_IDS, "holdout_ids": HOLDOUT_IDS,
               "fit_ids_sha256": si.ids_sha256(FIT_IDS), "holdout_ids_sha256": si.ids_sha256(HOLDOUT_IDS)}
    (tmp_path / "holdout.json").write_text(json.dumps(holdout), encoding="utf-8")
    (tmp_path / "exclude.txt").write_text("ISIC_0000999\n", encoding="utf-8")
    return tmp_path


def _common_argv(root: Path) -> list[str]:
    return ["--ham-root", str(root), "--holdout", str(root / "holdout.json"),
            "--exclude-ids-file", str(root / "exclude.txt"), "--device", "cpu"]


@pytest.mark.parametrize("pinned", [2, 3])
def test_the_selection_fits_every_arm_under_the_pinned_threads_and_records_them(
        monkeypatch, spy, evaluation_world, pinned):
    monkeypatch.setattr(probe_threads, "PROBE_BLAS_THREADS", pinned)
    out = evaluation_world / "selection.json"
    monkeypatch.setattr(sys, "argv", ["c1_select_variant.py", *_common_argv(evaluation_world),
                                      "--variant", "A1=dinov2_vitb14:a1", "--variant", "A2=dinov2_vitb14:a2",
                                      "--reference", "public=dinov2_vitb14:public", "--out", str(out)])
    assert select_variant.main() == 0
    assert spy.seen == [pinned] * 3
    assert sidecar_env(out)["knobs"]["loky_inner_threads"] == pinned


@pytest.mark.parametrize("pinned", [2, 3])
def test_the_v7_single_evaluation_fits_under_the_pinned_threads_and_records_them(
        monkeypatch, spy, evaluation_world, pinned):
    monkeypatch.setattr(probe_threads, "PROBE_BLAS_THREADS", pinned)
    out = evaluation_world / "final_eval.json"
    monkeypatch.setattr(sys, "argv", ["c1_final_eval.py", *_common_argv(evaluation_world), "--split", "holdout",
                                      "--plain", "dinov2_vitb14:public", "--proposed", "dinov2_vitb14:public",
                                      "--out", str(out)])
    assert final_eval.main() == 0
    assert spy.seen == [pinned] * 3
    assert sidecar_env(out)["knobs"]["loky_inner_threads"] == pinned
