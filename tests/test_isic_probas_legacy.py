"""ISIC 2020 contrasts failed on the pod: patient groups were saved as an object array (needs pickle)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from test_isic_eval_runner import FakeExtractor, make_ds, run

from dermssl import e3_store
from dermssl.isic_eval import runner as rn
from dermssl.isic_eval import scoring


def _legacy(store, name: str) -> None:
    """Rewrite a cell's probas as the first pod run did (object groups) and re-register its sha256."""
    path = store.root / f"{name}.probas.npz"
    saved = dict(np.load(path))
    saved["groups"] = saved["groups"].astype(object)
    np.savez_compressed(path, **saved)
    doc = store.read(name)
    doc["result"]["probas_sha256"] = rn._sha(path)
    e3_store.write_json_atomic(store.path(name), doc)


def _cells(tmp_path, grouped=True):
    ds = make_ds(grouped)
    if grouped:  # pandas gives object arrays: that is what the pod saved
        ds = dataclasses.replace(ds, groups=ds.groups.astype(object))
    store = e3_store.CellStore(tmp_path)
    for config, pool in (("B71", True), ("B71", False), ("PUBLIC_B71", False)):
        run(store, ds, config, FakeExtractor(ds), pool=pool)
    return ds, store


class TestWriter:
    def test_groups_are_saved_as_strings_loadable_without_pickle(self, tmp_path):
        ds, _ = _cells(tmp_path)
        saved = np.load(tmp_path / "B71__clean.probas.npz", allow_pickle=False)
        assert saved["groups"].dtype.kind == "U" and list(saved["groups"]) == [str(g) for g in ds.val_groups]

    def test_ungrouped_datasets_still_save_an_empty_groups_array(self, tmp_path):
        _, store = _cells(tmp_path, grouped=False)
        assert np.load(tmp_path / "B71__clean.probas.npz", allow_pickle=False)["groups"].size == 0


class TestReader:
    def test_a_legacy_object_file_loads_after_its_sha256_checks_and_groups_become_strings(self, tmp_path):
        _, store = _cells(tmp_path)
        _legacy(store, "B71__clean")
        sha = store.read("B71__clean")["result"]["probas_sha256"]
        loaded = rn.load_probas(tmp_path / "B71__clean.probas.npz", sha)
        assert loaded["groups"].dtype.kind == "U" and loaded["groups"].size > 0

    def test_a_legacy_file_with_another_sha256_is_refused_without_loading_pickle(self, tmp_path, monkeypatch):
        _, store = _cells(tmp_path)
        _legacy(store, "B71__clean")
        real = rn.np.load
        monkeypatch.setattr(rn.np, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pickle")) if k.get("allow_pickle") else real(*a, **k))
        with pytest.raises(rn.ProbasIntegrityError, match="B71__clean"):
            rn.load_probas(tmp_path / "B71__clean.probas.npz", "0" * 64)

    def test_current_files_never_enable_pickle(self, tmp_path, monkeypatch):
        _, store = _cells(tmp_path)
        real = rn.np.load
        monkeypatch.setattr(rn.np, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pickle")) if k.get("allow_pickle") else real(*a, **k))
        sha = store.read("B71__clean")["result"]["probas_sha256"]
        assert rn.load_probas(tmp_path / "B71__clean.probas.npz", sha)["proba"].ndim == 2

    def test_a_corrupt_file_that_is_not_a_legacy_one_still_raises_its_own_error(self, tmp_path):
        (tmp_path / "x.probas.npz").write_bytes(b"not a zip")
        with pytest.raises(Exception):
            rn.load_probas(tmp_path / "x.probas.npz", "0" * 64)


class TestContrasts:
    def test_legacy_files_give_the_same_contrasts_as_current_ones(self, tmp_path):
        _, fresh = _cells(tmp_path / "fresh")
        _, old = _cells(tmp_path / "old")
        for name in ("B71__clean", "B71__original", "PUBLIC_B71"):
            _legacy(old, name)
        docs = []
        for store in (fresh, old):
            cstore = e3_store.CellStore(store.root / "contrasts")
            assert rn.run_contrast(store, cstore, "B71__clean", "PUBLIC_B71", n_resamples=20) == "done"
            docs.append(cstore.read("B71__clean__vs__PUBLIC_B71")["result"])
        assert docs[0] == docs[1] and docs[0]["unit"] == "patient_id"

    def test_the_grouped_contrast_is_the_patient_bootstrap_with_the_same_seed(self, tmp_path):
        ds, store = _cells(tmp_path)
        cstore = e3_store.CellStore(tmp_path / "contrasts")
        rn.run_contrast(store, cstore, "B71__clean", "PUBLIC_B71", n_resamples=20)
        a, c = (np.load(tmp_path / f"{n}.probas.npz") for n in ("B71__clean", "PUBLIC_B71"))
        expected = scoring.contrasts("B71__clean", "PUBLIC_B71", a["y"], a["proba"], c["proba"],
                                     groups=np.asarray(ds.val_groups).astype(str), n_resamples=20)
        assert cstore.read("B71__clean__vs__PUBLIC_B71")["result"]["paired"] == expected

    def test_isic2019_style_ungrouped_contrasts_are_unchanged(self, tmp_path):
        _, store = _cells(tmp_path, grouped=False)
        cstore = e3_store.CellStore(tmp_path / "contrasts")
        rn.run_contrast(store, cstore, "B71__clean", "PUBLIC_B71", n_resamples=20)
        a, c = (np.load(tmp_path / f"{n}.probas.npz") for n in ("B71__clean", "PUBLIC_B71"))
        doc = cstore.read("B71__clean__vs__PUBLIC_B71")["result"]
        assert doc["unit"] == "image"
        assert doc["paired"] == scoring.contrasts("B71__clean", "PUBLIC_B71", a["y"], a["proba"], c["proba"],
                                                  groups=None, n_resamples=20)
