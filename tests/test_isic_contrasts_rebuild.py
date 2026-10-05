"""scripts/isic_contrasts_rebuild.py regenerates only the paired contrasts, from cells already on disk."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from test_isic_eval_runner import FakeExtractor, make_ds, run

from dermssl import e3_store, run_environment
from dermssl.isic_eval import runner as rn
from scripts import isic_contrasts_rebuild as rebuild


def _dataset_dir(root, name="isic2020", grouped=True):
    ds = make_ds(grouped)
    if grouped:
        ds = dataclasses.replace(ds, groups=ds.groups.astype(object))
    store = e3_store.CellStore(root / name)
    for config, pool in (("B71", True), ("A71", True), ("PUBLIC_B71", False), ("PUBLIC_A71", False)):
        run(store, ds, config, FakeExtractor(ds), pool=pool)
    return root / name, store


def _legacy(store, name):
    path = store.root / f"{name}.probas.npz"
    saved = dict(np.load(path))
    saved["groups"] = saved["groups"].astype(object)
    np.savez_compressed(path, **saved)
    doc = store.read(name)
    doc["result"]["probas_sha256"] = rn._sha(path)
    e3_store.write_json_atomic(store.path(name), doc)


def _main(*dirs, extra=()):
    argv = [a for d in dirs for a in ("--dataset-dir", str(d))] + ["--ci-resamples", "20", *extra]
    return rebuild.main(argv)


def test_it_builds_the_contrasts_of_legacy_cells_without_extracting(tmp_path):
    d, store = _dataset_dir(tmp_path)
    for name in ("B71__clean", "A71__clean", "PUBLIC_B71", "PUBLIC_A71"):
        _legacy(store, name)
    assert _main(d) == 0
    names = {p.name for p in (d / "contrasts").glob("*.json") if not p.name.endswith(".environment.json")}
    assert {"B71__clean__vs__PUBLIC_B71.json", "A71__clean__vs__PUBLIC_A71.json",
            "B71__clean__vs__A71__clean.json"} <= names
    doc = json.loads((d / "contrasts" / "B71__clean__vs__PUBLIC_B71.json").read_text())["result"]
    assert doc["unit"] == "patient_id" and "mcc" in {r["metric"] for r in doc["paired"]}
    assert run_environment.sidecar_path(d / "contrasts" / "B71__clean__vs__PUBLIC_B71.json").is_file()


def test_a_relaunch_skips_what_is_done(tmp_path, capsys):
    d, _ = _dataset_dir(tmp_path)
    _main(d)
    capsys.readouterr()
    assert _main(d) == 0
    assert "skipped" in capsys.readouterr().out


def test_several_datasets_in_one_call(tmp_path):
    d19, _ = _dataset_dir(tmp_path, "isic2019", grouped=False)
    d20, _ = _dataset_dir(tmp_path, "isic2020")
    assert _main(d19, d20) == 0
    assert (d19 / "contrasts").is_dir() and (d20 / "contrasts").is_dir()
    unit = lambda d: json.loads(next((d / "contrasts").glob("B71__clean__vs__PUBLIC_B71.json")).read_text())["result"]["unit"]  # noqa: E731
    assert (unit(d19), unit(d20)) == ("image", "patient_id")


def test_a_corrupt_legacy_cell_fails_that_contrast_and_lets_the_rest_go_on(tmp_path, capsys):
    d, store = _dataset_dir(tmp_path)
    _legacy(store, "PUBLIC_B71")
    doc = store.read("PUBLIC_B71")
    doc["result"]["probas_sha256"] = "0" * 64
    e3_store.write_json_atomic(store.path("PUBLIC_B71"), doc)
    assert _main(d) == 1
    out = capsys.readouterr().out
    assert "FAILED B71__clean__vs__PUBLIC_B71" in out and "sha256" in out
    assert (d / "contrasts" / "A71__clean__vs__PUBLIC_A71.json").is_file()


def test_contrasts_written_before_mcc_are_regenerated_with_it(tmp_path):
    d, _ = _dataset_dir(tmp_path)
    _main(d)
    path = d / "contrasts" / "B71__clean__vs__PUBLIC_B71.json"
    old = json.loads(path.read_text())
    old["key"].pop("metrics")
    old["key_sha256"] = e3_store.canonical_sha256(old["key"])
    e3_store.write_json_atomic(path, old)
    _main(d)
    assert "metrics" in json.loads(path.read_text())["key"] and list(d.glob("contrasts/*.stale-*"))


def test_a_missing_directory_aborts(tmp_path, capsys):
    assert _main(tmp_path / "nope") == 1
    assert "ABORT" in capsys.readouterr().err
