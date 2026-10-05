"""Rows of the bench tables: sources, sorting, pending rows and the numbers they carry."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from dermssl.bench import rows as R

REPO = R.REPO
ALL = {key: R.dataset_rows(REPO, ds) for key, ds in R.DATASETS.items()}


@pytest.mark.parametrize("key", list(ALL))
def test_every_row_has_an_existing_source_path(key):
    for row in ALL[key]:
        assert row.sources, row.name
        assert all((REPO / s).is_file() for s in row.sources), (row.name, row.sources)


@pytest.mark.parametrize("key", list(ALL))
def test_rows_are_sorted_by_macro_f1_inside_each_block_and_pending_rows_come_last(key):
    for block in R.BLOCKS:
        mine = [r for r in ALL[key] if r.block == block]
        statuses = [r.status == R.OK for r in mine]
        assert statuses == sorted(statuses, reverse=True), "pending rows must follow the measured ones"
        scores = [r.values["macro_f1"].point for r in mine if r.status == R.OK]
        assert scores == sorted(scores, reverse=True)


def test_counts_per_dataset():
    assert len(ALL["ham10000"]) == 8 + 8 + 5 + 1 + 2
    assert len(ALL["isic2019"]) == 8 + 8 + 6 + 4
    assert len(ALL["isic2020"]) == 8 + 6 + 4 + 8


def test_isic2020_finetune_block_is_an_explicit_not_run_row_and_originals_are_transductive():
    ft = [r for r in ALL["isic2020"] if r.block == R.FINETUNE]
    assert len(ft) == 8 and all(r.status in (R.OK, R.PENDING) for r in ft)
    originals = [r for r in ALL["isic2019"] if "original (canónico)" in r.name]
    assert len(originals) == 2 and all(r.transductive for r in originals)
    assert all(r.status == R.OK or "ETA ~15:00 UTC" in r.note for r in originals)
    assert not any(r.transductive for r in ALL["ham10000"])


def test_pending_rows_carry_no_numbers(tmp_path):
    """Nothing is pending in the committed repository; the property is checked on a copy without the HAM clean result."""
    import shutil

    from dermssl import ham_clean as hc

    for folder in ("resultados", "results"):
        shutil.copytree(REPO / folder, tmp_path / folder)
    shutil.rmtree(tmp_path / hc.DIR)
    pending = [r for ds in R.DATASETS.values() for r in R.dataset_rows(tmp_path, ds) if r.status != R.OK]
    assert pending and all(not r.values for r in pending)
    assert not [r for rs in ALL.values() for r in rs if r.status != R.OK]  # and none is left in the repository itself


def test_measured_rows_have_every_metric_and_the_source_ci():
    for rs in ALL.values():
        for r in (r for r in rs if r.status == R.OK):
            assert set(r.values) == set(R.METRICS)
            assert r.values["macro_f1"].ci
            if r.block in (R.FROZEN, R.FINETUNE) and r.sources[0].startswith(("results/", "resultados/clasificacion")):
                assert r.values["mcc"].ci is None


def test_baseline_value_is_the_json_logreg_number():
    row = next(r for r in ALL["ham10000"] if r.name == "DINOv2 ViT-S/14" and not r.finetune)
    doc = json.loads((REPO / row.sources[0]).read_text(encoding="utf-8"))["probes"]["logreg"]
    assert row.values["macro_f1"].point == doc["macro_f1"]
    assert row.values["auc"].point == doc["auc_ovr_macro"]
    assert row.values["macro_f1"].ci == tuple(doc["ic95_bootstrap"]["macro_f1"])


def test_v71_derived_numbers_reproduce_the_stored_ones_and_the_auc():
    npz = np.load(REPO / R.V71_NPZ)
    for key, _, _ in R.V71_ROWS:
        cm, values = R.v71_derived(REPO, key)
        proba, y = npz[key], npz["y_true"]
        pred = proba.argmax(axis=1)
        assert f1_score(y, pred, average="macro") == pytest.approx(values["macro_f1"].point, abs=1e-12)
        assert accuracy_score(y, pred) == pytest.approx(values["accuracy"].point, abs=1e-12)
        assert balanced_accuracy_score(y, pred) == pytest.approx(values["macro_recall"].point, abs=1e-12)
        assert roc_auc_score(y, proba, multi_class="ovr", average="macro") == pytest.approx(
            values["auc"].point, abs=1e-9)
        assert sum(map(sum, cm)) == 2004


def test_truncated_finetune_runs_are_flagged_from_the_json():
    row = next(r for r in ALL["isic2019"] if r.name == "DINOv2 ViT-B/14" and r.finetune)
    assert "sesgo optimista" in row.note and "sin converger" in row.note
