"""ISIC rows of the bench: canonical vs sensitivity blocks, pending cells that heal, numbers from the JSON."""

from __future__ import annotations

import json
import shutil

import pytest

from dermssl.bench import isic_rows as I
from dermssl.bench import rows as R

REPO = R.REPO


def _by_name(key: str) -> dict[str, R.Row]:
    return {r.name: r for r in R.dataset_rows(REPO, R.DATASETS[key])}


@pytest.mark.parametrize("key", ["isic2019", "isic2020"])
def test_canonical_rows_are_measured_in_the_frozen_block_and_carry_the_compute_label(key):
    rows = _by_name(key)
    for name in ("B71-limpio (canónico)", "DINOv2 público, protocolo de B71 (canónico)",
                 "DINOv2 público, protocolo de A71 (canónico)"):
        row = rows[name]
        assert row.block == R.FROZEN and row.status == R.OK and not row.transductive
        assert "batch 64 + autocast" in row.note and len(row.sources) == 2


@pytest.mark.parametrize("key", ["isic2019", "isic2020"])
def test_sensitivity_rows_are_labelled_non_canonical_and_originals_transductive(key):
    rows = [r for r in R.dataset_rows(REPO, R.DATASETS[key]) if r.block == R.SENSITIVITY]
    assert len(rows) == 4 and all("no canónico" in r.note and "batch 16" in r.name for r in rows)
    assert {r.name: r.transductive for r in rows} == {
        "B71 original (batch 16)": True, "A71 original (batch 16)": True,
        "DINOv2 público, protocolo de B71 (batch 16)": False, "DINOv2 público, protocolo de A71 (batch 16)": False}


def test_row_numbers_are_the_json_numbers_including_the_mcc_interval():
    row = _by_name("isic2020")["B71-limpio (canónico)"]
    metrics = json.loads((REPO / row.sources[0]).read_text(encoding="utf-8"))["result"]["metrics"]
    assert row.values["macro_f1"].point == metrics["macro_f1"]
    assert row.values["mcc"].ci == tuple(metrics["ci95"]["mcc"])
    assert row.values["macro_precision"].ci is None and row.n_val == 6660


@pytest.mark.parametrize("key", ["isic2019", "isic2020"])
def test_pending_rows_say_why_and_when_and_have_no_numbers(key):
    pend = [r for r in R.dataset_rows(REPO, R.DATASETS[key]) if r.status != R.OK]
    names = {r.name for r in pend}
    assert {"A71-limpio (canónico)", "B71 original (canónico)", "A71 original (canónico)"} <= names
    assert all(not r.values and "ETA" in r.note for r in pend)


def test_isic2020_finetune_pending_row_names_the_gpu_and_e2():
    row = _by_name("isic2020")["8 baselines con fine-tuning"]
    assert row.status == R.NOT_RUN and "GPU1" in row.note and "E2" in row.note and "optimista" in row.note


def test_a_pending_cell_becomes_a_measured_row_when_its_json_appears(tmp_path):
    src = REPO / I.CANONICAL_DIR / "isic2019"
    dst = tmp_path / I.CANONICAL_DIR / "isic2019"
    dst.mkdir(parents=True)
    for suffix in (".json", ".probas.npz"):
        shutil.copy(src / f"B71__clean{suffix}", dst / f"A71__clean{suffix}")
    shutil.copy(src / "PUBLIC_B71.json", dst / "PUBLIC_B71.json")
    shutil.copy(src / "PUBLIC_A71.json", dst / "PUBLIC_A71.json")
    shutil.copy(src / "B71__clean.json", dst / "B71__clean.json")
    rows = {r.name: r for r in I.canonical_rows(tmp_path, "isic2019")}
    assert rows["A71-limpio (canónico)"].status == R.OK
    assert rows["B71 original (canónico)"].status == R.PENDING
