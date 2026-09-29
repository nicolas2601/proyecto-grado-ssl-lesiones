"""The T3 script on synthetic inputs: integrity checks, document and generated table (docs/28, T3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dermssl import c1_luminance as cl
from dermssl import metrics as _metrics
from dermssl.c1_split_identity import ids_sha256
from dermssl.feature_cache import _digest, fingerprint_ids
from scripts import t3_c1_luminance as t3

N_VAL, N_TRAIN, N_CLASSES = 240, 50, 7
TERCILES = ["dark"] * 150 + ["medium"] * 60 + ["light"] * 20 + ["unavailable"] * 10


def _proba(y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pred = np.where(rng.rand(len(y)) < 0.7, y, rng.choice([0, 4, 5], size=len(y)))
    proba = rng.rand(len(y), N_CLASSES) * 0.1
    proba[np.arange(len(y)), pred] += 1.0
    return proba / proba.sum(axis=1, keepdims=True)


def _write_cell(grid: Path, backbone: str, condition: str, proba: np.ndarray, prov: dict) -> None:
    stem = grid / f"cell__{backbone}__{condition}"
    np.save(f"{stem}.proba.npy", proba)
    manifest = {"backbone": backbone, "condition": condition, "proba_sha256": _digest(proba), **prov}
    Path(f"{stem}.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, dict, Path, str]:
    root = tmp_path / "t3"
    val_ids = [f"ISIC_{i:07d}" for i in range(N_VAL)]
    train_ids = [f"ISIC_{i:07d}" for i in range(N_VAL, N_VAL + N_TRAIN)]
    rng = np.random.RandomState(0)
    y_val, y_train = rng.choice([0, 4, 5], size=N_VAL), rng.choice([0, 4, 5], size=N_TRAIN)
    (root / "labels").mkdir(parents=True)
    (root / "c1_split_index.json").write_text(json.dumps({"train_ids": train_ids, "val_ids": val_ids}))
    for split, ids, y in (("train", train_ids, y_train), ("val", val_ids, y_val)):
        np.save(root / "labels" / f"v1__dinov2_public_vitb14__{split}__r224__eval__{fingerprint_ids(ids)}.labels.npy", y)
    prov = {"c1_val_ids_sha256": ids_sha256(val_ids), "labels_sha256": _digest(np.concatenate([y_train, y_val]))}
    published = {}
    for s_index, segmenter in enumerate(cl.SEGMENTERS):
        grid = root / segmenter
        grid.mkdir()
        cells = {}
        for b_index, backbone in enumerate(t3.BACKBONES):
            for c_index, condition in enumerate(t3.ALL_CONDITIONS):
                proba = _proba(y_val, 100 * s_index + 10 * b_index + c_index)
                _write_cell(grid, backbone, condition, proba, prov)
                cells[f"{backbone}__{condition}"] = {"metrics": _metrics.summarize(y_val, proba, N_CLASSES)}
        published[segmenter] = tmp_path / f"{segmenter}_c1_cells.json"
        published[segmenter].write_text(json.dumps(cells))
    terciles_csv = tmp_path / "terciles.csv"
    t3.write_terciles(terciles_csv, dict(zip(val_ids, TERCILES)))
    return root, published, terciles_csv, ids_sha256(val_ids)


def test_the_document_holds_every_cell_the_key_cells_and_the_prediction(tmp_path):
    root, published, terciles, val_sha = _fixture(tmp_path)
    doc = t3.build_document(root, terciles, n_resamples=50, seed=20260923, published=published,
                            val_ids_sha256=val_sha)
    assert doc["coverage"] == {"dark": 150, "medium": 60, "light": 20, "unavailable": 10,
                               "not_in_manifest": 0, "total": 240}
    cell = doc["segmenters"]["a2"]["v7_astar_s42"]
    assert cell["medium"]["contrasts"]["mask_pool"]["macro_f1"]["verdict"] == cl.NO_POWER
    assert "ci95_low" in cell["dark"]["contrasts"]["bbox_crop"]["macro_f1"]
    assert len(doc["key_cells"]) == 12 and doc["prediction"]["n_key_cells"] == 12
    assert doc["multiplicity"] == {"n_contrasts": 270, "expected_ci_without_zero_by_chance": 13.5,
                                   "n_key_contrasts": 12, "expected_key_improves_by_chance": 0.3}


def test_a_proba_that_does_not_match_its_manifest_is_refused(tmp_path):
    root, published, terciles, val_sha = _fixture(tmp_path)
    path = root / "a2" / "cell__v7_bstar_s43__neutral_bg.proba.npy"
    np.save(path, np.load(path)[::-1])
    with pytest.raises(ValueError, match="proba_sha256"):
        t3.build_document(root, terciles, n_resamples=10, seed=0, published=published, val_ids_sha256=val_sha)


def test_a_val_order_other_than_the_registered_one_is_refused(tmp_path):
    root, published, terciles, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="orden de val"):
        t3.build_document(root, terciles, n_resamples=10, seed=0, published=published,
                          val_ids_sha256="0" * 64)


def test_labels_from_another_cache_are_refused(tmp_path):
    root, published, terciles, val_sha = _fixture(tmp_path)
    (label,) = (root / "labels").glob("*__val__*.labels.npy")
    np.save(label, np.load(label)[::-1])
    with pytest.raises(ValueError, match="labels_sha256"):
        t3.build_document(root, terciles, n_resamples=10, seed=0, published=published, val_ids_sha256=val_sha)


def test_predictions_that_do_not_reproduce_the_published_grid_are_refused(tmp_path):
    root, published, terciles, val_sha = _fixture(tmp_path)
    cells = json.loads(published["student_r3"].read_text())
    cells["dinov2_public_vitb14__none"]["metrics"]["macro_f1"] += 1e-6
    published["student_r3"].write_text(json.dumps(cells))
    with pytest.raises(ValueError, match="no reproduce"):
        t3.build_document(root, terciles, n_resamples=10, seed=0, published=published, val_ids_sha256=val_sha)


def test_the_table_is_generated_from_the_document(tmp_path):
    root, published, terciles, val_sha = _fixture(tmp_path)
    doc = t3.build_document(root, terciles, n_resamples=30, seed=1, published=published, val_ids_sha256=val_sha)
    table = t3.render_markdown(doc)
    assert "no editar a mano" in table and "no es tono de piel" in table
    assert f"**{doc['prediction']['verdict']}**" in table
    assert table.count("| oscuro |") >= 12 and "sin potencia" in table
    assert "270" in table
