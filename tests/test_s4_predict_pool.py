"""A segmenter over the pool cache, for the FST rerun with A2's masks (docs/27).

The published FST manifest came from ``pseudo_r2``: the student's
``predictions.npy`` (float16 sigmoid over the cache) passed through
``make_pseudo_masks.py --source student``. The predictor writes that same
file for any checkpoint ``eval_student.load_student`` accepts, so A2 takes the
student's exact downstream path and only the segmenter changes.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

import seg_paths  # noqa: F401

torch = pytest.importorskip("torch")
import cache_io  # noqa: E402
import eval_student  # noqa: E402
import fst_pseudo_manifest as fpm  # noqa: E402
import make_pseudo_masks as mpm  # noqa: E402
from ckpt_io import save_checkpoint  # noqa: E402
from dermssl import torch_s4_unet as s4  # noqa: E402
from dermssl.pool_manifest import sha256_file  # noqa: E402
from dermssl.torch_unet import UNet  # noqa: E402
from s4_fakes import FakeVit  # noqa: E402
from scripts import s4_predict_pool as pp  # noqa: E402

pytestmark = pytest.mark.torch
SIZE = 56
COUNT = 5


def _cache(tmp_path, size=SIZE):
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    images = rng.integers(150, 220, (COUNT, size, size, 3), dtype=np.uint8)
    images[:, 14:42, 14:42] = 40
    np.save(cache / "images.npy", images)
    records = [cache_io.ImageRecord("ham", f"ISIC_{i}", f"/x/{i}.jpg") for i in range(COUNT)]
    cache_io.write_manifest(cache / "manifest.csv", records, ["train"] * (COUNT - 1) + ["val"])
    return cache


def _s4_checkpoint(tmp_path, resolution=SIZE):
    torch.manual_seed(0)
    model = s4.S4UNet(FakeVit(), features=16)
    config = s4.checkpoint_config(model, backbone_arch="dinov2_vitb14", backbone_source="public",
                                  backbone_sha256="public", resolution=resolution)
    path = tmp_path / "best.pt"
    save_checkpoint(path, {"model": s4.decoder_state(model), "config": config, "epoch": 0})
    return path


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setattr(s4, "hub_backbone", lambda arch, source: FakeVit())
    monkeypatch.setattr(pp.rp, "clean_revision", lambda repo: "c0ffee")


def _run(tmp_path, checkpoint, *extra, name="a2_pool"):
    out = tmp_path / name
    pp.main(["--checkpoint", str(checkpoint), "--cache-dir", str(tmp_path / "cache"),
             "--output-dir", str(out), "--device", "cpu", "--batch-size", "2", *extra])
    return out


def _expected(checkpoint, cache):
    model, _ = eval_student.load_student(checkpoint, torch.device("cpu"))
    images = np.load(cache / "images.npy")
    return eval_student.predict_masks(model, images, torch.device("cpu"), COUNT)


def test_writes_the_student_predictions_layout_with_the_evaluator_probabilities(tmp_path):
    cache, checkpoint = _cache(tmp_path), _s4_checkpoint(tmp_path)
    out = _run(tmp_path, checkpoint)
    predictions = np.load(out / pp.PREDICTIONS_FILE)
    assert predictions.dtype == np.float16 and predictions.shape == (COUNT, SIZE, SIZE)
    np.testing.assert_allclose(predictions, _expected(checkpoint, cache), atol=1e-3)


def test_provenance_pins_checkpoint_pool_revision_and_output(tmp_path):
    cache, checkpoint = _cache(tmp_path), _s4_checkpoint(tmp_path)
    out = _run(tmp_path, checkpoint)
    run = json.loads((out / pp.RUN_FILE).read_text(encoding="utf-8"))
    assert run["checkpoint"]["sha256"] == sha256_file(checkpoint)
    assert run["checkpoint"]["arch"] == s4.ARCH
    assert run["pool"]["manifest_sha256"] == sha256_file(cache / "manifest.csv")
    assert run["pool"]["n_images"] == COUNT and run["n"] == COUNT
    assert run["code_revision"] == "c0ffee"
    assert run["outputs"][pp.PREDICTIONS_FILE] == sha256_file(out / pp.PREDICTIONS_FILE)
    assert run["inference"] == {"path": pp.INFERENCE_PATH, "tta": False, "precision": "fp32",
                                "resolution": SIZE}
    assert run["complete"] is True and run["partial"] is False
    assert run["reads_masks_or_labels"] is False


def test_the_predictions_take_the_student_path_into_the_fst_manifest(tmp_path):
    cache, checkpoint = _cache(tmp_path), _s4_checkpoint(tmp_path)
    out = _run(tmp_path, checkpoint)
    pseudo = tmp_path / "pseudo_a2"
    mpm.run(mpm.parse_args(["--saliency", str(out / pp.PREDICTIONS_FILE), "--source", "student",
                            "--output-dir", str(pseudo), "--min-area-fraction", "0.003"]))
    manifest = fpm.run(fpm.parse_args([
        "--manifest", str(cache / "manifest.csv"),
        "--pseudo-masks", str(pseudo / "pseudo_masks.npy"),
        "--images", str(cache / "images.npy"),
        "--output-json", str(tmp_path / "fst.json")]))
    assert manifest["summary"]["total"] == COUNT


def test_threshold_masks_are_the_evaluator_binarisation_and_carry_their_source(tmp_path):
    cache, checkpoint = _cache(tmp_path), _s4_checkpoint(tmp_path)
    out = _run(tmp_path, checkpoint, "--threshold-masks", "--mask-source", "s4_unet_a2")
    masks = np.load(out / pp.THRESHOLD_MASKS_FILE)
    assert masks.dtype == np.uint8 and masks.shape == (COUNT, SIZE, SIZE)
    np.testing.assert_array_equal(masks, (_expected(checkpoint, cache) > 0.5).astype(np.uint8))
    stats = json.loads((out / pp.THRESHOLD_STATS_FILE).read_text(encoding="utf-8"))
    assert stats["mask_source"] == "s4_unet_a2" and stats["threshold"] == 0.5
    manifest = fpm.run(fpm.parse_args([
        "--manifest", str(cache / "manifest.csv"),
        "--pseudo-masks", str(out / pp.THRESHOLD_MASKS_FILE),
        "--stats-json", str(out / pp.THRESHOLD_STATS_FILE),
        "--output-json", str(tmp_path / "fst.json"),
        "--images", str(cache / "images.npy")]))
    assert {r["mask_source"] for r in manifest["records"]} == {"s4_unet_a2"}


def test_a_student_unet_checkpoint_runs_through_the_same_path(tmp_path):
    cache = _cache(tmp_path, size=64)  # the U-Net halves four times
    path = tmp_path / "student.pt"
    torch.manual_seed(0)
    save_checkpoint(path, {"model": UNet(base_channels=8).state_dict(),
                           "config": {"base_channels": 8, "resolution": 64}})
    out = _run(tmp_path, path)
    np.testing.assert_allclose(np.load(out / pp.PREDICTIONS_FILE), _expected(path, cache),
                               atol=1e-3)
    run = json.loads((out / pp.RUN_FILE).read_text(encoding="utf-8"))
    assert run["checkpoint"]["arch"] == pp.STUDENT_ARCH


def test_limit_marks_a_partial_smoke_artifact(tmp_path):
    _cache(tmp_path)
    out = _run(tmp_path, _s4_checkpoint(tmp_path), "--limit", "3")
    assert np.load(out / pp.PREDICTIONS_FILE).shape[0] == 3
    run = json.loads((out / pp.RUN_FILE).read_text(encoding="utf-8"))
    assert run["partial"] is True and run["n"] == 3


def test_a_resolution_other_than_the_cache_is_refused(tmp_path):
    _cache(tmp_path)
    checkpoint = _s4_checkpoint(tmp_path, resolution=SIZE * 2)
    with pytest.raises(SystemExit, match="resolution"):
        _run(tmp_path, checkpoint)
    assert not (tmp_path / "a2_pool" / pp.PREDICTIONS_FILE).exists()


def test_an_existing_output_is_never_overwritten(tmp_path):
    _cache(tmp_path)
    (tmp_path / "a2_pool").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        _run(tmp_path, _s4_checkpoint(tmp_path))


def test_a_dirty_tree_is_refused_before_any_inference(tmp_path, monkeypatch):
    _cache(tmp_path)

    def dirty(repo):
        raise pp.rp.DirtyTreeError("dirty")

    monkeypatch.setattr(pp.rp, "clean_revision", dirty)
    with pytest.raises(SystemExit, match="dirty"):
        _run(tmp_path, _s4_checkpoint(tmp_path))
    assert not (tmp_path / "a2_pool").exists()


def test_the_mask_source_label_is_required_with_threshold_masks():
    with pytest.raises(SystemExit):
        pp.parse_args(["--checkpoint", "x", "--cache-dir", "c", "--output-dir", "o",
                       "--threshold-masks"])
    assert isinstance(pp.parse_args(["--checkpoint", "x", "--cache-dir", "c",
                                     "--output-dir", "o"]), argparse.Namespace)
