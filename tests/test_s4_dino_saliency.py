"""S4-UNet A1 (docs/27): S-diag saliency over the training pool.

The registration: the training-free S-diag map, NOT binarised, enters the
same uncertainty module as CCAM did. So the builder writes a CCAM-shaped
``saliency.npy`` with the S-diag PCA map (imported from ``dermssl.s_diag``),
and ``make_pseudo_masks.py --source dino`` turns it into pseudo-masks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import seg_paths  # noqa: F401
from env_sidecar import sidecar_env  # noqa: E402

torch = pytest.importorskip("torch")
import cache_io  # noqa: E402
import make_pseudo_masks as mpm  # noqa: E402
from dermssl import s_diag as sd  # noqa: E402
from s4_fakes import FakeVit  # noqa: E402
from scripts import s4_dino_saliency as ds  # noqa: E402

pytestmark = pytest.mark.torch
SIZE = 56  # 4x4 patches of 14 px
COUNT = 5


def _cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    images = np.full((COUNT, SIZE, SIZE, 3), 180, np.uint8)
    images[:, 14:42, 14:42] = 40  # dark lesion on the central 2x2 patches
    np.save(cache / "images.npy", images)
    records = [cache_io.ImageRecord("ham", f"ISIC_{i}", f"/x/{i}.jpg") for i in range(COUNT)]
    cache_io.write_manifest(cache / "manifest.csv", records, ["train"] * (COUNT - 1) + ["val"])
    return cache


@pytest.fixture
def fake_backbone(monkeypatch):
    calls = []

    def load(arch, source):
        calls.append((arch, source))
        return FakeVit().eval()

    monkeypatch.setattr(ds, "_load_backbone", load)
    monkeypatch.setattr(ds, "_BACKBONES", {})
    monkeypatch.setattr(ds.rp, "clean_revision", lambda repo: "c0ffee")
    return calls


def _run(tmp_path, *extra, name="a1"):
    out = tmp_path / name
    ds.main(["--cache-dir", str(_cache(tmp_path)), "--output-dir", str(out), *extra])
    return out


def test_saliency_is_the_s_diag_pca_map_repeated_to_the_pixel_grid():
    tokens = np.random.default_rng(0).normal(size=(16, 5))
    saliency = ds.saliency_map(tokens, grid=4, resolution=SIZE)
    pca = sd.pca_first_component_map(tokens, 4)
    assert saliency.shape == (SIZE, SIZE)
    binary = saliency > 0.1
    assert np.array_equal(binary, sd.upsample_nearest(pca > 0.1, (SIZE, SIZE)))


def test_a_resolution_that_is_not_a_whole_number_of_patches_is_refused():
    with pytest.raises(ValueError):
        ds.saliency_map(np.zeros((16, 3)), grid=4, resolution=57)


def test_shards_cover_every_image_once_in_order():
    shards = ds.shards(10, 3)
    covered = [i for start, stop in shards for i in range(start, stop)]
    assert covered == list(range(10)) and len(shards) == 3
    assert ds.shards(2, 8) == [(0, 1), (1, 2)]


def test_builder_writes_a_ccam_shaped_float16_saliency_and_provenance(tmp_path, fake_backbone):
    out = _run(tmp_path, "--workers", "1", "--batch-size", "2")
    saliency = np.load(out / "saliency.npy")
    assert saliency.shape == (COUNT, SIZE, SIZE) and saliency.dtype == np.float16
    assert np.all(saliency[:, 28, 28] > saliency[:, 2, 2])  # S-diag sign: centre positive
    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert record["code_revision"] == "c0ffee"
    assert record["pool"]["n_images"] == COUNT
    assert record["pool"]["manifest_sha256"] == ds.sha256_file(tmp_path / "cache" / "manifest.csv")
    assert record["rule"].startswith("S-diag primary")
    assert record["reads_masks_or_labels"] is False
    assert sidecar_env(out / "run.json")["schema"] == 1
    assert fake_backbone == [("dinov2_vitb14", "public")]


def test_two_shards_write_exactly_what_one_pass_writes(tmp_path, fake_backbone):
    whole = np.load(_run(tmp_path, "--workers", "1", name="whole") / "saliency.npy")
    cache = tmp_path / "cache"
    target = tmp_path / "split.npy"
    cache_io.create_cache(target, whole.shape, np.float16).flush()
    for start, stop in ds.shards(COUNT, 2):
        ds.process_shard((str(cache / "images.npy"), str(target), start, stop,
                          "dinov2_vitb14", "public", 2))
    assert np.array_equal(np.load(target), whole)


def test_the_uncertainty_module_turns_it_into_student_pseudo_masks(tmp_path, fake_backbone):
    out = _run(tmp_path, "--workers", "1")
    stats = mpm.run(mpm.parse_args(["--saliency", str(out / "saliency.npy"), "--source", "dino",
                                    "--output-dir", str(tmp_path / "pseudo")]))
    masks = cache_io.open_cache(tmp_path / "pseudo" / "pseudo_masks.npy")
    assert masks.shape == (COUNT, SIZE, SIZE) and masks.dtype == np.uint8
    assert np.all(masks[:, 28, 28] == cache_io.FOREGROUND)
    assert np.all(masks[:, 2, 2] == cache_io.BACKGROUND)
    assert stats["excluded"] == 0


def test_timing_projection_reports_hours_and_writes_no_saliency(tmp_path, fake_backbone, capsys):
    out = _run(tmp_path, "--workers", "1", "--timing-projection", "3")
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["images_timed"] == 3 and report["pool_images"] == COUNT
    assert report["projected_hours"] >= 0 and report["workers"] == 1
    assert not (out / "saliency.npy").exists()


def test_an_existing_output_is_never_overwritten(tmp_path, fake_backbone):
    (tmp_path / "a1").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        _run(tmp_path, "--workers", "1")
