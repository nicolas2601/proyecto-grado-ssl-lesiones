"""S4-UNet training (docs/27): same pool, same exclusions, same protocol as the student.

Tiny fake DINOv2, tiny cache, CPU. What is pinned: the images trained on are
the student's (confident pseudo-pixels only), a killed run resumes to the
exact weights of an uninterrupted one, the provenance names the pool, the
pseudo-mask file and the code, and the timing smoke projects GPU-hours
without writing a checkpoint.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import seg_paths  # noqa: F401
from env_sidecar import sidecar_env  # noqa: E402

torch = pytest.importorskip("torch")
import cache_io  # noqa: E402
from ckpt_io import load_checkpoint  # noqa: E402
from dermssl import torch_s4_unet as s4  # noqa: E402
from s4_fakes import FakeVit  # noqa: E402
from scripts import s4_unet_train as st  # noqa: E402

loop = st.loop  # the module object the script really calls

pytestmark = pytest.mark.torch
SIZE = 56
COUNT = 12


def _pool(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(0)
    images = rng.integers(150, 220, size=(COUNT, SIZE, SIZE, 3), dtype=np.uint8)
    images[:, 14:42, 14:42] = 40
    np.save(cache / "images.npy", images)
    splits = ["val" if i in (3, 7) else "train" for i in range(COUNT)]
    records = [cache_io.ImageRecord("ham", f"ISIC_{i}", f"/x/{i}.jpg") for i in range(COUNT)]
    cache_io.write_manifest(cache / "manifest.csv", records, splits)
    masks = np.zeros((COUNT, SIZE, SIZE), np.uint8)
    masks[:, 14:42, 14:42] = cache_io.FOREGROUND
    masks[5] = cache_io.IGNORE  # excluded by the uncertainty module, like r0's 4,922
    np.save(tmp_path / "pseudo_masks.npy", masks)
    return cache, tmp_path / "pseudo_masks.npy"


@pytest.fixture
def pool(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "load_backbone", lambda arch, source: FakeVit())
    monkeypatch.setattr(st.rp, "clean_revision", lambda repo: "c0ffee")
    return _pool(tmp_path)


def _argv(pool, out, *extra):
    cache, masks = pool
    return ["--cache-dir", str(cache), "--pseudo-masks", str(masks), "--output-dir", str(out),
            "--variant", "A2", "--epochs", "2", "--batch-size", "2", "--features", "16",
            "--precision", "fp32", "--force-cpu", "--log-every", "1000", *extra]


def _decoder(path):
    return load_checkpoint(path)["model"]


def test_registered_defaults_mirror_the_student_r3_round(pool, tmp_path):
    args = st.parse_args(_argv(pool, tmp_path / "o")[:8])
    assert (args.epochs, args.batch_size, args.lr, args.optimizer) == (20, 32, 1e-4, "adam")
    assert (args.warmup_epochs, args.dice_weight, args.selection) == (1, 1.0, "best-val")
    assert (args.precision, args.seed, args.backbone) == ("bf16", 42, ("dinov2_vitb14", "public"))


def test_training_uses_the_student_split_and_its_exclusions(pool, tmp_path):
    out = tmp_path / "run"
    st.main(_argv(pool, out))
    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert record["pool"]["n_images"] == COUNT
    assert record["training"]["n_train_confident"] == COUNT - 3  # 2 val + 1 all-ignore
    assert record["training"]["n_val_confident"] == 2
    assert record["complete"] is True and record["code_revision"] == "c0ffee"
    assert sidecar_env(out / "run.json")["torch"]["loaded"] is True
    assert record["pseudo_masks"]["sha256"] == st.sha256_file(pool[1])
    assert record["seed"] == 42 and record["variant"] == "A2"
    assert record["evaluation_regime"].startswith("transductive without labels")
    assert record["reads_masks_or_labels"] is False
    best = load_checkpoint(out / "best.pt")
    assert best["config"]["arch"] == s4.ARCH and best["config"]["resolution"] == SIZE
    assert not any(key.startswith("backbone") for key in best["model"])


def test_a_killed_run_resumes_to_the_exact_uninterrupted_weights(pool, tmp_path, monkeypatch):
    straight = tmp_path / "straight"
    st.main(_argv(pool, straight))

    killed = tmp_path / "killed"

    def die(epoch):
        if epoch == 0:
            raise KeyboardInterrupt

    monkeypatch.setattr(loop, "after_epoch", die)
    with pytest.raises(KeyboardInterrupt):
        st.main(_argv(pool, killed))
    assert (killed / "last.pt").is_file() and not (killed / "timing.json").exists()
    monkeypatch.setattr(loop, "after_epoch", lambda epoch: None)
    st.main(_argv(pool, killed))

    for name in ("last.pt", "best.pt"):
        a, b = _decoder(straight / name), _decoder(killed / name)
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_resuming_with_other_pseudo_masks_is_refused(pool, tmp_path, monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(loop, "after_epoch", lambda epoch: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        st.main(_argv(pool, out))
    other = np.load(pool[1])
    other[0, 0, 0] = cache_io.IGNORE
    np.save(pool[1], other)
    with pytest.raises(SystemExit, match="pseudo_masks"):
        st.main(_argv(pool, out))


def test_timing_smoke_projects_gpu_hours_and_saves_no_checkpoint(pool, tmp_path, capsys):
    out = tmp_path / "smoke"
    st.main(_argv(pool, out, "--timing-smoke", "3", "--epochs", "20"))
    report = json.loads((out / "timing_smoke.json").read_text(encoding="utf-8"))
    assert report["steps_timed"] == 3 and report["epochs"] == 20
    assert report["steps_per_epoch"] == (COUNT - 3) // 2
    assert report["seconds_per_step"] > 0 and report["projected_gpu_hours"] > 0
    assert "s/step" in capsys.readouterr().out
    assert not (out / "last.pt").exists() and not (out / "best.pt").exists()


def test_mismatched_pseudo_masks_are_refused(pool, tmp_path):
    np.save(pool[1], np.zeros((COUNT, 8, 8), np.uint8))
    with pytest.raises(SystemExit, match="pseudo"):
        st.main(_argv(pool, tmp_path / "bad"))
