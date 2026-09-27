"""A run killed and resumed must land on the same weights as one that never stopped.

The finals run for 4-7 hours on a pod that restarts. Until 2026-09-27 a
resume restarted the epoch from its first batch and redrew the shuffle, so
the resumed run trained on a different sequence of batches than the one it
replaced. Checkpoints are now written every N steps, atomically, with the
epoch position and the random state, and a resume replays exactly.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402

from dermssl.pretrain import config as config_mod  # noqa: E402
from dermssl.pretrain import torch_resume as rs  # noqa: E402
from dermssl.pretrain import torch_loop  # noqa: E402


class TestSampler:
    def test_same_epoch_same_order_and_epochs_differ(self):
        a, b = rs.ResumableRandomSampler(10, seed=3), rs.ResumableRandomSampler(10, seed=3)
        a.set_epoch(1), b.set_epoch(1)
        assert list(a) == list(b)
        b.set_epoch(2)
        assert list(a) != list(b)

    def test_start_skips_the_consumed_prefix_once(self):
        s = rs.ResumableRandomSampler(10, seed=3)
        s.set_epoch(0)
        full = list(s)
        s.start_at(4)
        assert list(s) == full[4:]
        assert list(s) == full  # the offset applies to one pass only


def test_random_state_roundtrip():
    state = rs.capture_rng()
    first = (torch.rand(3), np.random.rand(3))
    rs.restore_rng(state)
    second = (torch.rand(3), np.random.rand(3))
    assert torch.equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_atomic_save_leaves_no_partial_file(tmp_path):
    target = tmp_path / "resume.pt"
    rs.atomic_save({"a": torch.ones(2)}, target)
    assert torch.load(target, weights_only=True)["a"].sum() == 2
    assert [p.name for p in tmp_path.iterdir()] == ["resume.pt"]


def _images(root, n=6):
    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    for i in range(n):
        Image.fromarray((rng.random((40, 40, 3)) * 255).astype("uint8")).save(root / f"img{i}.png")


def _cfg(image_root, out, **extra):
    base = dict(
        method="simsiam", image_root=str(image_root), output_dir=str(out), split_hash="a" * 64,
        batch_size_per_gpu=2, num_workers=0, resolution=32, precision="fp32", max_steps=20,
        checkpoint_every_steps=10, seed=7,
    )
    base.update(extra)
    return config_mod.PretrainConfig(**base)


def test_kill_at_step_10_and_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    images = tmp_path / "pool"
    _images(images)
    kept = tmp_path / "resume_at_10.pt"
    real_save = rs.atomic_save

    def spy(obj, path):
        real_save(obj, path)
        if obj.get("step") == 10:
            shutil.copy(path, kept)

    monkeypatch.setattr(rs, "atomic_save", spy)
    straight = tmp_path / "straight"
    torch_loop.train(_cfg(images, straight))
    assert kept.is_file()

    resumed = tmp_path / "resumed"
    resumed.mkdir()
    shutil.copy(kept, resumed / "resume.pt")
    torch_loop.train(_cfg(images, resumed))

    a = torch.load(straight / "simsiam_backbone_final.pt", weights_only=True)
    b = torch.load(resumed / "simsiam_backbone_final.pt", weights_only=True)
    for key in a:
        torch.testing.assert_close(b[key], a[key], rtol=1e-5, atol=1e-6)
    run_a = json.loads((straight / "run.json").read_text())["execution"]
    run_b = json.loads((resumed / "run.json").read_text())["execution"]
    assert run_b["steps_executed"] == run_a["steps_executed"] == 20
    assert run_b["ssl_loss"] == pytest.approx(run_a["ssl_loss"], rel=1e-5)
