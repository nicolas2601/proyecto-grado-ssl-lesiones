"""HEAD trains A* and B* exactly as f1c2053, the revision of the s42/s43 finals.

The seed-44 replica runs from a later revision (the launch guards live there)
while another change splits ``torch_loop``. This test pins the training path:
the full ``torch_loop.train`` with the finals' levers (frozen blocks, layer
decay, L2-SP, head warm-up, and for B the paired tone view, the tone term and
gradient checkpointing, bf16 autocast), on CPU with a tiny ViT in place of
the hub model. f1c2053's ``src`` is extracted with ``git archive`` and both
revisions train in child processes on THIS machine, under PINNED_ENV; the two
results must be equal bit for bit. The only thing patched for f1c2053 is
``Path.rglob``, whose ``recurse_symlinks`` argument it needs Python 3.13 for
(no symlinks here, same file list).

Why not a stored golden: it was one, and CI failed on it from the day the
test landed. With one or two torch threads the same code gave 3.5107e-05 for
blocks.0.norm1.weight instead of the stored 3.5882e-05 (six threads); pinning
one thread, scalar ATen kernels, MKL_CBWR and no oneDNN made it reproducible
here, yet CI runners still differed by ~2 % run to run of the matrix,
because a bf16 training through six steps turns each CPU's last-bit rounding
into a visible drift. Comparing against f1c2053 on the same CPU removes the
CPU from the question and asks exactly what the test is for.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
nn = torch.nn

ROOT = Path(__file__).resolve().parents[1]
PINNED_ENV = {"ATEN_CPU_CAPABILITY": "default", "MKL_CBWR": "COMPATIBLE", "OMP_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
DIM, DEPTH, RES = 8, 3, 28
# Batch 128 as in the finals: the LR scales with it, and at batch 4 the whole
# update is smaller than float32 noise. One image is excluded, 128 remain.
BATCH, N_IMAGES = 128, 129
LOSS_KEYS = ("ssl_loss", "consistency_loss", "total_loss", "steps_executed", "samples_processed")


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, h):
        return h + self.fc(self.norm1(h))


class TinyImageViT(nn.Module):
    """The DINOv2 attribute layout the levers act on, over a 2x2 pooled image."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = DIM
        self.patch_embed = nn.Linear(12, DIM)
        self.cls_token = nn.Parameter(torch.zeros(1, DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, DIM))
        self.blocks = nn.ModuleList(TinyBlock(DIM) for _ in range(DEPTH))
        self.norm = nn.LayerNorm(DIM)

    def forward(self, x):
        h = self.patch_embed(nn.functional.adaptive_avg_pool2d(x, 2).flatten(1))
        h = h + self.cls_token + self.pos_embed
        for block in self.blocks:
            h = block(h)
        return self.norm(h)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pool(root: Path) -> dict:
    """The originals, their "tone-shifted" copies and the manifest pretrain validates."""
    from PIL import Image  # noqa: PLC0415

    rng = np.random.default_rng(0)
    orig, shifted = root / "pool", root / "shift"
    orig.mkdir(), shifted.mkdir()
    rows = []
    for i in range(N_IMAGES):
        pixels = (rng.random((32, 32, 3)) * 255).astype("uint8")
        Image.fromarray(pixels).save(orig / f"img{i}.png")
        Image.fromarray(255 - pixels).save(shifted / f"img{i}.png")
        rows.append({"relative_path": f"img{i}.png", "source_sha256": _sha(orig / f"img{i}.png"),
                     "transformed_sha256": _sha(shifted / f"img{i}.png")})
    payload = {"source_revision": "pool-final", "transforms": rows}
    revision = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    (shifted / "manifest.json").write_text(json.dumps({**payload, "revision": revision}), encoding="utf-8")
    (root / "exclude.txt").write_text(f"img{N_IMAGES - 1}\n", encoding="utf-8")
    return {"orig": orig, "shifted": shifted, "revision": revision, "exclude": root / "exclude.txt"}


def _config(config_mod, root: Path, pool: dict, arm: str):
    common = dict(
        method="simsiam_dinov2", backbone="dinov2_vitb14", image_root=str(pool["orig"]),
        output_dir=str(root / f"final_{arm}"), split_hash="a" * 64, data_revision="pool-final",
        freeze_blocks=1, layer_decay=0.75, anchor_l2sp=1e-4, anchor_distill=0.0, head_warmup_steps=2,
        hue_jitter=0.02, max_steps=6, batch_size_per_gpu=BATCH, num_workers=0, resolution=RES,
        precision="bf16", checkpoint_every_steps=3, budget_gpu_hours=8.0, seed=44,
        exclude_ids_file=str(pool["exclude"]),
    )
    if arm == "B":
        common.update(
            adaptation_mode="original_transformed_consistency", transformed_image_root=str(pool["shifted"]),
            transformed_manifest_path=str(pool["shifted"] / "manifest.json"),
            transformed_data_revision=pool["revision"], tone_consistency_weight=0.1, grad_checkpoint=True,
        )
    return config_mod.PretrainConfig(**common)


def run_arm(root: Path, arm: str, patch) -> dict:
    """Train one tiny final; per tensor, the sum and the sum of squares of the
    UPDATE (final minus initial weights), plus the run's losses."""
    from dermssl.pretrain import config as config_mod  # noqa: PLC0415
    from dermssl.pretrain import torch_loop, torch_simsiam  # noqa: PLC0415

    initial = {}

    def build(name, pretrained):
        model = TinyImageViT()
        initial.update({k: v.detach().clone() for k, v in model.state_dict().items()})
        return model, DIM

    patch(torch_simsiam, "_build_backbone", build)
    out = root / arm
    out.mkdir()
    final = torch_loop.train(_config(config_mod, out, _pool(out), arm))
    state = torch.load(final, map_location="cpu", weights_only=True)
    execution = json.loads((final.parent / "run.json").read_text(encoding="utf-8"))["execution"]
    deltas = {k: (v.double() - initial[k].double()) for k, v in state.items()}
    return {
        "update": {k: [float(d.sum()), float(d.pow(2).sum())] for k, d in sorted(deltas.items())},
        "run": {k: execution[k] for k in LOSS_KEYS},
    }


# The frozen-block check below needs a floor far above float32 noise.
ABS_SUM = 1e-7
REFERENCE = "f1c2053"


def reference_src(dest: Path) -> Path:
    """f1c2053's ``src`` tree, from git history (CI checks out with fetch-depth 0)."""
    import io  # noqa: PLC0415
    import tarfile  # noqa: PLC0415

    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", REFERENCE, "src"],
                             capture_output=True, check=False)
    if archive.returncode != 0:
        pytest.fail(f"{REFERENCE} is not in this clone ({archive.stderr.decode().strip()}); "
                    "the characterization needs history (a full clone; CI checks out with fetch-depth: 0)")
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(dest, filter="data")
    return dest / "src"


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("f1c2053")
    return run_pinned(root / "runs", "AB", reference_src(root))


@pytest.mark.parametrize("arm", ("A", "B"))
def test_head_trains_the_finals_like_f1c2053(arm, reference, tmp_path):
    got = run_pinned(tmp_path, arm)[arm]
    assert got["run"]["steps_executed"] == 6 and got["update"]["blocks.2.fc.weight"][1] > 0
    assert got == reference[arm]


def test_the_frozen_block_did_not_move_and_the_trained_ones_did(tmp_path, monkeypatch):
    """Guards the harness itself: the update is far above the tolerance and
    the levers are live, so the comparison above can fail."""
    update = run_arm(tmp_path, "A", monkeypatch.setattr)["update"]
    assert update["patch_embed.weight"] == [0.0, 0.0]
    assert update["blocks.0.fc.weight"] == [0.0, 0.0]
    assert update["blocks.2.fc.weight"][1] > 0 and abs(update["blocks.1.fc.bias"][0]) > 100 * ABS_SUM


def run_pinned(root: Path, arms: str, src: Path = ROOT / "src") -> dict:
    """``{arm: run_arm(...)}`` for each letter of ``arms``, in a child under PINNED_ENV."""
    root.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **PINNED_ENV}
    done = subprocess.run([sys.executable, __file__, str(src), arms, str(root)], env=env,
                          capture_output=True, text=True, check=False, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_child_really_runs_under_the_pinned_environment():
    env = {**os.environ, **PINNED_ENV}
    out = subprocess.run([sys.executable, __file__, "--probe"], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["DEFAULT", "1", "False"]


def _pin_torch():
    """What the environment cannot pin: oneDNN's CPU-dependent JIT kernels."""
    torch.backends.mkldnn.enabled = False
    return torch


def _generate(src: str, arms: list[str], workdir: str) -> None:
    """Print ``{arm: run_arm(...)}`` as the last stdout line. Run it under PINNED_ENV:
    ``env ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE OMP_NUM_THREADS=1 ... python <this> <src>``."""
    import pathlib  # noqa: PLC0415

    sys.path.insert(0, src)
    if sys.version_info < (3, 13):  # f1c2053 passes recurse_symlinks; no symlinks here
        rglob = pathlib.Path.rglob
        pathlib.Path.rglob = lambda self, pattern, **kw: rglob(self, pattern)

    def patch(obj, name, value):
        setattr(obj, name, value)

    _pin_torch()
    print(json.dumps({arm: run_arm(Path(workdir), arm, patch) for arm in arms}))


if __name__ == "__main__":  # pragma: no cover
    import tempfile

    if sys.argv[1] == "--probe":
        pinned = _pin_torch()
        print(pinned.backends.cpu.get_cpu_capability(), pinned.get_num_threads(), pinned.backends.mkldnn.enabled)
        sys.exit(0)
    src_arg, arms_arg = sys.argv[1], (sys.argv[2:3] or ["AB"])[0]
    with tempfile.TemporaryDirectory() as scratch:
        _generate(src_arg, list(arms_arg), sys.argv[3] if len(sys.argv) > 3 else scratch)
