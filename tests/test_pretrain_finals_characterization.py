"""HEAD trains A* and B* exactly as f1c2053, the revision of the s42/s43 finals.

The seed-44 replica runs from a later revision (the launch guards live there)
while another change splits ``torch_loop``. This test pins the training path:
the full ``torch_loop.train`` with the finals' levers (frozen blocks, layer
decay, L2-SP, head warm-up, and for B the paired tone view, the tone term and
gradient checkpointing), on CPU with a tiny ViT in place of the hub model.
``GOLDEN`` was produced by this same harness run against a checkout of
f1c2053 (``python tests/test_pretrain_finals_characterization.py <src>``);
the only thing patched there is ``Path.rglob``, whose ``recurse_symlinks``
argument f1c2053 needs Python 3.13 for (no symlinks here, same file list).

Both the golden and the comparison run in a child process under PINNED_ENV.
Without it the result depends on the machine: with 1 or 2 torch threads the
same code gives 3.5107e-05 for blocks.0.norm1.weight instead of 3.5882e-05
(six threads), a 2 % drift that six training steps grow from last-bit
rounding, and CI failed on it from the day the test landed. One thread
fixes the reduction order (six threads still differ even with the rest
pinned), scalar ATen kernels (ATEN_CPU_CAPABILITY=default) and MKL's
conditional numerical reproducibility (MKL_CBWR=COMPATIBLE) take the CPU's
vector unit out of it. They must be set before torch is imported, hence the
child. The child also turns oneDNN off: the augmentation's depthwise
convolution ran through its JIT kernels, chosen by the CPU (AVX2 here,
AVX-512 on some CI runners), which is why the first pinned version still
failed on two of the three CI jobs. Under all of it f1c2053 and HEAD give
GOLDEN bit for bit.
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


# Under PINNED_ENV the run is bit-for-bit reproducible on this machine; the
# tolerance is for another x86 CPU whose libm or MKL build rounds differently.
# A real change to the training path (schedule, levers, augmentation, loss)
# moves the update by tens of percent. The sums can cancel, so they also get an
# absolute floor; the sums of squares cannot, so they get none.
REL, ABS_SUM, REL_LOSS = 1e-2, 1e-7, 1e-4
# Generated at f1c2053 under PINNED_ENV (see the module docstring); HEAD, under the same
# environment, produced the identical output bit for bit.
GOLDEN = {'A': {'run': {'consistency_loss': None,
               'samples_processed': 768,
               'ssl_loss': -0.01045061481030037,
               'steps_executed': 6,
               'total_loss': -0.01045061481030037},
       'update': {'blocks.0.fc.bias': [0.0, 0.0],
                  'blocks.0.fc.weight': [0.0, 0.0],
                  'blocks.0.norm1.bias': [-1.7361821846861858e-05, 1.1477656785196688e-08],
                  'blocks.0.norm1.weight': [3.516674041748047e-05, 8.690697939073289e-09],
                  'blocks.1.fc.bias': [0.00011029466986656189, 1.789893029624512e-08],
                  'blocks.1.fc.weight': [7.827644003555179e-05, 1.4620651452168176e-07],
                  'blocks.1.norm1.bias': [0.00012660841639444698, 2.1953596318396276e-08],
                  'blocks.1.norm1.weight': [2.0325183868408203e-05, 2.1428572694048853e-08],
                  'blocks.2.fc.bias': [-4.5515596866607666e-05, 3.010484100895994e-08],
                  'blocks.2.fc.weight': [0.0001562254037708044, 2.4023388324820186e-07],
                  'blocks.2.norm1.bias': [-0.0002429707956252969, 3.148658427983256e-08],
                  'blocks.2.norm1.weight': [0.00013953447341918945, 2.2311905212291094e-08],
                  'cls_token': [0.0, 0.0],
                  'norm.bias': [6.738127126482141e-06, 8.644614904035569e-10],
                  'norm.weight': [-7.450580596923828e-05, 4.8545850006576075e-08],
                  'patch_embed.bias': [0.0, 0.0],
                  'patch_embed.weight': [0.0, 0.0],
                  'pos_embed': [0.0, 0.0]}},
 'B': {'run': {'consistency_loss': -0.009796655581643185,
               'samples_processed': 768,
               'ssl_loss': -0.009796655581643185,
               'steps_executed': 6,
               'total_loss': -0.007296227539579074},
       'update': {'blocks.0.fc.bias': [0.0, 0.0],
                  'blocks.0.fc.weight': [0.0, 0.0],
                  'blocks.0.norm1.bias': [2.4745757855271222e-05, 9.069063679520658e-09],
                  'blocks.0.norm1.weight': [-6.186962127685547e-05, 1.0578737885680312e-08],
                  'blocks.1.fc.bias': [-4.714354872703552e-05, 1.0958675591976608e-08],
                  'blocks.1.fc.weight': [0.00017984281294047832, 1.2507356624246986e-07],
                  'blocks.1.norm1.bias': [-2.930683422164293e-05, 1.3491261924406104e-08],
                  'blocks.1.norm1.weight': [2.944469451904297e-05, 2.1719650078466657e-08],
                  'blocks.2.fc.bias': [9.175552986562252e-05, 3.0800812712025814e-08],
                  'blocks.2.fc.weight': [-4.752911627292633e-05, 1.8478839024574367e-07],
                  'blocks.2.norm1.bias': [-0.00024156909421435557, 2.723401142152083e-08],
                  'blocks.2.norm1.weight': [7.921457290649414e-05, 1.2530794180065641e-08],
                  'cls_token': [0.0, 0.0],
                  'norm.bias': [-8.928356692194939e-06, 8.299129576025322e-08],
                  'norm.weight': [-6.377696990966797e-05, 5.7234203154621355e-08],
                  'patch_embed.bias': [0.0, 0.0],
                  'patch_embed.weight': [0.0, 0.0],
                  'pos_embed': [0.0, 0.0]}}}


@pytest.mark.parametrize("arm", ("A", "B"))
def test_head_trains_the_finals_like_f1c2053(arm, tmp_path, monkeypatch):
    got = run_pinned(tmp_path, arm)
    want = GOLDEN[arm]
    assert got["run"]["steps_executed"] == want["run"]["steps_executed"] == 6
    assert sorted(got["update"]) == sorted(want["update"])
    for key, (total, squares) in want["update"].items():
        assert got["update"][key][0] == pytest.approx(total, rel=REL, abs=ABS_SUM), key
        assert got["update"][key][1] == pytest.approx(squares, rel=REL, abs=0.0), key
    for key, value in want["run"].items():
        assert got["run"][key] == pytest.approx(value, rel=REL_LOSS), key


def test_the_frozen_block_did_not_move_and_the_trained_ones_did(tmp_path, monkeypatch):
    """Guards the harness itself: the update is far above the tolerance and
    the levers are live, so the comparison above can fail."""
    update = run_arm(tmp_path, "A", monkeypatch.setattr)["update"]
    assert update["patch_embed.weight"] == [0.0, 0.0]
    assert update["blocks.0.fc.weight"] == [0.0, 0.0]
    assert update["blocks.2.fc.weight"][1] > 0 and abs(update["blocks.1.fc.bias"][0]) > 100 * ABS_SUM


def run_pinned(root: Path, arm: str, src: Path = ROOT / "src") -> dict:
    """``run_arm`` in a child under PINNED_ENV; the same entry point produced GOLDEN."""
    env = {**os.environ, **PINNED_ENV}
    done = subprocess.run([sys.executable, __file__, str(src), arm, str(root)], env=env,
                          capture_output=True, text=True, check=False, timeout=600)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])[arm]


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
