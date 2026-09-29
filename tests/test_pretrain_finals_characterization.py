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
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
nn = torch.nn

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


# Another CPU may round a float32 kernel differently in the last bit, which
# moves a summed update by ~1e-8 (weights ~0.3, updates ~1e-5 per element).
# A real change to the training path (schedule, levers, augmentation, loss)
# moves it by tens of percent. The sums can cancel, so they also get an
# absolute floor; the sums of squares cannot, so they get none.
REL, ABS_SUM, REL_LOSS = 1e-2, 1e-7, 1e-4
# Generated at f1c2053 (see the module docstring); HEAD produced the identical output.
GOLDEN = {'A': {'run': {'consistency_loss': None,
               'samples_processed': 768,
               'ssl_loss': -0.01044647543070217,
               'steps_executed': 6,
               'total_loss': -0.01044647543070217},
       'update': {'blocks.0.fc.bias': [0.0, 0.0],
                  'blocks.0.fc.weight': [0.0, 0.0],
                  'blocks.0.norm1.bias': [-1.6915576452447567e-05, 1.143969345001629e-08],
                  'blocks.0.norm1.weight': [3.5881996154785156e-05, 8.71504113320043e-09],
                  'blocks.1.fc.bias': [0.00011227279901504517, 1.791252018934486e-08],
                  'blocks.1.fc.weight': [7.214443758130074e-05, 1.4664417343965546e-07],
                  'blocks.1.norm1.bias': [0.00012831932144763414, 2.198456895379145e-08],
                  'blocks.1.norm1.weight': [1.9669532775878906e-05, 2.1257015703213256e-08],
                  'blocks.2.fc.bias': [-4.736077971756458e-05, 2.9606398322149733e-08],
                  'blocks.2.fc.weight': [5.420530214905739e-06, 2.4015551030725233e-07],
                  'blocks.2.norm1.bias': [-0.00024213392225647112, 3.159251573234226e-08],
                  'blocks.2.norm1.weight': [0.00013375282287597656, 2.286163436338029e-08],
                  'cls_token': [0.0, 0.0],
                  'norm.bias': [8.796030670055188e-06, 3.663216368728565e-09],
                  'norm.weight': [2.300739288330078e-05, 4.4051596148619865e-08],
                  'patch_embed.bias': [0.0, 0.0],
                  'patch_embed.weight': [0.0, 0.0],
                  'pos_embed': [0.0, 0.0]}},
 'B': {'run': {'consistency_loss': -0.009807558031752706,
               'samples_processed': 768,
               'ssl_loss': -0.009807558031752706,
               'steps_executed': 6,
               'total_loss': -0.007307125876347224},
       'update': {'blocks.0.fc.bias': [0.0, 0.0],
                  'blocks.0.fc.weight': [0.0, 0.0],
                  'blocks.0.norm1.bias': [2.377521514063119e-05, 9.054252376139144e-09],
                  'blocks.0.norm1.weight': [-6.157159805297852e-05, 1.0568793840093349e-08],
                  'blocks.1.fc.bias': [-4.8041343688964844e-05, 1.1015624745702723e-08],
                  'blocks.1.fc.weight': [0.00018159340834245086, 1.2560770952605915e-07],
                  'blocks.1.norm1.bias': [-3.0239303669077344e-05, 1.3479651695113836e-08],
                  'blocks.1.norm1.weight': [2.8967857360839844e-05, 2.169941382135221e-08],
                  'blocks.2.fc.bias': [8.933991193771362e-05, 3.098484308017291e-08],
                  'blocks.2.fc.weight': [-4.8837391659617424e-05, 1.86007173446029e-07],
                  'blocks.2.norm1.bias': [-0.00024194231536966981, 2.7420947620678967e-08],
                  'blocks.2.norm1.weight': [8.475780487060547e-05, 1.2559375761611591e-08],
                  'cls_token': [0.0, 0.0],
                  'norm.bias': [-8.927432645577937e-06, 8.299059535201982e-08],
                  'norm.weight': [-7.241964340209961e-05, 5.885090459400999e-08],
                  'patch_embed.bias': [0.0, 0.0],
                  'patch_embed.weight': [0.0, 0.0],
                  'pos_embed': [0.0, 0.0]}}}


@pytest.mark.parametrize("arm", ("A", "B"))
def test_head_trains_the_finals_like_f1c2053(arm, tmp_path, monkeypatch):
    got = run_arm(tmp_path, arm, monkeypatch.setattr)
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


def _generate(src: str) -> None:  # pragma: no cover - run by hand against f1c2053
    import pathlib  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    sys.path.insert(0, src)
    rglob = pathlib.Path.rglob
    pathlib.Path.rglob = lambda self, pattern, **kw: rglob(self, pattern)

    def patch(obj, name, value):
        setattr(obj, name, value)

    with tempfile.TemporaryDirectory() as tmp:
        print(json.dumps({arm: run_arm(Path(tmp), arm, patch) for arm in ("A", "B")}, indent=1))


if __name__ == "__main__":  # pragma: no cover
    _generate(sys.argv[1])
