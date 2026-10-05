"""``evaluate_segmentation.py --save-probabilities`` keeps the maps behind the Dice (T6, docs/27).

T6 re-binarises A2's probabilities with other thresholds, so it needs the
exact maps the evaluator scored at 0.5, on its own image path. They are kept
in float32 so that the Dice at 0.5 recomputed from them is identical to the
``per_image.csv`` of the same run; float16 could flip pixels next to 0.5.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401

torch = pytest.importorskip("torch")
import evaluate_segmentation as ev  # noqa: E402
from ckpt_io import save_checkpoint  # noqa: E402
from dermssl import luminance_threshold as lt  # noqa: E402
from dermssl.torch_unet import UNet  # noqa: E402
from eval_inputs import load_binary_mask  # noqa: E402

pytestmark = pytest.mark.torch
SIZE = 32


def _fixture(tmp_path):
    torch.manual_seed(0)
    checkpoint = tmp_path / "student.pt"
    save_checkpoint(checkpoint, {"model": UNet(base_channels=8).state_dict(),
                                 "config": {"base_channels": 8, "resolution": SIZE}})
    images, masks = tmp_path / "img", tmp_path / "gt"
    images.mkdir()
    masks.mkdir()
    rng = np.random.RandomState(1)
    for name in ("ISIC_2", "ISIC_0", "ISIC_1"):
        Image.fromarray(rng.randint(0, 255, (SIZE, SIZE, 3), dtype=np.uint8)).save(images / f"{name}.png")
        truth = np.zeros((SIZE, SIZE), np.uint8)
        truth[8:24, 8:24] = 255
        Image.fromarray(truth).save(masks / f"{name}_segmentation.png")
    return checkpoint, images, masks


def _args(checkpoint, images, masks, out, *extra):
    return ["--checkpoint", str(checkpoint), "--image-dir", str(images), "--mask-dir", str(masks),
            "--output-dir", str(out), *extra]


def test_the_saved_maps_reproduce_the_reported_dice_exactly(tmp_path):
    checkpoint, images, masks = _fixture(tmp_path)
    out, probs = tmp_path / "eval", tmp_path / "probs"
    summary = ev.evaluate(ev.parse_args(_args(checkpoint, images, masks, out, "--save-probabilities", str(probs))))
    assert summary["dice"]["n"] == 3
    maps = np.load(probs / ev.PROBABILITIES_FILE)
    ids = (probs / ev.PROBABILITY_IDS_FILE).read_text(encoding="utf-8").split()
    assert maps.dtype == np.float32 and maps.shape == (3, SIZE, SIZE)
    with open(out / "per_image.csv", newline="") as handle:
        reported = {r["image_id"]: float(r["dice"]) for r in csv.DictReader(handle)}
    assert ids == sorted(reported)
    for image_id, probability in zip(ids, maps, strict=True):
        truth = load_binary_mask(masks / f"{image_id}_segmentation.png", SIZE)
        assert lt.dice_at(probability, truth, ev.DEFAULT_THRESHOLD) == reported[image_id]


def test_saving_probabilities_needs_a_checkpoint(tmp_path, capsys):
    with pytest.raises(SystemExit):
        ev.parse_args(["--pred-mask-dir", str(tmp_path), "--image-dir", str(tmp_path),
                       "--mask-dir", str(tmp_path), "--output-dir", str(tmp_path / "o"),
                       "--save-probabilities", str(tmp_path / "p")])
    assert "--save-probabilities necesita --checkpoint" in capsys.readouterr().err


def test_existing_probabilities_are_never_overwritten(tmp_path, capsys):
    checkpoint, images, masks = _fixture(tmp_path)
    probs = tmp_path / "probs"
    probs.mkdir()
    with pytest.raises(SystemExit):
        ev.parse_args(_args(checkpoint, images, masks, tmp_path / "eval", "--save-probabilities", str(probs)))
    assert "ya existe" in capsys.readouterr().err
