"""The S4-UNet is scored by the SAME evaluator as student_r3 (docs/27).

``evaluate_segmentation.py --checkpoint`` dispatches on the checkpoint's
``arch``; everything after the model (binarisation at 0.5, metrics, report)
is the student's code path, untouched.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401

torch = pytest.importorskip("torch")
import eval_student  # noqa: E402
import evaluate_segmentation as ev  # noqa: E402
from ckpt_io import save_checkpoint  # noqa: E402
from dermssl import torch_s4_unet as s4  # noqa: E402
from dermssl.torch_unet import UNet  # noqa: E402
from s4_fakes import FakeVit  # noqa: E402

pytestmark = pytest.mark.torch
SIZE = 56


def _s4_checkpoint(tmp_path, source="public", sha="public"):
    torch.manual_seed(0)
    model = s4.S4UNet(FakeVit(), features=16)
    config = s4.checkpoint_config(model, backbone_arch="dinov2_vitb14", backbone_source=source,
                                  backbone_sha256=sha, resolution=SIZE)
    path = tmp_path / "best.pt"
    save_checkpoint(path, {"model": s4.decoder_state(model), "config": config, "epoch": 0})
    return path, model.eval()


@pytest.fixture
def fake_hub(monkeypatch):
    calls = []

    def hub(arch, source):
        calls.append((arch, source))
        return FakeVit()

    monkeypatch.setattr(s4, "hub_backbone", hub)
    return calls


def test_an_s4_checkpoint_loads_as_the_s4_model_at_its_training_resolution(tmp_path, fake_hub):
    path, trained = _s4_checkpoint(tmp_path)
    model, resolution = eval_student.load_student(path, torch.device("cpu"))
    assert isinstance(model, s4.S4UNet) and resolution == SIZE
    assert fake_hub == [("dinov2_vitb14", "public")]
    x = torch.randn(1, 3, SIZE, SIZE)
    assert torch.equal(model(x), trained(x))


def test_a_student_checkpoint_without_arch_still_loads_the_unet(tmp_path):
    path = tmp_path / "student.pt"
    save_checkpoint(path, {"model": UNet(base_channels=8).state_dict(),
                           "config": {"base_channels": 8, "resolution": 64}})
    model, resolution = eval_student.load_student(path, torch.device("cpu"))
    assert isinstance(model, UNet) and resolution == 64


def test_backbone_weights_that_changed_since_training_are_refused(tmp_path, fake_hub):
    weights = tmp_path / "astar.pt"
    weights.write_bytes(b"not the trained weights")
    path, _ = _s4_checkpoint(tmp_path, source=str(weights), sha="0" * 64)
    with pytest.raises(ValueError, match="sha256"):
        eval_student.load_student(path, torch.device("cpu"))


def test_the_evaluator_scores_an_s4_checkpoint_with_the_student_report(tmp_path, fake_hub):
    path, _ = _s4_checkpoint(tmp_path)
    images, masks = tmp_path / "img", tmp_path / "gt"
    images.mkdir()
    masks.mkdir()
    for index in range(3):
        Image.fromarray(np.full((SIZE, SIZE, 3), 40 * index, np.uint8)).save(images / f"ISIC_{index}.png")
        truth = np.zeros((SIZE, SIZE), np.uint8)
        truth[20:36, 20:36] = 255
        Image.fromarray(truth).save(masks / f"ISIC_{index}_segmentation.png")
    out = tmp_path / "eval"
    ev.evaluate(ev.parse_args(["--checkpoint", str(path), "--image-dir", str(images),
                               "--mask-dir", str(masks), "--output-dir", str(out),
                               "--eval-name", "isic2018", "--hausdorff"]))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["n_images"] == 3 and summary["meta"]["resolution"] == SIZE
    assert summary["meta"]["checkpoint"] == str(path)
