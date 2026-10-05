"""Frozen-backbone wrapper: eval mode, no gradients, width asserted, resize geometry of the bank."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from dermssl.e3b import backbones as bb  # noqa: E402
from dermssl.e3b import protocol as proto  # noqa: E402


class Tiny(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin, self.bn = nn.Linear(3, width), nn.BatchNorm1d(width)

    def forward(self, x):
        return self.bn(self.lin(x.mean(dim=(2, 3))))


def loader_of(width, interp="bilinear"):
    return lambda name: (Tiny(width), interp)


def test_a_wrapped_backbone_is_frozen_in_eval_mode_and_reports_its_width():
    back = bb.build_backbone("resnet50", torch.device("cpu"), loader=loader_of(2048))
    assert back.dim == 2048 and len(back.weights_sha256) == 64
    assert not back.model.training and not any(p.requires_grad for p in back.model.parameters())


def test_the_vgg19_4096_trap_fails_loudly():
    with pytest.raises(proto.FeatureWidthError, match="4096"):
        bb.build_backbone("vgg19", torch.device("cpu"), loader=loader_of(4096))


def test_extract_returns_float32_features_of_the_batch_without_gradients():
    back = bb.build_backbone("resnet50", torch.device("cpu"), loader=loader_of(2048))
    images = np.random.default_rng(0).integers(0, 255, (5, 224, 224, 3), dtype=np.uint8)
    out = back.extract(images)
    assert out.shape == (5, 2048) and out.dtype == np.float32
    assert np.allclose(out, back.extract(images), atol=1e-6)  # eval mode: batch norm uses running stats


def test_the_weights_sha_changes_with_the_weights():
    a = bb.build_backbone("resnet50", torch.device("cpu"), loader=loader_of(2048))
    torch.manual_seed(1)
    b = bb.build_backbone("resnet50", torch.device("cpu"), loader=loader_of(2048))
    assert a.weights_sha256 != b.weights_sha256


def test_resize_crop_is_the_bank_geometry_and_efficientnet_is_bicubic():
    image = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    from dermssl import torch_v71 as tv
    from PIL import Image

    bilinear = bb.make_resize_crop("bilinear")(image, 224)
    assert bilinear.shape == (224, 224, 3) and bilinear.dtype == np.uint8
    assert np.array_equal(bilinear, tv.resize_crop_uint8(Image.fromarray(image), 224))
    assert not np.array_equal(bb.make_resize_crop("bicubic")(image, 224), bilinear)
    with pytest.raises(ValueError):
        bb.make_resize_crop("lanczos")


def test_the_interpolation_name_follows_the_loader_of_the_bank():
    assert bb.interpolation_name(type("I", (), {"value": "bicubic"})()) == "bicubic"
