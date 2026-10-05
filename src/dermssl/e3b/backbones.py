"""The eight frozen public backbones with the frozen bank's weights, geometry and feature point.

Loading is delegated to ``dermssl.finetune.torch_run.load_backbone`` (timm tags of
``finetune.protocol.TIMM_HF_TAGS``; alexnet from torchvision; SimSiam R50 from the FAIR
ImageNet-1k checkpoint; vgg19 read at the GAP of conv5, 512-d, not timm's 4096-d
pre-logits), so the weights and the extraction point are the ones the bank used. Imports of
torch and timm are lazy: tests inject ``loader``.

Geometry is ``resize(int(224 * 1.14))`` + centre crop 224 (bicubic for efficientnet_b0, the
bank's choice) and ImageNet mean/std for every backbone, exactly like the bank notebooks.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dermssl.e3b import protocol as proto

INTERPOLATIONS = ("bilinear", "bicubic")


@dataclass
class Backbone:
    name: str
    dim: int
    model: object
    resize_crop: Callable[[np.ndarray, int], np.ndarray]
    extract: Callable[[np.ndarray], np.ndarray]
    weights_sha256: str
    interpolation: str


def interpolation_name(interp) -> str:
    """``torchvision.InterpolationMode`` (or anything with ``.value``) -> ``bilinear`` / ``bicubic``."""
    name = getattr(interp, "value", interp)
    if name not in INTERPOLATIONS:
        raise ValueError(f"interpolacion {name!r} no registrada; se esperaba una de {INTERPOLATIONS}")
    return name


def make_resize_crop(interpolation: str) -> Callable[[np.ndarray, int], np.ndarray]:
    interpolation = interpolation_name(interpolation)
    from PIL import Image  # noqa: PLC0415
    from torchvision import transforms  # noqa: PLC0415

    mode = {"bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC}[interpolation]

    def resize_crop(image: np.ndarray, resolution: int) -> np.ndarray:
        geometry = transforms.Compose([transforms.Resize(int(resolution * 1.14), interpolation=mode),
                                       transforms.CenterCrop(resolution)])
        return np.asarray(geometry(Image.fromarray(image)), dtype=np.uint8)

    return resize_crop


def state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _default_loader(name: str):
    from dermssl.finetune import torch_run  # noqa: PLC0415

    return torch_run.load_backbone(name)


def _make_extract(model, device) -> Callable[[np.ndarray], np.ndarray]:
    import torch  # noqa: PLC0415

    from dermssl import torch_v71 as tv  # noqa: PLC0415

    def extract(images: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(tv.normalize_uint8(images).to(device)).float().cpu().numpy().astype(np.float32)

    return extract


def build_backbone(name: str, device, *, loader=None) -> Backbone:
    """Frozen, eval-mode backbone; the width is asserted on a dummy forward pass."""
    if name not in proto.MODELS:
        raise ValueError(f"modelo {name!r} no esta en el banco: {proto.MODELS}")
    model, interp = (loader or _default_loader)(name)
    for param in model.parameters():
        param.requires_grad = False
    model.eval().to(device)
    interpolation = interpolation_name(interp)
    extract = _make_extract(model, device)
    probe = extract(np.zeros((2, proto.RESOLUTION, proto.RESOLUTION, 3), dtype=np.uint8))
    proto.check_width(name, probe.shape[-1])
    return Backbone(name, int(probe.shape[-1]), model, make_resize_crop(interpolation), extract,
                    state_sha256(model), interpolation)
