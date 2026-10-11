"""Observe the real SimSiamPairDataset without changing its image augmentations.

Single-process CPU audit only. Masks are read outside the dataset, never returned
to SSL training. Geometry is replayed with nearest interpolation on pseudo-masks.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode

from dermssl.pretrain.torch_data import IMAGENET_MEAN, IMAGENET_STD
from dermssl.view_audit import crop_stats


class GeometryObserver:
    """Wrap the existing Compose; extra draws are rewound, preserving all outputs."""

    def __init__(self, geometric, mask: np.ndarray):
        self.geometric = geometric
        self.mask = mask
        self.records = []

    def __call__(self, image):
        before = torch.get_rng_state()
        resize = self.geometric.transforms[0]
        crop = resize.get_params(image, resize.scale, resize.ratio)
        torch.set_rng_state(before)
        result = self.geometric(image)
        after = torch.get_rng_state()
        try:
            torch.set_rng_state(before)
            replay = copy.deepcopy(self.geometric)
            replay.transforms[0].interpolation = InterpolationMode.NEAREST
            mask_image = Image.fromarray(self.mask.astype(np.uint8) * 255)
            warped = np.asarray(replay(mask_image)) > 0
        finally:
            torch.set_rng_state(after)
        self.records.append({"crop_tlhw": list(crop), "mask": warped,
                             **crop_stats(self.mask, crop)})
        return result


def tensor_image(tensor) -> Image.Image:
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    rgb = ((tensor.detach().cpu() * std + mean).clamp(0, 1) * 255).round()
    return Image.fromarray(rgb.permute(1, 2, 0).to(torch.uint8).numpy())


def audit_item(dataset, index: int, mask: np.ndarray):
    """Return views and aligned masks in dataset order, restoring global RNG."""
    before = torch.get_rng_state()
    geometric = dataset.geometric
    observer = GeometryObserver(geometric, mask)
    try:
        torch.manual_seed(dataset.seed + index)
        dataset.geometric = observer
        views = dataset[index]
    finally:
        dataset.geometric = geometric
        torch.set_rng_state(before)
    # _with_tone_view executes v1, tone, v2; its return order is v1, v2, tone.
    order = (0, 2, 1) if dataset.tone_view else (0, 1)
    if len(observer.records) != len(order):
        raise ValueError("Unexpected dataset geometry calls; audit adapter must be updated")
    return views, [observer.records[i] for i in order]
