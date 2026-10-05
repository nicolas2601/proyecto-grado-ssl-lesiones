"""Data loading that never needs /dev/shm.

PyTorch workers return batches through shared memory. The cluster pod has a
64 MB /dev/shm, so the first float32 batch dies with "unable to allocate shared
memory". Here the workers only do the PIL work (resize, crop, flips, colour
jitter, in the same order and with the same RNG draws as before) and return
uint8 numpy arrays, which travel through the worker pipe by pickle. The
ToTensor / Normalize arithmetic runs in the main process, on the device,
with the same float32 operations torchvision uses, so batches are bit-identical.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

from dermssl.finetune import protocol


class ToUint8Array:
    """PIL image -> HWC uint8 numpy array (no torch tensor, so no shared memory)."""

    def __call__(self, img) -> np.ndarray:
        return np.asarray(img, dtype=np.uint8)


def pil_ops(train: bool, interp) -> list:
    """The PIL half of the recipe: identical op order to the original Compose."""
    res = protocol.RESOLUTION
    resize = transforms.Resize(int(res * 1.14), interpolation=interp)
    if not train:
        return [resize, transforms.CenterCrop(res)]
    return [resize, transforms.RandomCrop(res), transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(), transforms.ColorJitter(0.1, 0.1, 0.05)]


def build_worker_transforms(interp):
    """(train, eval) pipelines that end in a uint8 numpy array."""
    return tuple(transforms.Compose([*pil_ops(train, interp), ToUint8Array()])
                 for train in (True, False))


def collate_uint8(batch):
    """Stack samples into numpy arrays (B,H,W,3) uint8 and (B,) int64."""
    xs, ys = zip(*batch)
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def to_model_input(xb: np.ndarray, device) -> torch.Tensor:
    """uint8 NHWC -> normalised float32 NCHW, same ops as ToTensor + Normalize."""
    x = torch.from_numpy(xb).to(device).permute(0, 3, 1, 2).contiguous()
    x = x.to(torch.float32).div(255)
    mean = torch.tensor(protocol.IMAGENET_MEAN, dtype=torch.float32, device=x.device)
    std = torch.tensor(protocol.IMAGENET_STD, dtype=torch.float32, device=x.device)
    return x.sub_(mean.view(1, 3, 1, 1)).div_(std.view(1, 3, 1, 1))


def to_labels(yb: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(yb).to(device)
