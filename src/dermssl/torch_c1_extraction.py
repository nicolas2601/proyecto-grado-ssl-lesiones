"""The torch half of C1 extraction.

Split from :mod:`dermssl.c1_extraction` to respect the repo's ``torch_*``
convention: tests/test_torch_unet_static.py forbids torch imports in the pure
modules so that geometry, provenance and pooling stay importable on a machine
with no GPU stack. Round-3 review caught c1_extraction violating that guard.

Everything here needs a live model; everything there can be unit-tested
without one.
"""

from __future__ import annotations

import numpy as np
import torch

from dermssl.c1_extraction import GRID
from dermssl.mask_conditions import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    mask_pool,
    mask_to_patch_grid,
    uses_patch_tokens,
)


def to_normalized_tensor(image: np.ndarray) -> torch.Tensor:
    """uint8 HWC -> normalized float CHW tensor, matching the bank exactly."""
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1)).contiguous()


def extract_feature(
    model, tensor: torch.Tensor, condition: str, mask: np.ndarray
) -> tuple[np.ndarray, bool]:
    """One feature vector for one image under one condition.

    Returns ``(vector, used_cls_fallback)``. The flag only ever becomes True for
    ``mask_pool``; the caller must aggregate it, because a condition that falls
    back often has quietly collapsed into the baseline.
    """
    batch = tensor.unsqueeze(0)
    with torch.no_grad():
        if not uses_patch_tokens(condition):
            return model(batch).float().cpu().numpy()[0], False

        out = model.forward_features(batch)
        patch_tokens = out["x_norm_patchtokens"].float().cpu().numpy()[0]
        cls_token = out["x_norm_clstoken"].float().cpu().numpy()[0]

    return mask_pool(patch_tokens, mask_to_patch_grid(mask, GRID), cls_token)
