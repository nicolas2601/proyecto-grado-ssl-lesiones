"""Running a trained student over an evaluation set.

Kept apart from ``evaluate_segmentation`` so that file stays small and torch
stays out of import time. Nothing here decodes a reference mask or defines a
metric: the caller injects ``make_row``, which owns both. That inversion is
what keeps this module free of the evaluation policy.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from eval_inputs import load_image


def load_student(checkpoint_path: Path, device):
    """Return ``(model in eval mode, training resolution)``."""
    from ckpt_io import load_checkpoint
    from dermssl.torch_unet import UNet

    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config", {})
    model = UNet(base_channels=int(config.get("base_channels", 32)))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), int(config.get("resolution", 224))


# Flip axes of an (N, C, H, W) batch: identity, horizontal, vertical, both.
TTA_FLIPS = ((), (3,), (2,), (2, 3))


def _probabilities(model, batch, tta: bool):
    """Sigmoid output, averaged over flipped views (undone in place) when ``tta``."""
    import torch

    if not tta:
        return torch.sigmoid(model(batch))
    total = None
    for axes in TTA_FLIPS:
        view = torch.flip(batch, axes) if axes else batch
        prob = torch.sigmoid(model(view))
        prob = torch.flip(prob, axes) if axes else prob
        total = prob if total is None else total + prob
    return total / len(TTA_FLIPS)


def predict_masks(
    model, images: np.ndarray, device, batch_size: int, tta: bool = False
) -> np.ndarray:
    """Probabilities ``(N, H, W)`` float32 for a uint8 image stack ``(N, H, W, 3)``.

    ``tta`` averages the prediction over the four flip combinations; it never
    touches training, thresholds or checkpoints, only the inference view.
    """
    import torch
    from gpu_augment import normalize_batch, to_float_batch

    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = torch.from_numpy(images[start : start + batch_size]).to(device)
            batch = normalize_batch(to_float_batch(chunk))
            outputs.append(_probabilities(model, batch, tta).squeeze(1).float().cpu().numpy())
    return np.concatenate(outputs)


def score_pairs(
    model,
    device,
    pairs: Sequence[tuple[str, Path, Path]],
    resolution: int,
    batch_size: int,
    make_row: Callable[[str, np.ndarray, Path], dict],
    tta: bool = False,
) -> list[dict]:
    """One row per image, running the student in batches of ``batch_size``."""
    rows = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        images = np.stack([load_image(image_path, resolution) for _, image_path, _ in chunk])
        probabilities = predict_masks(model, images, device, len(chunk), tta=tta)
        for (image_id, _, mask_path), probability in zip(chunk, probabilities, strict=True):
            rows.append(make_row(image_id, probability, mask_path))
    return rows
