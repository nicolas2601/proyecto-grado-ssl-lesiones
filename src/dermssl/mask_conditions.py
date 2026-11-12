"""The four C1 conditions: how the SSL pseudo-mask guides a frozen backbone.

Experiment C1 asks whether segmentation, given to every backbone equally,
closes the gap between public DINOv2 and the proposed encoder. The mask is
always the SSL pseudo-mask predicted by ``student_r3`` -- never the reference
mask. Using the reference mask here would make the classifier label-dependent
and the whole arm would stop being SSL, so it is not offered as an
option at all rather than left as a flag someone could flip.

The four conditions:

``none``        the current baseline; the backbone sees the whole image.
``bbox_crop``   crop to the lesion bounding box plus margin, resized to 224.
``neutral_bg``  everything outside the mask becomes ImageNet mean grey, which
                lands at ~0 after Normalize, so the background contributes
                almost nothing to the first layer.
``mask_pool``   average the patch tokens whose cell falls inside the mask,
                instead of taking the global CLS token.

Every condition changes WHERE the feature comes from. ``bbox_crop`` and
``mask_pool`` also change the geometry: the first replaces Resize+CenterCrop
with a direct resize of the crop, the second reads ``x_norm_patchtokens``
rather than the pooled output. That is the point of the experiment, and it is
also the hazard -- see :class:`ConditionedFeatures`.
"""

from __future__ import annotations

import numpy as np

LESION_LABEL = 1
UNCERTAIN_LABEL = 255

# The exporter already resolves lesion as ``mask == LESION_LABEL``
# (plan-segmentacion/comun/datos/export_tone_inputs.py:111), which makes 255
# non-lesion. We keep that convention rather than inventing a second one:
# the uncertainty band is exactly where the segmenter was least sure, so
# counting it as foreground would inflate the crop where it is least earned.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CONDITIONS = ("none", "bbox_crop", "neutral_bg", "mask_pool")
_PATCH_TOKEN_CONDITIONS = frozenset({"mask_pool"})


def uses_patch_tokens(condition: str) -> bool:
    """Whether this condition reads ``x_norm_patchtokens`` instead of the pooled output."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    return condition in _PATCH_TOKEN_CONDITIONS


def lesion_mask(raw: np.ndarray) -> np.ndarray:
    """Boolean lesion mask. ``UNCERTAIN_LABEL`` (255) counts as NOT lesion."""
    return np.asarray(raw) == LESION_LABEL


def bbox_with_margin(mask: np.ndarray, margin_frac: float) -> tuple[int, int, int, int] | None:
    """Lesion bounding box grown by ``margin_frac`` of its side, clamped to the image.

    Returns ``None`` when the mask is empty so the caller decides the fallback
    and can count it, instead of silently cropping to a degenerate box.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None

    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    dy = int(round((y1 - y0) * margin_frac))
    dx = int(round((x1 - x0) * margin_frac))
    height, width = mask.shape[:2]

    return (
        max(0, y0 - dy),
        min(height, y1 + dy),
        max(0, x0 - dx),
        min(width, x1 + dx),
    )


def apply_neutral_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace everything outside the mask with ImageNet mean grey.

    Done in uint8 at the original resolution so the standard eval transform can
    run afterwards unchanged. After Normalize those pixels sit within ~0.01 of
    zero, which the tests pin down.
    """
    grey = np.round(np.array(IMAGENET_MEAN, dtype=np.float64) * 255).astype(np.uint8)
    out = np.array(image, dtype=np.uint8, copy=True)
    out[~mask] = grey
    return out


def mask_to_patch_grid(mask: np.ndarray, grid: int) -> np.ndarray:
    """Downsample a mask to the ViT patch grid.

    A patch counts as lesion when any pixel of its cell is lesion. Nearest-style
    block reduction on purpose: interpolating a mask invents fractional
    foreground that has no meaning for a token-level selection.
    """
    height, width = mask.shape[:2]
    ys = np.linspace(0, height, grid + 1).round().astype(int)
    xs = np.linspace(0, width, grid + 1).round().astype(int)

    out = np.zeros((grid, grid), dtype=bool)
    for i in range(grid):
        for j in range(grid):
            block = mask[ys[i] : ys[i + 1], xs[j] : xs[j + 1]]
            out[i, j] = bool(block.any())
    return out


def mask_pool(
    patch_tokens: np.ndarray,
    grid_mask: np.ndarray,
    cls_token: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Average the patch tokens inside the mask; fall back to CLS when none are.

    Returns ``(vector, used_fallback)``. The flag is returned rather than logged
    because the caller has to COUNT the fallbacks: if a large share of images
    fall back, condition (c) is quietly degenerating into condition (0) and the
    comparison stops meaning what its name says.
    """
    tokens = np.asarray(patch_tokens)
    expected = int(grid_mask.size)
    if tokens.shape[0] != expected:
        raise ValueError(
            f"got {tokens.shape[0]} patch tokens but the grid holds {expected}; "
            "token count and grid must describe the same layout"
        )

    selected = grid_mask.reshape(-1)
    if not selected.any():
        return np.asarray(cls_token, dtype=tokens.dtype), True

    return tokens[selected].mean(axis=0), False
