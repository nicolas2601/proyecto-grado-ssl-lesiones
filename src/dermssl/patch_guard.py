"""Resolution guardrails for patch-based backbones (ViT).

Why this module exists
----------------------
A Vision Transformer cuts the image into a grid of square *patches*. DINOv2 uses
14x14 patches, so its input side must be an exact multiple of 14. A ResNet has
no such constraint, which is why v4 never needed this check.

The trap: 224 = 16 x 14 works, but 256 is **not** a multiple of 14 (256 / 14 =
18.28...). The ISIC 2020 dataset we use is distributed at 256x256. Feeding 256
straight into DINOv2 either raises a shape error deep inside attention or, worse,
silently interpolates the positional embeddings and quietly degrades the
features. Both outcomes waste a cluster run.

This module makes the constraint explicit and loud, before any GPU time is spent.
"""

from __future__ import annotations

KNOWN_PATCH_SIZES = {
    "dinov2_vits14": 14,
    "dinov2_vitb14": 14,
    "dinov2_vitl14": 14,
    "vit_base_patch16_224": 16,
    "vit_small_patch16_224": 16,
    "resnet50": None,  # convolutional: no patch constraint
}


class PatchSizeError(ValueError):
    """Raised when an input resolution is incompatible with a patch size."""


def patch_size_for(backbone: str) -> int | None:
    """Patch size for a known backbone, or None when it has no constraint.

    Unknown names fall back to a name-based heuristic (``...14`` -> 14,
    ``...16`` -> 16) so a new checkpoint does not silently skip the check.
    """
    if backbone in KNOWN_PATCH_SIZES:
        return KNOWN_PATCH_SIZES[backbone]
    for suffix, size in (("14", 14), ("16", 16), ("32", 32), ("8", 8)):
        if backbone.endswith(suffix) or f"patch{suffix}" in backbone:
            return size
    return None


def is_compatible(resolution: int, patch: int | None) -> bool:
    """True when ``resolution`` divides evenly into ``patch``-sized tiles."""
    if patch is None:
        return True
    return resolution > 0 and resolution % patch == 0


def snap_to_patch(resolution: int, patch: int | None, mode: str = "down") -> int:
    """Round ``resolution`` to the nearest valid multiple of ``patch``.

    ``mode='down'`` never increases memory use, which is the safe default on a
    16 GB T4. ``mode='up'`` keeps a little more of the image. ``mode='nearest'``
    minimises the change.
    """
    if patch is None:
        return resolution
    if resolution <= 0:
        raise PatchSizeError(f"resolution must be positive, got {resolution}")

    lower = (resolution // patch) * patch
    upper = lower + patch if lower < resolution else lower
    if lower == 0:
        lower = patch

    if mode == "down":
        return lower
    if mode == "up":
        return upper
    if mode == "nearest":
        return lower if (resolution - lower) <= (upper - resolution) else upper
    raise ValueError(f"unknown mode {mode!r}; expected 'down', 'up' or 'nearest'")


def assert_compatible(resolution: int, backbone: str) -> int:
    """Validate ``resolution`` for ``backbone``; return it unchanged when valid.

    Raises :class:`PatchSizeError` with the two nearest valid resolutions so the
    fix is obvious from the message alone.
    """
    patch = patch_size_for(backbone)
    if is_compatible(resolution, patch):
        return resolution
    down = snap_to_patch(resolution, patch, "down")
    up = snap_to_patch(resolution, patch, "up")
    raise PatchSizeError(
        f"resolution {resolution} is not a multiple of patch size {patch} "
        f"(required by '{backbone}'). Use {down} or {up} instead."
    )


def resolve_resolution(requested: int, backbone: str, mode: str = "down") -> tuple[int, str]:
    """Return ``(resolution, note)``, snapping instead of raising.

    Used by the notebook so a single ``SUP_RES`` value can drive both ResNet and
    ViT arms: each arm silently gets a resolution it can actually consume, and
    the note is printed so the change is never invisible.
    """
    patch = patch_size_for(backbone)
    if is_compatible(requested, patch):
        return requested, "ok"
    snapped = snap_to_patch(requested, patch, mode)
    note = (
        f"{requested} -> {snapped} (snapped {mode}: '{backbone}' needs a multiple of {patch})"
    )
    return snapped, note


def num_patch_tokens(resolution: int, patch: int | None) -> int:
    """Token count fed to attention, excluding the CLS token.

    Handy for memory budgeting: attention cost grows with the square of this.
    """
    if patch is None:
        return 0
    assert_compatible(resolution, f"generic{patch}")
    side = resolution // patch
    return side * side
