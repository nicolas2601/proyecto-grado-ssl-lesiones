"""Batched GPU augmentation with plain torch ops (no CPU workers, no torchvision).

Every function takes a float batch ``(B, 3, H, W)`` in [0, 1] and an explicit
``torch.Generator`` so runs are reproducible. Randomness is per sample.
Geometric transforms can carry a mask ``(B, 1, H, W)`` through the same affine
grid with nearest-neighbour sampling, so label values (including the ignore
value 255) are never interpolated.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# RGB <-> YIQ, used to rotate hue as a rotation of the chroma plane.
_RGB_TO_YIQ = torch.tensor(
    [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]]
)


def to_float_batch(images_uint8: torch.Tensor) -> torch.Tensor:
    """``(B, H, W, 3)`` uint8 -> ``(B, 3, H, W)`` float in [0, 1]."""
    return images_uint8.permute(0, 3, 1, 2).float().div_(255.0)


def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def _uniform(count: int, low: float, high: float, generator, device) -> torch.Tensor:
    sample = torch.rand(count, generator=generator, device=device)
    return low + (high - low) * sample


def _per_sample_gate(count: int, probability: float, generator, device) -> torch.Tensor:
    return (torch.rand(count, generator=generator, device=device) < probability).view(-1, 1, 1, 1)


def random_resized_crop(
    images: torch.Tensor,
    generator: torch.Generator,
    *,
    scale: tuple[float, float] = (0.2, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    masks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Random crop (area fraction ``scale``, aspect ``ratio``) resized back, plus flips.

    The crop always lies inside the image, so no padding value ever appears.
    """
    count, _, height, width = images.shape
    device = images.device
    area = _uniform(count, scale[0], scale[1], generator, device)
    log_ratio = _uniform(count, math.log(ratio[0]), math.log(ratio[1]), generator, device)
    aspect = torch.exp(log_ratio)
    crop_w = torch.sqrt(area * aspect).clamp(max=1.0)
    crop_h = torch.sqrt(area / aspect).clamp(max=1.0)
    center_x = (torch.rand(count, generator=generator, device=device) * 2 - 1) * (1 - crop_w)
    center_y = (torch.rand(count, generator=generator, device=device) * 2 - 1) * (1 - crop_h)
    sign_x = torch.ones(count, device=device)
    sign_y = torch.ones(count, device=device)
    if horizontal_flip:
        flip = torch.rand(count, generator=generator, device=device) < 0.5
        sign_x = torch.where(flip, -sign_x, sign_x)
    if vertical_flip:
        flip = torch.rand(count, generator=generator, device=device) < 0.5
        sign_y = torch.where(flip, -sign_y, sign_y)

    theta = torch.zeros(count, 2, 3, device=device)
    theta[:, 0, 0] = crop_w * sign_x
    theta[:, 0, 2] = center_x
    theta[:, 1, 1] = crop_h * sign_y
    theta[:, 1, 2] = center_y

    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    cropped = F.grid_sample(
        images, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    cropped_masks = None
    if masks is not None:
        cropped_masks = F.grid_sample(
            masks, grid, mode="nearest", padding_mode="border", align_corners=False
        )
    return cropped, cropped_masks


def _grayscale(images: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], device=images.device).view(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def color_jitter(
    images: torch.Tensor,
    generator: torch.Generator,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
    *,
    probability: float = 0.8,
) -> torch.Tensor:
    """Brightness, contrast, saturation and hue jitter applied with ``probability``."""
    count = images.shape[0]
    device = images.device
    out = images
    if brightness > 0:
        factor = _uniform(count, 1 - brightness, 1 + brightness, generator, device)
        out = out * factor.view(-1, 1, 1, 1)
    if contrast > 0:
        factor = _uniform(count, 1 - contrast, 1 + contrast, generator, device).view(-1, 1, 1, 1)
        mean = _grayscale(out).mean(dim=(1, 2, 3), keepdim=True)
        out = (out - mean) * factor + mean
    if saturation > 0:
        factor = _uniform(count, 1 - saturation, 1 + saturation, generator, device)
        gray = _grayscale(out)
        out = (out - gray) * factor.view(-1, 1, 1, 1) + gray
    if hue > 0:
        angle = _uniform(count, -hue, hue, generator, device) * 2 * math.pi
        out = _rotate_hue(out, angle)
    out = out.clamp(0.0, 1.0)
    gate = _per_sample_gate(count, probability, generator, device)
    return torch.where(gate, out, images)


def _rotate_hue(images: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    to_yiq = _RGB_TO_YIQ.to(images.device)
    to_rgb = torch.linalg.inv(to_yiq)
    cos, sin = torch.cos(angle), torch.sin(angle)
    rotation = torch.zeros(len(angle), 3, 3, device=images.device)
    rotation[:, 0, 0] = 1.0
    rotation[:, 1, 1], rotation[:, 1, 2] = cos, -sin
    rotation[:, 2, 1], rotation[:, 2, 2] = sin, cos
    matrix = to_rgb.unsqueeze(0) @ rotation @ to_yiq.unsqueeze(0)
    return torch.einsum("bij,bjhw->bihw", matrix, images)


def random_grayscale(
    images: torch.Tensor, generator: torch.Generator, *, probability: float = 0.2
) -> torch.Tensor:
    gate = _per_sample_gate(images.shape[0], probability, generator, images.device)
    return torch.where(gate, _grayscale(images).expand_as(images), images)


def gaussian_blur(
    images: torch.Tensor,
    generator: torch.Generator,
    *,
    probability: float = 0.5,
    sigma_range: tuple[float, float] = (0.1, 2.0),
) -> torch.Tensor:
    """Per-sample Gaussian blur; sigma is given in pixels at 224 and scaled to the size."""
    count, channels, height, width = images.shape
    device = images.device
    kernel_size = max(3, int(0.1 * height) // 2 * 2 + 1)
    sigma = _uniform(count, sigma_range[0], sigma_range[1], generator, device) * (height / 224)
    sigma = sigma.clamp(min=1e-3)
    offsets = torch.arange(kernel_size, device=device) - kernel_size // 2
    kernel = torch.exp(-(offsets.view(1, -1) ** 2) / (2 * sigma.view(-1, 1) ** 2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)  # (B, k)
    kernel = kernel.repeat_interleave(channels, dim=0)  # (B*C, k)

    stacked = images.reshape(1, count * channels, height, width)
    pad = kernel_size // 2
    padded = F.pad(stacked, (pad, pad, pad, pad), mode="reflect")
    horizontal = F.conv2d(padded, kernel.view(-1, 1, 1, kernel_size), groups=count * channels)
    blurred = F.conv2d(horizontal, kernel.view(-1, 1, kernel_size, 1), groups=count * channels)
    blurred = blurred.reshape(count, channels, height, width)
    gate = _per_sample_gate(count, probability, generator, device)
    return torch.where(gate, blurred, images)


def random_vignette(
    images: torch.Tensor,
    generator: torch.Generator,
    *,
    probability: float = 0.3,
    radius_range: tuple[float, float] = (0.7, 0.85),
    strength_range: tuple[float, float] = (0.3, 0.8),
    gamma_range: tuple[float, float] = (1.5, 3.0),
) -> torch.Tensor:
    """Multiplicative radial darkening, the optical signature of a dermoscope.

    Half of the pool is captured through a dermoscope and half is not, so the
    vignette is a domain cue the backbone can latch onto instead of the lesion.
    Adding it at random on both sides removes the shortcut. The attenuation is
    ``1 - strength * clip((r - r0) / (1 - r0), 0, 1) ** gamma`` with ``r``
    normalized so that it is 1 at the edge midpoints, measured from a centre
    jittered by up to 5 percent. Never applied to masks: it changes colour, not
    geometry, so labels stay valid as they are.
    """
    count, _, height, width = images.shape
    device = images.device
    rows = torch.linspace(-1.0, 1.0, height, device=device).view(1, -1, 1)
    columns = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, -1)
    offset_y = _uniform(count, -0.05, 0.05, generator, device).view(-1, 1, 1)
    offset_x = _uniform(count, -0.05, 0.05, generator, device).view(-1, 1, 1)
    radius = torch.sqrt((rows - offset_y) ** 2 + (columns - offset_x) ** 2)

    start = _uniform(count, radius_range[0], radius_range[1], generator, device).view(-1, 1, 1)
    strength = _uniform(count, strength_range[0], strength_range[1], generator, device)
    gamma = _uniform(count, gamma_range[0], gamma_range[1], generator, device).view(-1, 1, 1)
    falloff = ((radius - start) / (1.0 - start).clamp(min=1e-3)).clamp(0.0, 1.0) ** gamma
    factor = (1.0 - strength.view(-1, 1, 1) * falloff).unsqueeze(1)
    darkened = (images * factor).clamp(0.0, 1.0)
    gate = _per_sample_gate(count, probability, generator, device)
    return torch.where(gate, darkened, images)


def moco_augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """MoCo v2 view: resized crop, flip, vignette, color jitter, grayscale and blur."""
    out, _ = random_resized_crop(images, generator, scale=(0.2, 1.0))
    out = random_vignette(out, generator, probability=0.3)
    out = color_jitter(out, generator, 0.4, 0.4, 0.4, 0.1, probability=0.8)
    out = random_grayscale(out, generator, probability=0.2)
    return gaussian_blur(out, generator, probability=0.5)


def student_augment(
    images: torch.Tensor, masks: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student view: geometric transform shared with the mask, vignette and jitter.

    The vignette is photometric, so it touches the image only; the mask keeps
    the labels produced by the shared geometric transform.
    """
    out, out_masks = random_resized_crop(
        images,
        generator,
        scale=(0.5, 1.0),
        ratio=(0.8, 1.25),
        horizontal_flip=True,
        vertical_flip=True,
        masks=masks,
    )
    out = random_vignette(out, generator, probability=0.3)
    out = color_jitter(out, generator, 0.4, 0.4, 0.4, 0.1, probability=0.8)
    out = random_grayscale(out, generator, probability=0.1)
    return out, out_masks


#: T7 (docs/27): the registered darkening, applied after ``student_augment``.
DARKEN_DEFAULTS = {"probability": 0.5, "gamma_range": (1.0, 2.2),
                   "brightness_range": (0.85, 1.0), "contrast_range": (0.85, 1.0)}


def random_darken(
    images: torch.Tensor,
    generator: torch.Generator,
    eligible: torch.Tensor,
    *,
    probability: float = DARKEN_DEFAULTS["probability"],
    gamma_range: tuple[float, float] = DARKEN_DEFAULTS["gamma_range"],
    brightness_range: tuple[float, float] = DARKEN_DEFAULTS["brightness_range"],
    contrast_range: tuple[float, float] = DARKEN_DEFAULTS["contrast_range"],
) -> torch.Tensor:
    """Gamma, then brightness, then contrast towards the grey mean, on eligible samples with ``probability``.

    Photometric only: masks are never touched. Every sample draws its values
    whether or not it is eligible, so the generator advances by a fixed amount.
    With gamma >= 1 and factors <= 1 the mean luminance never goes up.
    """
    count, device = images.shape[0], images.device
    gamma = _uniform(count, *gamma_range, generator, device).view(-1, 1, 1, 1)
    brightness = _uniform(count, *brightness_range, generator, device).view(-1, 1, 1, 1)
    contrast = _uniform(count, *contrast_range, generator, device).view(-1, 1, 1, 1)
    gate = _per_sample_gate(count, probability, generator, device)
    out = images.clamp(0.0, 1.0) ** gamma * brightness
    mean = _grayscale(out).mean(dim=(1, 2, 3), keepdim=True)
    out = ((out - mean) * contrast + mean).clamp(0.0, 1.0)
    return torch.where(gate & eligible.to(device).view(-1, 1, 1, 1), out, images)
