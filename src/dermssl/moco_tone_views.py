"""Experimental mask-aligned MoCo views; no training or clinical claims.

Matches the audited helper's PIL RandomResizedCrop + horizontal flip order.
Masks follow each crop. Tone and restoration occur at the final 224 input,
before normalization. No further geometric transform is allowed downstream.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .pilot_adapters import protected_encoder_view, reference_free_qa, validate_rgb


def tone_assignment(*, seed, step, image_sha256):
    """Choose one view and one sign without labels or any mutable RNG state.

    Hash bits provide a symmetric assignment distribution, not exact balance
    in a finite sample. Record realized attempts, shifts and QA fallbacks.
    Positive bucket means -8 L*, following the historical API convention.
    """
    if (isinstance(seed, bool) or seed not in (42, 43, 44)
            or not isinstance(step, int) or isinstance(step, bool) or step < 1
            or not isinstance(image_sha256, str) or len(image_sha256) != 64
            or any(c not in "0123456789abcdef" for c in image_sha256)):
        raise ValueError("Expected seed 42/43/44, positive step and lowercase image SHA-256")
    key = f"moco-tone-v1:{seed}:{step}:{image_sha256}".encode("ascii")
    bits = hashlib.sha256(key).digest()[0]
    return bits & 1, (1 if bits & 2 else -1)


def _project_mask(mask, crop, flip):
    top, left, height, width = crop
    # Float BOX preserves nonzero area occupancy instead of thresholding a
    # bilinear mask. It is a geometric safeguard, not expert ground truth.
    image = Image.fromarray(mask.astype(np.float32)).crop(
        (left, top, left + width, top + height))
    projected = np.asarray(image.resize((224, 224), Image.Resampling.BOX)) > 0
    return np.fliplr(projected).copy() if flip else projected.copy()


def make_moco_tone_view(image, lesion_mask, valid_fov, *, shift_buckets=0):
    """Return normalized tensor, final protected mask and serializable receipt.

    Rejected parent/crop QA retains the unaltered view. Malformed inputs stop
    rather than silently omitting a training row. This function does not load
    masks, check their provenance, build batches or implement a trainer.
    """
    validate_rgb(image)
    if type(shift_buckets) is not int or shift_buckets not in (-1, 0, 1):
        raise ValueError("Expected an explicit shift in {-1, 0, 1}")
    parent_qa = reference_free_qa(image, lesion_mask, valid_fov)
    pil = Image.fromarray(image)
    crop = T.RandomResizedCrop.get_params(pil, scale=(.4, 1.), ratio=(3 / 4, 4 / 3))
    resized = TF.resized_crop(pil, *crop, [224, 224],
                              interpolation=InterpolationMode.BILINEAR, antialias=True)
    flip = bool(torch.rand(1) < .5)
    if flip:
        resized = TF.hflip(resized)
    baseline = np.asarray(resized).copy()
    lesion = _project_mask(lesion_mask, crop, flip)
    # Any outside-FOV occupancy is protected rather than made eligible skin.
    field = ~_project_mask(~valid_fov, crop, flip)
    crop_qa = reference_free_qa(baseline, lesion, field)
    if not parent_qa.qa_passed:
        crop_qa = replace(crop_qa, qa_passed=False, fallback_reason=parent_qa.fallback_reason)
    altered, protected, info = protected_encoder_view(
        baseline, crop_qa, shift_buckets=shift_buckets)
    # Restore the parent's rejection reason if crop-only checks passed.
    if not parent_qa.qa_passed:
        info["fallback_reason"] = parent_qa.fallback_reason
    tensor = TF.normalize(TF.to_tensor(Image.fromarray(altered)),
                          (.485, .456, .406), (.229, .224, .225))
    info.update({"view_profile": "moco_tone_crop_flip_final224_v1",
                 "geometry": {"crop_tlhw": list(crop), "horizontal_flip": flip},
                 "requested_shift_buckets": shift_buckets,
                 "delta_l_star": -8 * shift_buckets,
                 "parent_qa_passed": parent_qa.qa_passed,
                 "crop_qa_passed": crop_qa.qa_passed})
    return tensor, protected, info
