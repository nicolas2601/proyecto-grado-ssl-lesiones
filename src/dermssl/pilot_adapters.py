"""Offline pilot adapters. Technical QA does not certify clinical skin purity.

The segmenter reproduces the evaluated raw-v8 recipe: standard hair removal,
224-square input, 8-bit probability resize and threshold 0.5. It does not invoke
legacy SLIC, component selection, hole filling or micro-lesion rescue.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .field_roi import field_validity_mask
from .pipeline_segment_circle import (
    DermLIPCirclePipeline,
    PipelineMaskResult,
    MIN_HEALTHY_SKIN_FRACTION,
    MIN_LESION_AREA_FRACTION,
    MAX_LESION_AREA_FRACTION,
)
from .segmentation_methods.common.hair import remove_hair
from .segmentation_methods.dermlip_slic.trainer import build_segmentation_model


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def hub_source_hash(directory):
    """Identify the offline model implementation, excluding mutable caches."""
    root = Path(directory).resolve()
    paths = [root / "hubconf.py", *sorted((root / "dinov2").rglob("*.py"))]
    entries = {}
    for path in paths:
        if not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Invalid offline model source tree")
        entries[path.relative_to(root).as_posix()] = file_sha256(path)
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_verified_weights(path, expected_sha256):
    """Reject drift before safe deserialization; never fall back to pickle mode."""
    path = Path(path)
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or file_sha256(path) != expected_sha256):
        raise ValueError("Checkpoint SHA-256 does not match the reviewed receipt")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Checkpoint SHA-256 changed while loading")
    return payload


def validate_rgb(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2):
        raise ValueError("Expected a nonempty uint8 RGB image")


def reference_free_qa(image, lesion_mask, valid_fov):
    """Check geometry and skin availability, NOT reference-mask contamination.

    Coverage thresholds reuse the existing pipeline's technical heuristics.
    An all-black/empty field fails closed; rejected images stay in the cohort
    and are classified unchanged. No expert mask or diagnostic label is read.
    """
    validate_rgb(image)
    for mask in (lesion_mask, valid_fov):
        if not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != image.shape[:2]:
            raise ValueError("Expected boolean masks matching the RGB image")
    lesion = lesion_mask.copy()
    healthy = valid_fov & ~lesion
    coverage, skin_coverage = float(lesion.mean()), float(healthy.mean())
    reason = None
    if not valid_fov.any() or not image[valid_fov].any():
        reason = "empty_valid_field"
    elif coverage < MIN_LESION_AREA_FRACTION:
        reason = "lesion_too_small"
    elif coverage > MAX_LESION_AREA_FRACTION:
        reason = "lesion_too_large"
    elif skin_coverage < MIN_HEALTHY_SKIN_FRACTION:
        reason = "insufficient_healthy_skin"
    return PipelineMaskResult(lesion, valid_fov.copy(), healthy, coverage,
                              skin_coverage, reason is None, reason)


def shift_healthy_skin(image, mask_result, *, shift_buckets):
    """Reuse bounded L* mechanics, preserving predicted lesion and full frame.

    Bucket terminology is historical in the underlying API, not a clinical
    Fitzpatrick estimate. This is a fixed perturbation, not a validated tone
    normalizer. QA failures are explicit identity fallbacks.
    """
    validate_rgb(image)
    # Recompute structural checks rather than trusting caller-supplied qa_passed.
    checked = reference_free_qa(image, mask_result.lesion_mask, mask_result.valid_fov_mask)
    if not mask_result.qa_passed:
        return image.copy(), False
    if not isinstance(shift_buckets, int) or isinstance(shift_buckets, bool) or abs(shift_buckets) > 2:
        raise ValueError("Explicit integer L* shift steps in [-2, 2] required")
    return DermLIPCirclePipeline().apply_circle_shift(image, checked, shift_buckets)


class FrozenResNetSegmenter:
    """Explicit ResNet34 teacher, pinned to a known checkpoint and run ID."""

    def __init__(self, checkpoint, expected_sha256, expected_run_id, *, device="cpu"):
        payload = load_verified_weights(checkpoint, expected_sha256)
        if not isinstance(payload, dict) or payload.get("checkpoint_component") != "teacher":
            raise ValueError("Expected an explicit teacher checkpoint component")
        identity = payload.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("Missing checkpoint run identity")
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (payload.get("run_id") != expected_run_id or not expected_run_id
                or payload.get("identity_fingerprint") != fingerprint):
            raise ValueError("Checkpoint identity differs from the reviewed run")
        if payload.get("backbone") != "resnet34" or identity.get("config", {}).get("backbone") != "resnet34":
            raise ValueError("Expected the reviewed ResNet34 architecture")
        implementation_sha = file_sha256(inspect.getfile(build_segmentation_model))
        if identity.get("code", {}).get("source_sha256", {}).get("trainer.py") != implementation_sha:
            raise ValueError("Segmenter implementation SHA differs from the training identity")
        self.model = build_segmentation_model(backbone="resnet34", pretrained=False)
        self.model.load_state_dict(payload["teacher_state_dict"], strict=True)
        self.model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.provenance = {"checkpoint_sha256": expected_sha256, "run_id": expected_run_id,
                           "identity_fingerprint": fingerprint, "backbone": "resnet34",
                           "implementation_sha256": implementation_sha,
                           "profile": "raw_v8_standard_hair_224_prob8_threshold050"}

    @torch.inference_mode()
    def segment(self, image):
        validate_rgb(image)
        height, width = image.shape[:2]
        resized = np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR))
        cleaned, hair_fraction = remove_hair(resized, shaped=False)
        batch = torch.from_numpy(cleaned.astype(np.float32).transpose(2, 0, 1) / 255).unsqueeze(0).to(self.device)
        logits = self.model(batch)
        if logits.shape != (1, 1, 224, 224) or not torch.isfinite(logits).all():
            raise ValueError("Segmenter produced invalid logits")
        probability = logits.sigmoid()[0, 0].cpu().numpy()
        restored = np.asarray(Image.fromarray((probability * 255).astype(np.uint8)).resize(
            (width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        # Raw lesion mask is kept intact; FOV only limits the skin-color pool.
        lesion = restored >= 0.5
        field, field_metadata = field_validity_mask(image)
        result = reference_free_qa(image, lesion, field)
        metadata = {**self.provenance, "hair_fraction": float(hair_fraction),
                    "field": field_metadata, "technical_qa_passed": result.qa_passed,
                    "technical_qa_policy": {"min_lesion_fraction": MIN_LESION_AREA_FRACTION,
                                            "max_lesion_fraction": MAX_LESION_AREA_FRACTION,
                                            "min_healthy_skin_fraction": MIN_HEALTHY_SKIN_FRACTION,
                                            "coverage_denominator": "full_frame"},
                    "clinical_skin_purity_verified": False, "reference_mask_used": False,
                    "fallback_reason": result.fallback_reason}
        return result, metadata


class FrozenCLSAdapter:
    """Nicolás' no-TTA DINOv2 CLS export, with strict plain-state loading."""

    def __init__(self, model, checkpoint, expected_sha256, *, device="cpu"):
        state = load_verified_weights(checkpoint, expected_sha256)
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(x) for x in state.values()):
            raise ValueError("Expected a plain tensor state_dict for the SSL encoder")
        model.load_state_dict(state, strict=True)
        self.model = model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.provenance = {"checkpoint_sha256": expected_sha256, "features": "x_norm_clstoken",
                           "dimensions": 768, "resize_shorter": 256, "interpolation": "bicubic",
                           "center_crop": 224, "tta": False}

    @classmethod
    def from_local_hub(cls, hub_directory, checkpoint, expected_sha256, *, expected_source_sha256, device="cpu"):
        hub = Path(hub_directory).resolve()
        if not (hub / "hubconf.py").is_file():
            raise ValueError("An existing offline DINOv2 hub checkout is required")
        actual_source = hub_source_hash(hub)
        if actual_source != expected_source_sha256:
            raise ValueError("Offline encoder source SHA differs from the reviewed checkout")
        model = torch.hub.load(str(hub), "dinov2_vitb14", source="local", pretrained=False)
        adapter = cls(model, checkpoint, expected_sha256, device=device)
        if hub_source_hash(hub) != actual_source:
            raise ValueError("Offline encoder source changed while loading")
        adapter.provenance.update({"implementation_sha256": actual_source, "torch_version": str(torch.__version__)})
        return adapter

    @staticmethod
    def preprocess(images):
        tensors = []
        for image in images:
            validate_rgb(image)
            height, width = image.shape[:2]
            size = (256, int(256 * height / width)) if width <= height else (int(256 * width / height), 256)
            resized = Image.fromarray(image).resize(size, Image.Resampling.BICUBIC)
            left, top = int(round((resized.width - 224) / 2)), int(round((resized.height - 224) / 2))
            crop = np.asarray(resized.crop((left, top, left + 224, top + 224))).copy()
            tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255
            tensors.append((tensor - torch.tensor([.485, .456, .406])[:, None, None])
                           / torch.tensor([.229, .224, .225])[:, None, None])
        if not tensors:
            raise ValueError("At least one RGB image is required")
        return torch.stack(tensors)

    @torch.inference_mode()
    def encode(self, images):
        batch = self.preprocess(images).to(self.device)
        features = self.model.forward_features(batch)["x_norm_clstoken"]
        if features.shape != (len(images), 768) or not torch.isfinite(features).all():
            raise ValueError("Expected finite CLS features [N, 768]")
        return features.cpu()
