"""Bind verified F4clean artifacts to the published segmentation and E3 code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .f4_service import F4Service, F4ServiceError, PROTOCOL

B71_SHA256 = "a392ebe8eedcc519a6131e3930bed5ea232d4be7084010f82740e55cadbe39c1"
SEGMENTER_SHA256 = "ac87b133dd50e74a88d9137b2f1ba76b986ee176017b8b009b0caf52a00c5cc0"
EXPECTED_RECIPE = {"variant": "F4clean", "protocol": PROTOCOL, "b71_sha256": B71_SHA256,
                   "segmenter_sha256": SEGMENTER_SHA256, "feature_precision": "float32",
                   "attention": "math", "tf32": False, "hair_flag": True}


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scientific_paths(root: Path) -> None:
    if not (root / "src/dermssl/e3_render.py").is_file():
        raise F4ServiceError("canonical scientific source is missing")
    for folder in (root / "src", root / "scripts", root / "plan-segmentacion/comun/evaluacion",
                   root / "plan-segmentacion/comun/nucleo"):
        value = str(folder)
        if value not in sys.path:
            sys.path.insert(0, value)


def make_segment_mask(checkpoint: Path, source_root: Path, device):
    """Published ImageNet-normalized prediction, resized before thresholding."""
    if file_sha256(checkpoint) != SEGMENTER_SHA256:
        raise F4ServiceError("final segmenter checkpoint SHA256 mismatch")
    scientific_paths(source_root)
    import eval_student
    import c1_predict_masks
    from dermssl.c1_mask_provenance import PUBLISHED_INFERENCE

    model, resolution = eval_student.load_student(checkpoint, device)
    if resolution != PUBLISHED_INFERENCE["resolution"]:
        raise F4ServiceError("segmenter resolution differs from published inference")

    def segment(rgb: np.ndarray) -> np.ndarray:
        resized = np.asarray(Image.fromarray(rgb).resize((resolution, resolution), Image.Resampling.BILINEAR))
        probability = eval_student.predict_masks(model, resized[None], device, 1, tta=False)[0]
        original = c1_predict_masks.upsample_probabilities(probability, rgb.shape[0], rgb.shape[1])
        return original > PUBLISHED_INFERENCE["threshold"]

    return segment


def create_f4_service(backbone, model_dir: Path, source_root: Path, device) -> F4Service:
    """Construct for offline golden; serving additionally requires an approval receipt."""
    manifest = json.loads((model_dir / "F4_classifier_manifest.json").read_text())
    if any(manifest.get(key) != value for key, value in EXPECTED_RECIPE.items()):
        raise F4ServiceError("F4clean manifest does not match fixed inference recipe")
    probe_path = model_dir / "F4_probe.npz"
    if file_sha256(probe_path) != manifest.get("probe_sha256"):
        raise F4ServiceError("F4 probe SHA256 mismatch")
    with np.load(probe_path, allow_pickle=False) as data:
        artifact = {key: data[key].copy() for key in data.files}
    artifact["classes"] = artifact["classes"].tolist()
    artifact["feature_dim"] = int(artifact["feature_dim"])
    artifact["protocol"] = PROTOCOL
    scientific_paths(source_root)
    import cv2
    import e3_io
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from dermssl import e3_render, e3_variants, e3_descriptors, torch_v71

    cv2.setNumThreads(1)
    hair_fn = e3_io.make_hair_fn()
    if hair_fn is None:
        raise F4ServiceError("canonical hair detector is unavailable")

    def extract(model, images, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return torch_v71.extract_split(model, images, autocast_enabled=False, **kwargs)

    return F4Service(backbone=backbone,
                     segment_mask=make_segment_mask(model_dir / "student_r3_final_best.pt", source_root, device),
                     resize_crop=e3_io.make_resize_crop(), extract_split=extract, artifact=artifact,
                     device=device, batch_size=1, hair_fn=hair_fn,
                     renderer=e3_render.render, describer=e3_render.describe,
                     variant_builder=e3_variants.concat_features, descriptor_module=e3_descriptors)
