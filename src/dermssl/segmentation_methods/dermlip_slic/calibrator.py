"""DermLIP-Guided Superpixel Calibrator for Dermatoscopic Lesion Segmentation.

Fuses multi-scale SLIC superpixels with domain-specific Vision-Language Foundation
Models (PanDerm / DermLIP or WhyLesionCLIP) to produce spatially regularized,
uncertainty-aware pseudo-masks and boundary confidence fields.
"""
from __future__ import annotations

import os
import hashlib
import json
import subprocess
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from skimage.segmentation import slic
import torch
import torch.nn.functional as F

from dermssl.segmentation_methods.dermlip_slic.tripleprior import PeripheralRingNormalizer

DEFAULT_POS_PROMPTS = [
    "a dermatoscopic photo of a skin lesion with pigmentation",
    "melanocytic skin lesion with irregular border",
    "dark pigmented skin lesion under dermatoscopy",
    "atypical pigment network and globules in a dermatoscopic lesion",
]

DEFAULT_NEG_PROMPTS = [
    "normal healthy background human skin",
    "pale skin without lesions or marks",
    "peripheral healthy skin background",
    "homogeneous surrounding normal skin with dermatoglyphics",
]

DERMLIP_MODEL_REPOSITORY = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
DERMLIP_MODEL_REVISION = "b455d4ca3b39a742ae9a8ea7d3246216c6d4bf4f"
DERMLIP_TOKENIZER_REPOSITORY = "NeuML/pubmedbert-base-embeddings"
DERMLIP_TOKENIZER_REVISION = "b79526d6ef3645e0df4530322e266f24c829f5ef"
DERMLIP_LICENSE = "CC-BY-NC-ND-4.0"
DERMLIP_MODEL_NAME = "DermLIP PanDerm-base-w-PubMed-256"
DERMLIP_CONFIG_SOURCE_SHA256 = "0d9dcaeebcd6fe219392578f3da73ab025d471d4c7289698b56788172ddc2ba1"
DERMLIP_CHECKPOINT_SOURCE_SHA256 = "c01315cc8bda77decb7bfaaafaf4668979d065d0ded33c1cb42108cd8d56fedf"
DERMLIP_VENDOR_REPO_ROOT = "/home/jovyan/work/vendor/Derm1M"
DERMLIP_VENDOR_COMMIT = "92d8fa9d9775aaf65bf3150c998d32daf4f1d32f"
DERMLIP_VENDOR_FACTORY_SHA256 = "b6dd54bd45279c6fa50f4856fa8450c724463f1697c3c47459432698821461da"
DERMLIP_VENDOR_TOKENIZER_SHA256 = "d5ea0a1b633d74076cef9616fe0b0cc7333eefeee3656c48de4cd79f2173bbc8"


def _verify_dermlip_cache_on_disk(source: Dict[str, Any]) -> None:
    """Recheck active refs and declared files instead of trusting receipt mappings."""
    cache = source["cache_verification"]
    cache_root = os.path.realpath(str(cache.get("cache_root", "")))
    if not cache_root or not os.path.isdir(cache_root):
        raise RuntimeError("DermLIP verified HF cache directory is unavailable")
    repos = {
        "model": (DERMLIP_MODEL_REPOSITORY, DERMLIP_MODEL_REVISION),
        "tokenizer": (DERMLIP_TOKENIZER_REPOSITORY, DERMLIP_TOKENIZER_REVISION),
    }
    snapshots = {}
    for label, (repository, revision) in repos.items():
        repo_name = f"models--{repository.replace('/', '--')}"
        repo_cache = os.path.join(cache_root, repo_name)
        if not os.path.isdir(repo_cache):
            candidates = [
                os.path.join(cache_root, d) for d in os.listdir(cache_root)
                if d.lower() == repo_name.lower()
            ]
            if candidates:
                repo_cache = candidates[0]
        ref_path = os.path.join(repo_cache, "refs", "main")
        if not os.path.isfile(ref_path):
            raise RuntimeError(f"DermLIP {label} HF cache active ref is missing")
        with open(ref_path, "r", encoding="utf-8") as ref_file:
            active_ref = ref_file.read().strip().lower()
        if active_ref != revision:
            raise RuntimeError(f"DermLIP {label} HF cache active ref does not match the approved revision")
        snapshot = os.path.join(repo_cache, "snapshots", active_ref)
        if not os.path.isdir(snapshot):
            raise RuntimeError(f"DermLIP {label} HF cache snapshot is missing")
        snapshots[label] = snapshot

    expected_files = source["files"]
    for label, filename, expected_hash in (
        ("model", "open_clip_config.json", DERMLIP_CONFIG_SOURCE_SHA256),
        ("model", "open_clip_model.safetensors", DERMLIP_CHECKPOINT_SOURCE_SHA256),
    ):
        if expected_files.get(filename) != expected_hash:
            raise RuntimeError(f"DermLIP {filename} manifest hash is not approved")
        path = os.path.join(snapshots[label], filename)
        if not os.path.isfile(path):
            raise RuntimeError(f"DermLIP {label} cache file is missing: {filename}")
        with open(path, "rb") as model_file:
            actual_hash = hashlib.sha256(model_file.read()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"DermLIP {label} cache file SHA-256 mismatch: {filename}")

    tokenizer_hashes = cache.get("tokenizer_file_sha256")
    if not isinstance(tokenizer_hashes, dict) or not tokenizer_hashes:
        raise RuntimeError("DermLIP tokenizer cache hashes are missing")
    recognized_tokenizer_files = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt",
    }
    declared_tokenizer_files = {
        name for name in expected_files if os.path.basename(name).lower() in recognized_tokenizer_files
    }
    if set(tokenizer_hashes) != declared_tokenizer_files:
        raise RuntimeError("DermLIP tokenizer cache hash inventory differs from the manifest")
    actual_tokenizer_files = {
        name for name in os.listdir(snapshots["tokenizer"])
        if os.path.isfile(os.path.join(snapshots["tokenizer"], name))
        and name.lower() in recognized_tokenizer_files
    }
    if actual_tokenizer_files != {os.path.basename(name) for name in declared_tokenizer_files}:
        raise RuntimeError("DermLIP tokenizer cache files differ from the declared inventory")
    for filename, expected_hash in tokenizer_hashes.items():
        if (
            os.path.dirname(filename)
            or expected_files.get(filename) != expected_hash
            or not os.path.isfile(os.path.join(snapshots["tokenizer"], filename))
        ):
            raise RuntimeError(f"DermLIP tokenizer cache file is undeclared or missing: {filename}")
        with open(os.path.join(snapshots["tokenizer"], filename), "rb") as tokenizer_file:
            actual_hash = hashlib.sha256(tokenizer_file.read()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"DermLIP tokenizer cache SHA-256 mismatch: {filename}")


def _validate_dermlip_source_verification(model_name: str, source: Optional[Dict[str, Any]]) -> None:
    """Reject untrusted or mismatched source claims before loading the vendor library."""
    if not isinstance(source, dict):
        raise RuntimeError("DermLIP initialization requires approved source verification")
    if model_name != f"hf-hub:{DERMLIP_MODEL_REPOSITORY}":
        raise RuntimeError("DermLIP must use the approved hf-hub model repository")
    approved_identity = {
        "model_repository": DERMLIP_MODEL_REPOSITORY,
        "model_revision": DERMLIP_MODEL_REVISION,
        "tokenizer_repository": DERMLIP_TOKENIZER_REPOSITORY,
        "tokenizer_revision": DERMLIP_TOKENIZER_REVISION,
        "license": DERMLIP_LICENSE,
        "terms_acceptance_status": "user_attested",
        "terms_acceptance_verification_method": "manual_user_attestation",
        "model_source_hashes_verified": True,
        "tokenizer_source_hashes_verified": False,
        "tokenizer_bundle_integrity_verified": True,
    }
    for key, expected in approved_identity.items():
        if source.get(key) != expected:
            raise RuntimeError(f"DermLIP source verification does not match approved {key}")
    if source.get("model_name") != model_name:
        raise RuntimeError("DermLIP source verification model name does not match the requested model")
    declared_files = source.get("files")
    if not isinstance(declared_files, dict):
        raise RuntimeError("DermLIP source verification is missing its approved file hash map")
    if declared_files.get("open_clip_config.json") != DERMLIP_CONFIG_SOURCE_SHA256:
        raise RuntimeError("DermLIP source verification config hash is not approved")
    if declared_files.get("open_clip_model.safetensors") != DERMLIP_CHECKPOINT_SOURCE_SHA256:
        raise RuntimeError("DermLIP source verification weights hash is not approved")
    attestation = source.get("terms_human_attestation")
    if (
        not isinstance(attestation, dict)
        or attestation.get("repository") != DERMLIP_MODEL_REPOSITORY
        or attestation.get("model") != DERMLIP_MODEL_NAME
        or attestation.get("model_revision") != DERMLIP_MODEL_REVISION
        or attestation.get("license") != DERMLIP_LICENSE
        or attestation.get("terms_version") != DERMLIP_LICENSE
        or not isinstance(attestation.get("terms_reference"), str)
        or not attestation["terms_reference"].strip()
        or not attestation.get("accepted_by")
        or not attestation.get("accepted_on")
        or not attestation.get("attested_at")
        or not attestation.get("receipt_sha256")
        or source.get("terms_acceptance_evidence_sha256") != attestation.get("receipt_sha256")
    ):
        raise RuntimeError("DermLIP source verification lacks the approved human attestation")
    try:
        accepted_on = date.fromisoformat(str(attestation["accepted_on"]))
    except ValueError as exc:
        raise RuntimeError("DermLIP source verification acceptance date is not ISO YYYY-MM-DD") from exc
    if accepted_on.isoformat() != attestation["accepted_on"]:
        raise RuntimeError("DermLIP source verification acceptance date must use YYYY-MM-DD")
    try:
        attested_at = datetime.fromisoformat(str(attestation["attested_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError("DermLIP source verification attested_at is not valid ISO-8601") from exc
    if attested_at.tzinfo is None or attested_at.utcoffset() is None:
        raise RuntimeError("DermLIP source verification attested_at must include a timezone")
    receipt_path_value = source.get("terms_acceptance_evidence_path")
    receipt_sha256 = str(source.get("terms_acceptance_evidence_sha256", "")).lower()
    if (
        not isinstance(receipt_path_value, str)
        or not receipt_path_value
        or len(receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in receipt_sha256)
        or receipt_sha256 != str(attestation.get("receipt_sha256", "")).lower()
    ):
        raise RuntimeError("DermLIP source verification lacks a valid attestation receipt SHA-256")
    receipt_path = os.path.realpath(receipt_path_value)
    if not os.path.isfile(receipt_path) or os.path.getsize(receipt_path) == 0:
        raise RuntimeError("DermLIP attestation receipt is missing or empty")
    with open(receipt_path, "rb") as receipt_file:
        receipt_bytes = receipt_file.read()
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha256:
        raise RuntimeError("DermLIP attestation receipt SHA-256 does not match source verification")
    try:
        receipt = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("DermLIP attestation receipt is not valid structured JSON") from exc
    expected_receipt = {key: value for key, value in attestation.items() if key != "receipt_sha256"}
    if receipt != expected_receipt:
        raise RuntimeError("DermLIP attestation receipt fields do not match source verification")
    cache = source.get("cache_verification")
    if (
        not isinstance(cache, dict)
        or cache.get("verified") is not True
        or cache.get("model_ref") != DERMLIP_MODEL_REVISION
        or cache.get("tokenizer_ref") != DERMLIP_TOKENIZER_REVISION
        or cache.get("model_config_sha256") != DERMLIP_CONFIG_SOURCE_SHA256
        or cache.get("model_weights_sha256") != DERMLIP_CHECKPOINT_SOURCE_SHA256
        or cache.get("tokenizer_source_hashes_verified") is not False
        or cache.get("tokenizer_cache_integrity_verified") is not True
        or not cache.get("cache_root")
        or cache.get("cache_mutated") is not False
    ):
        raise RuntimeError("DermLIP source verification cache receipt does not match approved refs and hashes")
    tokenizer_hashes = cache.get("tokenizer_file_sha256")
    declared_files = source.get("files")
    if not isinstance(tokenizer_hashes, dict) or not tokenizer_hashes:
        raise RuntimeError("DermLIP source verification lacks tokenizer cache hashes")
    for filename, digest in tokenizer_hashes.items():
        if declared_files.get(filename) != digest:
            raise RuntimeError(f"DermLIP tokenizer source verification hash mismatch for {filename}")
    _verify_dermlip_cache_on_disk(source)


def _validate_open_clip_fork(
    open_clip_module: Any, vendor_root: Union[str, os.PathLike[str]] = DERMLIP_VENDOR_REPO_ROOT
) -> None:
    """Require the specific Derm1M vendor checkout and its PanDerm extension API."""
    vendor_root = os.path.realpath(vendor_root)
    expected_package = os.path.realpath(os.path.join(vendor_root, "src", "open_clip"))
    module_path = os.path.realpath(getattr(open_clip_module, "__file__", ""))
    try:
        is_expected_path = os.path.commonpath((expected_package, module_path)) == expected_package
    except ValueError:
        is_expected_path = False
    if not is_expected_path:
        raise RuntimeError(f"Imported open_clip is outside the expected Derm1M vendor path: {module_path}")

    git_marker = os.path.join(vendor_root, ".git")
    if not os.path.exists(git_marker):
        raise RuntimeError("Required Derm1M vendor git checkout and HEAD are unavailable")
    try:
        result = subprocess.run(
            ["git", "-C", vendor_root, "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not verify the Derm1M vendor git checkout HEAD") from exc
    if result.stdout.strip().lower() != DERMLIP_VENDOR_COMMIT:
        raise RuntimeError("Derm1M vendor commit does not match the approved pinned commit")

    try:
        status_result = subprocess.run(
            ["git", "-C", vendor_root, "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not verify the Derm1M vendor worktree status") from exc
    expected_changes = [
        " M src/open_clip/factory.py",
        " M src/open_clip/tokenizer.py",
    ]
    if status_result.stdout.splitlines() != expected_changes:
        raise RuntimeError("Derm1M vendor worktree status contains unexpected or missing changes")

    expected_file_hashes = {
        "src/open_clip/factory.py": DERMLIP_VENDOR_FACTORY_SHA256,
        "src/open_clip/tokenizer.py": DERMLIP_VENDOR_TOKENIZER_SHA256,
    }
    for relative_path, expected_hash in expected_file_hashes.items():
        source_path = os.path.join(vendor_root, relative_path)
        if not os.path.isfile(source_path):
            raise RuntimeError(f"Required Derm1M vendor source file is missing: {relative_path}")
        with open(source_path, "rb") as source_file:
            actual_hash = hashlib.sha256(source_file.read()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(f"Derm1M vendor source SHA-256 mismatch: {relative_path}")

    api_modules = [
        open_clip_module,
        getattr(open_clip_module, "factory", None),
        getattr(open_clip_module, "model", None),
    ]
    if not any(callable(getattr(module, "call_PanDerm_base_visual", None)) for module in api_modules):
        raise RuntimeError("Imported open_clip fork does not expose the required call_PanDerm_base_visual PanDerm API")


def clean_connected_components(
    binary_mask: np.ndarray,
    min_relative_ratio: float = 0.20,
    min_absolute_pixels: int = 60,
    fill_holes: bool = True,
    keep_largest_only: bool = True,
) -> np.ndarray:
    """Filter spurious isolated components, retain primary lesion, and fill internal holes."""
    from scipy import ndimage

    if not np.any(binary_mask):
        return binary_mask.astype(np.uint8)

    labeled, num_features = ndimage.label(binary_mask)
    if num_features <= 1:
        clean = binary_mask.copy()
    else:
        sizes = ndimage.sum(binary_mask, labeled, range(1, num_features + 1))
        if keep_largest_only:
            largest_label = int(np.argmax(sizes)) + 1
            clean = labeled == largest_label
        else:
            max_size = float(sizes.max())
            cutoff = max(float(min_absolute_pixels), max_size * min_relative_ratio)
            keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= cutoff]
            clean = np.isin(labeled, keep_labels)

    if fill_holes and np.any(clean):
        clean = ndimage.binary_fill_holes(clean)

    return clean.astype(np.uint8)


class DermLIPSuperpixelCalibrator:
    """Multi-scale superpixel calibrator using dermatology foundation vision-language models."""

    def __init__(
        self,
        model_name: str = "hf-hub:redlessone/DermLIP_PanDerm-base-w-PubMed-256",
        device: Optional[str] = None,
        fine_segments: int = 300,
        coarse_segments: int = 75,
        compactness: float = 20.0,
        temperature: float = 0.07,
        tau_pos: float = 0.5,
        tau_unc: float = 0.85,
        calibration_mode: str = "dense",
        use_otsu_threshold: bool = False,
        apply_field_roi: bool = True,
        filter_connected_components: bool = True,
        keep_largest_only: bool = True,
        min_component_ratio: float = 0.20,
        min_component_pixels: int = 60,
        fill_holes: bool = True,
        pos_prompts: Optional[List[str]] = None,
        neg_prompts: Optional[List[str]] = None,
        source_verification: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.source_verification = source_verification
        self.device = device or ("cuda:1" if torch.cuda.is_available() else "cpu")
        self.fine_segments = fine_segments
        self.coarse_segments = coarse_segments
        self.compactness = compactness
        self.temperature = temperature
        self.tau_pos = tau_pos
        self.tau_unc = tau_unc
        self.calibration_mode = calibration_mode
        self.use_otsu_threshold = use_otsu_threshold
        self.apply_field_roi = apply_field_roi
        self.filter_connected_components = filter_connected_components
        self.keep_largest_only = keep_largest_only
        self.min_component_ratio = min_component_ratio
        self.min_component_pixels = min_component_pixels
        self.fill_holes = fill_holes

        self.pos_prompts = pos_prompts or DEFAULT_POS_PROMPTS
        self.neg_prompts = neg_prompts or DEFAULT_NEG_PROMPTS
        self.ring_normalizer = PeripheralRingNormalizer(ring_margin_frac=0.12)

        self.model: Optional[Any] = None
        self.preprocess: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.pos_proto: Optional[torch.Tensor] = None
        self.neg_proto: Optional[torch.Tensor] = None

    def initialize_model(self) -> None:
        """Instantiate the foundation vision-language model and precompute text prototypes."""
        if self.model is not None:
            return

        is_dermlip = "DermLIP" in self.model_name or "PanDerm" in self.model_name
        if is_dermlip:
            _validate_dermlip_source_verification(self.model_name, self.source_verification)
            # The vendor fork resolves hf-hub: via refs/main and has no revision argument.
            # Force offline behavior before importing OpenCLIP so it cannot fetch mutable refs.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"

            import sys

            vendor_root = os.path.realpath(DERMLIP_VENDOR_REPO_ROOT)
            vendor_src = os.path.join(vendor_root, "src")
            vendor_package = os.path.join(vendor_src, "open_clip")
            if not os.path.isdir(vendor_package):
                raise RuntimeError(f"Required Derm1M vendor open_clip package is missing: {vendor_package}")
            if vendor_src in sys.path:
                sys.path.remove(vendor_src)
            sys.path.insert(0, vendor_src)

        import open_clip

        if is_dermlip:
            _validate_open_clip_fork(open_clip)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

        with torch.no_grad():
            pos_tokens = self._tokenize(self.pos_prompts).to(self.device)
            neg_tokens = self._tokenize(self.neg_prompts).to(self.device)
            pos_feats = self.model.encode_text(pos_tokens)
            neg_feats = self.model.encode_text(neg_tokens)

            pos_feats = F.normalize(pos_feats, dim=-1)
            neg_feats = F.normalize(neg_feats, dim=-1)

            pos_proto = pos_feats.mean(dim=0, keepdim=True)
            neg_proto = neg_feats.mean(dim=0, keepdim=True)

            self.pos_proto = F.normalize(pos_proto, dim=-1)
            self.neg_proto = F.normalize(neg_proto, dim=-1)

    def _tokenize(self, texts: List[str]) -> torch.Tensor:
        """Tokenize text prompts with safety fallback for different HuggingFace versions."""
        assert self.tokenizer is not None
        try:
            return self.tokenizer(texts)
        except AttributeError:
            if hasattr(self.tokenizer, "tokenizer"):
                encoded = self.tokenizer.tokenizer(
                    texts,
                    return_tensors="pt",
                    max_length=256,
                    padding="max_length",
                    truncation=True,
                )
                return encoded.input_ids
            raise

    def fit_labeled_prototypes(
        self,
        labeled_dataset: Any,
        max_images: int = 160,
        proto_blend: float = 0.65,
    ) -> Dict[str, Any]:
        """Calibrate foreground/background prototypes using PH2 labeled dense visual tokens.

        Closes the vision-language modality gap (DenseCLIP, CVPR 2022; TiSage, MICCAI 2026)
        by blending zero-shot text prototypes with normalized visual patch prototypes
        pooled strictly over the authorized PH2 training split.
        """
        self.initialize_model()
        if not self._can_extract_dense_features():
            return {"calibrated": False, "reason": "visual_encoder_has_no_dense_tokens"}
        assert self.model is not None and self.preprocess is not None
        assert self.pos_proto is not None and self.neg_proto is not None

        visual = self.model.visual
        fg_sum: Optional[torch.Tensor] = None
        bg_sum: Optional[torch.Tensor] = None
        fg_count = 0
        bg_count = 0
        n_samples = min(len(labeled_dataset), max_images)

        with torch.no_grad():
            for idx in range(n_samples):
                img_t, mask_t = labeled_dataset[idx]
                img_np = np.clip(
                    np.transpose(img_t.detach().cpu().numpy(), (1, 2, 0)) * 255.0,
                    0.0,
                    255.0,
                ).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                inp = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                B = inp.shape[0]

                p = visual.patch_embed(inp)
                cls_tokens = visual.cls_token.expand(B, -1, -1)
                p = torch.cat((cls_tokens, p), dim=1)
                if visual.pos_embed is not None:
                    p = p + visual.pos_embed.expand(B, -1, -1).type_as(p)
                p = visual.pos_drop(p)
                rel_pos_bias = (
                    visual.rel_pos_bias()
                    if hasattr(visual, "rel_pos_bias") and visual.rel_pos_bias is not None
                    else None
                )
                for blk in visual.blocks:
                    p = blk(p, rel_pos_bias=rel_pos_bias)
                p = visual.norm(p)
                patch_tokens = p[:, 1:, :]
                if hasattr(visual, "head") and isinstance(visual.head, torch.nn.Module):
                    dense_embeds = visual.head(patch_tokens)
                else:
                    dense_embeds = patch_tokens
                dense_embeds = F.normalize(dense_embeds, dim=-1)  # [1, N, D]

                num_patches = dense_embeds.shape[1]
                grid_size = int(round(num_patches ** 0.5))
                mask_4d = mask_t.unsqueeze(0).to(self.device).float()
                mask_grid = F.adaptive_avg_pool2d(mask_4d, (grid_size, grid_size)).view(1, -1)

                fg_sel = mask_grid[0] >= 0.70
                bg_sel = mask_grid[0] <= 0.15
                tokens = dense_embeds[0]  # [N, D]

                if bool(fg_sel.any()):
                    fg_tok = tokens[fg_sel].sum(dim=0, keepdim=True)
                    fg_sum = fg_tok if fg_sum is None else (fg_sum + fg_tok)
                    fg_count += int(fg_sel.sum().item())
                if bool(bg_sel.any()):
                    bg_tok = tokens[bg_sel].sum(dim=0, keepdim=True)
                    bg_sum = bg_tok if bg_sum is None else (bg_sum + bg_tok)
                    bg_count += int(bg_sel.sum().item())

            if fg_sum is not None and bg_sum is not None and fg_count > 0 and bg_count > 0:
                vis_pos = F.normalize(fg_sum / float(fg_count), dim=-1)
                vis_neg = F.normalize(bg_sum / float(bg_count), dim=-1)
                blend = float(np.clip(proto_blend, 0.0, 1.0))
                self.pos_proto = F.normalize(
                    (1.0 - blend) * self.pos_proto + blend * vis_pos, dim=-1
                )
                self.neg_proto = F.normalize(
                    (1.0 - blend) * self.neg_proto + blend * vis_neg, dim=-1
                )
                return {
                    "calibrated": True,
                    "images_processed": n_samples,
                    "fg_patch_tokens": fg_count,
                    "bg_patch_tokens": bg_count,
                    "proto_blend": blend,
                }

        return {"calibrated": False, "reason": "insufficient_tokens"}

    def calibrate_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        target_size: Tuple[int, int] = (224, 224),
    ) -> Dict[str, Any]:
        """Calibrate superpixels for a single dermatoscopic image."""
        self.initialize_model()
        assert self.model is not None and self.preprocess is not None
        assert self.pos_proto is not None and self.neg_proto is not None

        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        else:
            pil_img = image_input.convert("RGB")

        if pil_img.size != target_size:
            pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)

        img_np = np.array(pil_img)
        h, w, _ = img_np.shape

        # Extract multi-scale SLIC superpixels
        slic_fine = slic(
            img_np,
            n_segments=self.fine_segments,
            compactness=self.compactness,
            start_label=0,
        )
        slic_coarse = slic(
            img_np,
            n_segments=self.coarse_segments,
            compactness=self.compactness * 1.25,
            start_label=0,
        )

        if self.calibration_mode == "dense" and self._can_extract_dense_features():
            dense_prob = self._extract_dense_probability_map(pil_img, h, w)
            p_fine = self._pool_superpixels_from_dense(dense_prob, slic_fine)
            p_coarse = self._pool_superpixels_from_dense(dense_prob, slic_coarse)
        else:
            p_fine = self._calibrate_single_scale_crops(pil_img, slic_fine, h, w)
            p_coarse = self._calibrate_single_scale_crops(pil_img, slic_coarse, h, w)

        p_fused = 0.5 * (p_fine + p_coarse)

        img_rgb_norm = img_np.astype(np.float32) / 255.0
        valid_fov, _ = self._compute_fov_and_peripheral_ring(img_rgb_norm)

        # Apply field of view / dermoscope ROI mask if requested
        if self.apply_field_roi:
            try:
                from dermssl.field_roi import field_validity_mask

                roi_mask, _ = field_validity_mask(img_np)
                if np.any(valid_fov & roi_mask.astype(bool)):
                    valid_fov = valid_fov & roi_mask.astype(bool)
            except Exception:
                pass
            p_fused = np.where(valid_fov, p_fused, 0.02).astype(np.float32)

        # Information-theoretic binary Shannon entropy: H(x) in [0, 1]
        eps = 1e-7
        p_clip = np.clip(p_fused, eps, 1.0 - eps)
        entropy = -(p_clip * np.log2(p_clip) + (1.0 - p_clip) * np.log2(1.0 - p_clip))

        if self.use_otsu_threshold:
            try:
                from skimage.filters import threshold_otsu

                fov_vals = p_fused[valid_fov] if np.any(valid_fov) else p_fused.ravel()
                t_otsu = float(threshold_otsu(fov_vals))
                t_thresh = float(np.clip(t_otsu, 0.45, 0.68))
            except Exception:
                t_thresh = self.tau_pos
        else:
            t_thresh = self.tau_pos

        raw_mask = ((p_fused >= t_thresh) & valid_fov).astype(np.uint8)

        if self.filter_connected_components:
            pseudo_mask = clean_connected_components(
                raw_mask,
                min_relative_ratio=self.min_component_ratio,
                min_absolute_pixels=self.min_component_pixels,
                fill_holes=self.fill_holes,
                keep_largest_only=self.keep_largest_only,
            )
            # Attenuate probability for spurious filtered-out pixels so they stay strictly < 0.25
            p_fused = np.where((raw_mask == 1) & (pseudo_mask == 0), p_fused * 0.15, p_fused)
        else:
            pseudo_mask = raw_mask

        if self.apply_field_roi:
            p_fused = np.where(valid_fov, p_fused, 0.02).astype(np.float32)

        uncertainty_mask = (entropy >= self.tau_unc).astype(np.uint8)
        certain_mask = 1 - uncertainty_mask

        return {
            "probability_map": p_fused,
            "pseudo_mask": pseudo_mask,
            "uncertainty_map": entropy,
            "uncertainty_mask": uncertainty_mask,
            "certain_mask": certain_mask,
            "foreground_fraction": float(pseudo_mask.mean()),
            "mean_uncertainty": float(entropy.mean()),
            "slic_fine_count": int(len(np.unique(slic_fine))),
            "slic_coarse_count": int(len(np.unique(slic_coarse))),
        }

    def calibrate_batch(
        self,
        images: Union[torch.Tensor, np.ndarray, List[Union[str, np.ndarray, Image.Image]]],
        target_size: Tuple[int, int] = (224, 224),
    ) -> Dict[str, torch.Tensor]:
        """Calibrate a batch of dermatoscopic images and return stacked tensor outputs."""
        if isinstance(images, torch.Tensor):
            imgs_np = images.detach().cpu().numpy()
            if imgs_np.ndim == 4 and imgs_np.shape[1] == 3:
                imgs_np = np.transpose(imgs_np, (0, 2, 3, 1))
            if imgs_np.max() <= 1.0:
                imgs_np = np.clip(imgs_np * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                imgs_np = np.clip(imgs_np, 0.0, 255.0).astype(np.uint8)
            item_list: List[Union[str, np.ndarray, Image.Image]] = [imgs_np[i] for i in range(imgs_np.shape[0])]
        elif isinstance(images, np.ndarray) and images.ndim == 4:
            item_list = [images[i] for i in range(images.shape[0])]
        else:
            item_list = list(images)

        prob_maps, pseudo_masks, unc_maps, cert_masks = [], [], [], []
        for item in item_list:
            res = self.calibrate_image(item, target_size=target_size)
            prob_maps.append(torch.from_numpy(res["probability_map"]).unsqueeze(0).float())
            pseudo_masks.append(torch.from_numpy(res["pseudo_mask"]).unsqueeze(0).float())
            unc_maps.append(torch.from_numpy(res["uncertainty_map"]).unsqueeze(0).float())
            cert_masks.append(torch.from_numpy(res["certain_mask"]).unsqueeze(0).float())

        return {
            "probability_map": torch.stack(prob_maps, dim=0),
            "pseudo_mask": torch.stack(pseudo_masks, dim=0),
            "uncertainty_map": torch.stack(unc_maps, dim=0),
            "certain_mask": torch.stack(cert_masks, dim=0),
        }

    def _compute_fov_and_peripheral_ring(
        self, img_rgb_norm: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute valid dermoscopic FOV mask and perilesional peripheral skin ring mask."""
        from scipy import ndimage

        h, w = img_rgb_norm.shape[:2]
        valid_fov = img_rgb_norm.mean(axis=-1) > 0.08
        if not np.any(valid_fov):
            valid_fov = np.ones((h, w), dtype=bool)

        fov_interior = valid_fov.copy()
        fov_interior[0, :] = False
        fov_interior[-1, :] = False
        fov_interior[:, 0] = False
        fov_interior[:, -1] = False

        dist = ndimage.distance_transform_edt(fov_interior)
        scale = min(h, w) / 224.0
        inner_px = max(2.0, 12.0 * scale)
        outer_px = max(inner_px + 4.0, 36.0 * scale)
        ring_mask = (dist >= inner_px) & (dist <= outer_px) & valid_fov

        if int(ring_mask.sum()) < 32:
            fallback_ring = self.ring_normalizer.create_peripheral_ring_mask(h, w) & valid_fov
            ring_mask = fallback_ring if np.any(fallback_ring) else valid_fov

        return valid_fov, ring_mask

    def _can_extract_dense_features(self) -> bool:
        """Check whether the loaded visual encoder supports direct spatial token extraction."""
        if self.model is None:
            return False
        visual = getattr(self.model, "visual", None)
        if visual is None or not isinstance(visual, torch.nn.Module):
            return False
        return hasattr(visual, "patch_embed") and hasattr(visual, "blocks") and hasattr(visual, "norm")

    def _extract_dense_probability_map(self, pil_img: Image.Image, h: int, w: int) -> np.ndarray:
        """Extract continuous dense semantic probability map calibrated against peripheral skin ring."""
        visual = self.model.visual
        img_t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        B = img_t.shape[0]

        with torch.no_grad():
            p = visual.patch_embed(img_t)
            cls_tokens = visual.cls_token.expand(B, -1, -1)
            p = torch.cat((cls_tokens, p), dim=1)
            if visual.pos_embed is not None:
                p = p + visual.pos_embed.expand(B, -1, -1).type_as(p)
            p = visual.pos_drop(p)
            rel_pos_bias = visual.rel_pos_bias() if hasattr(visual, "rel_pos_bias") and visual.rel_pos_bias is not None else None
            for blk in visual.blocks:
                p = blk(p, rel_pos_bias=rel_pos_bias)
            p = visual.norm(p)
            patch_tokens = p[:, 1:, :]  # [B, num_patches, embed_dim]

            if hasattr(visual, "head") and isinstance(visual.head, torch.nn.Module):
                dense_embeds = visual.head(patch_tokens)
            else:
                dense_embeds = patch_tokens
            dense_embeds = F.normalize(dense_embeds, dim=-1)

            num_patches = dense_embeds.shape[1]
            grid_size = int(round(num_patches ** 0.5))
            dense_embeds = dense_embeds.view(B, grid_size, grid_size, -1)

            feat_grid = dense_embeds.permute(0, 3, 1, 2)
            feat_up = F.interpolate(
                feat_grid,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
            feat_up = F.normalize(feat_up, dim=-1).cpu().numpy().astype(np.float32)

            sim_pos = (dense_embeds @ self.pos_proto.T).squeeze(-1)
            sim_neg = (dense_embeds @ self.neg_proto.T).squeeze(-1)
            raw_diff_grid = sim_pos - sim_neg

            raw_diff = F.interpolate(
                raw_diff_grid.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy().astype(np.float32)

        pil_resized = pil_img if pil_img.size == (w, h) else pil_img.resize((w, h), Image.Resampling.BILINEAR)
        img_u8 = np.asarray(pil_resized, dtype=np.uint8)
        img_rgb_norm = img_u8.astype(np.float32) / 255.0
        valid_fov, ring_mask = self._compute_fov_and_peripheral_ring(img_rgb_norm)

        luminance = (
            0.299 * img_rgb_norm[..., 0]
            + 0.587 * img_rgb_norm[..., 1]
            + 0.114 * img_rgb_norm[..., 2]
        )
        darkness = 1.0 - luminance
        ring_lum = luminance[ring_mask]

        # Conditional ring contamination check:
        # In huge lesions (>= 50% area) extending to the boundary, dark lesion tissue invades the ring.
        # We detect this invasion only when:
        # 1. Ring has substantial luminance dispersion (std > 0.10)
        # 2. The valid FOV is dominated by a dark core (> 40% area darker than ring upper quartile)
        # In normal lesions (< 50% area), the ring is homogeneous skin, so we preserve the full ring untouched.
        proto_mask = ring_mask
        if ring_lum.size >= 32 and float(np.std(ring_lum)) > 0.10:
            lum_q75 = float(np.percentile(ring_lum, 75.0))
            dark_core_frac = float((luminance[valid_fov] < (lum_q75 - 0.15)).mean())
            if dark_core_frac > 0.40:
                lum_thresh = float(np.percentile(ring_lum, 60.0))
                clean_ring = ring_mask & (luminance >= lum_thresh)
                if int(clean_ring.sum()) >= 16:
                    proto_mask = clean_ring

        # 1. Peripheral ring semantic prototype distance & polarity-aligned VLM contrast
        ring_feats = feat_up[proto_mask]
        ring_proto = ring_feats.mean(axis=0) if ring_feats.shape[0] > 0 else feat_up.mean(axis=(0, 1))
        ring_proto = ring_proto / (np.linalg.norm(ring_proto) + 1e-8)
        vlm_dist = 1.0 - (feat_up @ ring_proto)
        r_vdist = vlm_dist[ring_mask]
        vdist_med = float(np.median(vlm_dist[proto_mask])) if int(proto_mask.sum()) > 0 else float(np.median(vlm_dist))
        vdist_std = max(float(np.std(r_vdist)) if r_vdist.size > 0 else 0.05, 0.04)
        z_vdist = (vlm_dist - vdist_med) / (vdist_std + 1e-5)

        ring_vlm = raw_diff[ring_mask]
        ring_vlm_mean = float(np.median(raw_diff[proto_mask])) if int(proto_mask.sum()) > 0 else float(np.mean(raw_diff))
        ring_vlm_std = max(float(np.std(ring_vlm)) if ring_vlm.size > 0 else float(np.std(raw_diff)), 0.03)
        z_vtext = (raw_diff - ring_vlm_mean) / (ring_vlm_std + 1e-5)
        fg_Semantic_candidates = vlm_dist > float(np.median(vlm_dist[valid_fov]))
        if np.any(fg_Semantic_candidates) and float(np.mean(z_vtext[fg_Semantic_candidates])) < 0.0:
            z_vtext = -z_vtext

        z_vlm = 0.70 * z_vdist + 0.30 * z_vtext

        # 2. Intrinsic color/darkness lesion contrast relative to patient's peripheral skin ring
        skin_rgb_u8, _ = self.ring_normalizer.extract_skin_baseline(img_u8)
        ring_rgb = img_rgb_norm[proto_mask]
        ring_rgb_mean = np.median(ring_rgb, axis=0) if ring_rgb.shape[0] > 0 else (skin_rgb_u8.astype(np.float32) / 255.0)

        ring_dark = darkness[ring_mask]
        dark_mean = float(np.median(darkness[proto_mask])) if int(proto_mask.sum()) > 0 else float(np.median(darkness))
        dark_std = max(float(np.std(ring_dark)) if ring_dark.size > 0 else 0.05, 0.04)
        z_dark = (darkness - dark_mean) / (dark_std + 1e-5)

        color_dist = np.linalg.norm(img_rgb_norm - ring_rgb_mean.reshape(1, 1, 3), axis=-1)
        ring_cdist = color_dist[ring_mask]
        cdist_mean = float(np.median(color_dist[proto_mask])) if int(proto_mask.sum()) > 0 else 0.0
        cdist_std = max(float(np.std(ring_cdist)) if ring_cdist.size > 0 else 0.05, 0.04)
        z_chroma = (color_dist - cdist_mean) / (cdist_std + 1e-5)

        z_intrinsic = 0.70 * z_dark + 0.30 * z_chroma
        z_combined = 0.50 * z_vlm + 0.50 * z_intrinsic

        ring_comb = z_combined[ring_mask]
        comb_mean = float(np.median(z_combined[proto_mask])) if int(proto_mask.sum()) > 0 else 0.0
        comb_std = max(float(np.std(ring_comb)) if ring_comb.size > 0 else 1.0, 0.30)
        z_diff = (z_combined - comb_mean) / (comb_std + 1e-5)

        # Strict contrast margin: peripheral healthy skin (z_diff <= 0) maps to < 0.08
        logits_cal = np.clip((z_diff - 1.30) * 2.0, -20.0, 20.0)
        prob_map = 1.0 / (1.0 + np.exp(-logits_cal))
        prob_map = np.where(valid_fov, prob_map, 0.02)

        return prob_map.astype(np.float32)

    def _pool_superpixels_from_dense(
        self, dense_prob: np.ndarray, segments: np.ndarray
    ) -> np.ndarray:
        """Pool continuous dense probability map strictly within each superpixel boundary."""
        pooled = np.zeros_like(dense_prob, dtype=np.float32)
        unique_labels = np.unique(segments)
        for lab in unique_labels:
            mask = segments == lab
            pooled[mask] = float(dense_prob[mask].mean())
        return pooled

    def _calibrate_single_scale_crops(
        self, pil_img: Image.Image, segments: np.ndarray, h: int, w: int
    ) -> np.ndarray:
        """Legacy crop-based projection for models without dense ViT hooks."""
        unique_labels = np.unique(segments)
        crops = []
        for lab in unique_labels:
            mask = segments == lab
            ys, xs = np.where(mask)
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            h_pad = max(4, int((ymax - ymin) * 0.15))
            w_pad = max(4, int((xmax - xmin) * 0.15))
            c_ymin = max(0, ymin - h_pad)
            c_ymax = min(h, ymax + h_pad)
            c_xmin = max(0, xmin - w_pad)
            c_xmax = min(w, xmax + w_pad)

            crop = pil_img.crop((c_xmin, c_ymin, c_xmax, c_ymax))
            crops.append(self.preprocess(crop))

        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            img_feats = self.model.encode_image(batch)
            img_feats = F.normalize(img_feats, dim=-1)

            sim_pos = (img_feats @ self.pos_proto.T).squeeze(-1)
            sim_neg = (img_feats @ self.neg_proto.T).squeeze(-1)

            logits = torch.stack([sim_neg / self.temperature, sim_pos / self.temperature], dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        prob_map = np.zeros((h, w), dtype=np.float32)
        for i, lab in enumerate(unique_labels):
            prob_map[segments == lab] = probs[i]

        return prob_map
