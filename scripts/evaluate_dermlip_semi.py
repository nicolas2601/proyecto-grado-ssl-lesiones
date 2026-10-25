"""Evaluation script for Semi-Supervised DermLIP segmentation models on ISIC 2018 Task 1 holdout.

Computes:
1. Overlap metrics: Mean Dice (F1), Mean IoU (Jaccard).
2. Diagnostic metrics: Sensitivity (Recall), Specificity, Accuracy.
3. Boundary metrics: 95th Percentile Symmetric Hausdorff Distance (HD95 in pixels).
4. Subgroup breakdown: Full benchmark (2,594 cases) vs. provisional clean cohort candidates (2,011 cases).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from scipy import ndimage
import torch

# Ensure src is on sys.path
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dermssl.segmentation_methods.dermlip_slic.trainer import DermLIPSegmentationUNet


def validate_manifest_case_ids(records: list[dict], label: str) -> set[str]:
    """Require nonempty unique case IDs so manifests cannot silently collapse rows."""
    ids = [str(record.get("image_id", "")).strip() for record in records]
    if not records or any(not image_id for image_id in ids):
        raise ValueError(f"{label} manifest must contain nonempty image_id values")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} manifest contains duplicate image_id values")
    return set(ids)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_fingerprint(identity: dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_scored_cases(scored_count: int, candidate_count: int) -> None:
    if candidate_count <= 0 or scored_count <= 0:
        raise ValueError(f"No evaluation cases were scored (scored={scored_count}, candidates={candidate_count})")


def require_fresh_output_directory(path: Path) -> Path:
    """Refuse to overwrite artifacts from a previous evaluation run."""
    path = Path(path).resolve()
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"Evaluation output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"Refusing nonempty evaluation output directory: {path}")
    return path


def validate_reference_mask(
    mask_values: np.ndarray, minimum_coverage: float = 0.01
) -> tuple[np.ndarray, float]:
    """Validate binary expert masks without imposing an arbitrary upper lesion-size limit."""
    mask_values = np.asarray(mask_values)
    if not set(np.unique(mask_values).tolist()).issubset({0, 1, 255}):
        raise ValueError("reference_mask_not_binary")
    binary_mask = mask_values > 127
    coverage = float(np.mean(binary_mask))
    if not binary_mask.any() or coverage < minimum_coverage:
        raise ValueError("reference_mask_empty_or_below_minimum_coverage")
    return binary_mask, coverage


def load_verified_clean_ids(manifest_path: Path, receipt_path: Path) -> tuple[set[str], str]:
    """Load a clean cohort only when its exact bytes match the recorded receipt digest."""
    manifest_path = Path(manifest_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("--clean-only was requested, but verified clean cohort manifest/receipt could not be resolved")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = str(receipt.get("clean_eval_manifest_sha256", "")).lower()
    if len(expected) != 64 or digest != expected:
        raise ValueError("Clean evaluation manifest SHA-256 does not match receipt")
    recorded_path = receipt.get("clean_eval_manifest_path")
    if recorded_path:
        recorded = Path(recorded_path)
        if not recorded.is_absolute():
            recorded = receipt_path.parent / recorded
        if recorded.resolve() != manifest_path:
            raise ValueError("Clean evaluation manifest path does not match receipt")
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = doc.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Verified clean cohort manifest must contain records")
    ids = [str(record.get("image_id", "")).strip() for record in records]
    if any(not image_id for image_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Verified clean cohort manifest must contain nonempty unique image IDs")
    return set(ids), digest


def load_full_pair_manifest(
    manifest_path: Path, receipt_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Validate the receipt and one-to-one structure of an ISIC2018 full-pair manifest."""
    manifest_path = Path(manifest_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError("Full-pair evaluation manifest and receipt must both exist")

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Full-pair manifest or receipt is unreadable") from exc

    if not isinstance(document, dict) or document.get("schema") != "isic2018_task1_full_pair_manifest.v1":
        raise ValueError("Unsupported full-pair evaluation manifest schema")
    if not isinstance(receipt, dict) or receipt.get("schema") != "isic2018_task1_full_pair_manifest_receipt.v1":
        raise ValueError("Unsupported full-pair evaluation receipt schema")

    digest = sha256_file(manifest_path)
    if receipt.get("manifest_sha256") != digest:
        raise ValueError("Full-pair evaluation manifest SHA-256 does not match receipt")
    recorded_path = receipt.get("manifest_path")
    if not isinstance(recorded_path, str) or Path(recorded_path).resolve() != manifest_path:
        raise ValueError("Full-pair evaluation manifest path does not match receipt")
    if receipt.get("status") != document.get("status"):
        raise ValueError("Full-pair evaluation manifest status does not match receipt")

    records = document.get("records")
    membership = receipt.get("official_zip_membership")
    if (
        not isinstance(records, list)
        or len(records) != 2594
        or document.get("count") != 2594
        or not isinstance(membership, dict)
        or membership.get("expected_count") != 2594
    ):
        raise ValueError("Full-pair evaluation must contain exactly 2594 records and receipt expected_count")

    seen_ids: set[str] = set()
    seen_image_paths: set[str] = set()
    seen_mask_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Full-pair manifest record {index} must be an object")
        image_id = record.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("Full-pair manifest must contain nonempty image_id values")
        if image_id in seen_ids:
            raise ValueError("Full-pair manifest contains duplicate image_id values")
        seen_ids.add(image_id)

        for kind, seen_paths in (("image", seen_image_paths), ("mask", seen_mask_paths)):
            asset = record.get(kind)
            if not isinstance(asset, dict):
                raise ValueError(f"Full-pair manifest record {image_id} is missing its {kind} object")
            relative_path = asset.get("relative_path")
            checksum = asset.get("sha256")
            width, height = asset.get("width"), asset.get("height")
            if (
                not isinstance(relative_path, str)
                or not relative_path.strip()
                or Path(relative_path).is_absolute()
                or not isinstance(checksum, str)
                or len(checksum) != 64
                or any(char not in "0123456789abcdef" for char in checksum.lower())
                or not isinstance(width, int)
                or isinstance(width, bool)
                or width <= 0
                or not isinstance(height, int)
                or isinstance(height, bool)
                or height <= 0
            ):
                raise ValueError(f"Full-pair manifest record {image_id} has invalid {kind} metadata")
            normalized_path_obj = Path(relative_path)
            if ".." in normalized_path_obj.parts:
                raise ValueError(f"Full-pair manifest record {image_id} has unsafe {kind} path")
            normalized_path = normalized_path_obj.as_posix()
            if normalized_path in seen_paths:
                raise ValueError(f"Full-pair manifest contains duplicate {kind} paths")
            seen_paths.add(normalized_path)

        image, mask = record["image"], record["mask"]
        if image["width"] != mask["width"] or image["height"] != mask["height"]:
            raise ValueError(f"Full-pair image/mask dimensions differ for {image_id}")

    return document, digest, receipt


def require_full_pair_manifest_authorized(
    document: dict[str, Any], receipt: dict[str, Any], *, evaluation_unit: str = "image"
) -> None:
    """Admit only reviewed image-level scoring, without claiming patient independence."""
    if evaluation_unit != "image":
        raise ValueError("Full-pair ISIC2018 evaluation supports image-level scoring only")
    if (
        document.get("verified_for_use") is not True
        or receipt.get("verified_for_use") is not True
        or document.get("evaluation_unit") != "image"
        or receipt.get("evaluation_unit") != "image"
        or receipt.get("status") == "preparation_pending_independent_review_and_new_run_identity"
    ):
        raise ValueError("Full-pair manifest is preparation-only and not authorized for metric scoring")


def validate_full_pair_assets(
    records: list[dict[str, Any]], image_root: Path, mask_root: Path
) -> None:
    """Validate every paired source asset before scoring to preserve the 2,594-case denominator."""
    for record in records:
        image_id = record["image_id"]
        image_path = (Path(image_root) / record["image"]["relative_path"]).resolve()
        mask_path = (Path(mask_root) / record["mask"]["relative_path"]).resolve()
        for path, asset, kind in (
            (image_path, record["image"], "image"),
            (mask_path, record["mask"], "mask"),
        ):
            if not path.is_file():
                raise ValueError(
                    f"Full-pair evaluation requires all 2594 image/mask assets; missing {kind} for {image_id}"
                )
            if sha256_file(path) != asset["sha256"].lower():
                raise ValueError(f"Full-pair {kind} SHA-256 mismatch for {image_id}")
            try:
                with Image.open(path) as source:
                    source.load()
                    width, height = source.size
                    if kind == "mask":
                        validate_reference_mask(
                            np.asarray(source.convert("L")), minimum_coverage=0.0
                        )
            except Exception as exc:
                raise ValueError(f"Full-pair {kind} is unreadable for {image_id}") from exc
            if width != asset["width"] or height != asset["height"]:
                raise ValueError(f"Full-pair {kind} dimensions differ from manifest for {image_id}")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def refine_diffuse_mask_with_slic(
    img_rgb: np.ndarray,
    prob_map: np.ndarray,
    valid_fov: np.ndarray,
    n_segments: int = 300,
    compactness: float = 12.0,
    saturation_threshold: float = 0.75,
) -> np.ndarray:
    """Refine diffuse/saturated probabilities with local SLIC superpixel contrast.

    When a U-Net is globally saturated (> 75% coverage or low dynamic range),
    pixel-wise thresholding floods the field. SLIC superpixels group local coherent
    color in CIE-Lab. By measuring luminance and color contrast against the peripheral
    healthy skin, the pigmented core is decoupled from diffuse surrounding erythema.
    """
    from skimage.color import rgb2lab
    from skimage.segmentation import slic

    h, w = img_rgb.shape[:2]
    raw_binary = (prob_map >= 0.5) & valid_fov
    if raw_binary.mean() < saturation_threshold:
        return raw_binary

    segments = slic(
        img_rgb,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        mask=valid_fov,
    )
    n_segs = int(segments.max())
    if n_segs <= 1:
        return raw_binary

    lab = rgb2lab(img_rgb)
    l_chan = lab[..., 0]

    border_mask = np.zeros((h, w), dtype=bool)
    m_h, m_w = int(h * 0.12), int(w * 0.12)
    border_mask[:m_h, :] = True
    border_mask[-m_h:, :] = True
    border_mask[:, :m_w] = True
    border_mask[:, -m_w:] = True
    bg_sample_mask = border_mask & valid_fov

    if bg_sample_mask.sum() > 50:
        bg_l = float(np.median(l_chan[bg_sample_mask]))
    else:
        bg_l = float(np.percentile(l_chan[valid_fov], 80))

    sp_scores = np.zeros(n_segs)
    cy, cx = h / 2.0, w / 2.0
    diag = np.hypot(h, w) / 2.0

    for s_idx in range(1, n_segs + 1):
        sp_mask = segments == s_idx
        if not sp_mask.any():
            continue

        sp_p = float(prob_map[sp_mask].mean())
        sp_l = float(l_chan[sp_mask].mean())

        contrast = max(0.0, bg_l - sp_l)
        y_c, x_c = ndimage.center_of_mass(sp_mask)
        d_norm = np.hypot(y_c - cy, x_c - cx) / (diag + 1e-7)
        spatial_prior = np.exp(-((d_norm / 0.40) ** 2))

        contrast_factor = 1.0 - np.exp(-max(0.0, contrast) / 10.0)
        score = sp_p * contrast_factor * spatial_prior
        sp_scores[s_idx - 1] = score

    pos_scores = sp_scores[sp_scores > 0]
    if len(pos_scores) > 0:
        th_cut = float(np.percentile(pos_scores, 45))
        active_segs = np.where(sp_scores >= th_cut)[0] + 1
        refined_binary = np.isin(segments, active_segs) & valid_fov
    else:
        refined_binary = raw_binary

    return refined_binary


def select_best_connected_component(
    pred_binary: np.ndarray,
    prob_orig: np.ndarray,
    img_rgb: np.ndarray,
    center_prior: bool = True,
) -> np.ndarray:
    """Select the most plausible lesion connected component.

    Penalizes synthetic calibration stickers (blue/cyan/green or neon orange/yellow),
    dampens raw area using square root, and scores by high central confidence.
    """
    labeled_mask, num_features = ndimage.label(pred_binary)
    if num_features <= 1:
        return pred_binary

    if not center_prior:
        sizes = ndimage.sum(pred_binary, labeled_mask, range(1, num_features + 1))
        largest_label = int(np.argmax(sizes)) + 1
        return labeled_mask == largest_label

    h_p, w_p = pred_binary.shape
    cy, cx = h_p / 2.0, w_p / 2.0
    diag = np.sqrt(h_p**2 + w_p**2) / 2.0
    sizes = ndimage.sum(pred_binary, labeled_mask, range(1, num_features + 1))
    coms = ndimage.center_of_mass(pred_binary, labeled_mask, range(1, num_features + 1))

    hsv = None
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        import cv2
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    scores = []
    for i in range(num_features):
        comp_mask_i = (labeled_mask == (i + 1))
        y_i, x_i = coms[i]
        d_norm = np.sqrt((y_i - cy)**2 + (x_i - cx)**2) / (diag + 1e-7)

        color_penalty = 1.0
        if hsv is not None:
            comp_rgb = img_rgb[comp_mask_i]
            comp_hsv = hsv[comp_mask_i]
            mean_r, mean_g, mean_b = comp_rgb.mean(axis=0)
            mean_s = comp_hsv[:, 1].mean()
            mean_v = comp_hsv[:, 2].mean()

            # Synthetic blue/cyan/teal/green
            if mean_b > mean_r or mean_g > (mean_r * 1.05):
                color_penalty *= 0.0001
            # Neon bright stickers (orange/yellow/magenta)
            elif mean_v > 190 and mean_s > 150:
                color_penalty *= 0.001

        mean_p = float(prob_orig[comp_mask_i].mean())
        size_factor = np.sqrt(float(sizes[i]))
        center_factor = np.exp(-((d_norm / 0.35)**2))
        prob_factor = mean_p**2

        score = size_factor * center_factor * prob_factor * color_penalty
        scores.append(score)

    best_label = int(np.argmax(scores)) + 1
    return (labeled_mask == best_label)


def compute_hausdorff95(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Compute the symmetric 95th percentile Hausdorff Distance in pixels."""
    pred_b = np.asarray(prediction, dtype=bool)
    truth_b = np.asarray(truth, dtype=bool)

    if not pred_b.any() or not truth_b.any():
        return math.nan

    def surface(mask: np.ndarray) -> np.ndarray:
        return mask & ~ndimage.binary_erosion(mask, border_value=0)

    surf_pred = surface(pred_b)
    surf_truth = surface(truth_b)

    if not surf_pred.any() or not surf_truth.any():
        return math.nan

    # Exact surface-to-surface Euclidean distance transforms
    dist_to_surf_truth = ndimage.distance_transform_edt(~surf_truth)
    dist_to_surf_pred = ndimage.distance_transform_edt(~surf_pred)

    dists_p2t = dist_to_surf_truth[surf_pred]
    dists_t2p = dist_to_surf_pred[surf_truth]

    combined = np.concatenate([dists_p2t, dists_t2p])
    return float(np.percentile(combined, 95))


def compute_binary_metrics(
    prediction: np.ndarray, truth: np.ndarray, compute_hd: bool = False
) -> Dict[str, float]:
    """Compute standard clinical and morphological segmentation metrics for a single sample."""
    pred = np.asarray(prediction > 0, dtype=bool)
    gt = np.asarray(truth > 0, dtype=bool)

    tp = np.count_nonzero(pred & gt)
    tn = np.count_nonzero(~pred & ~gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)

    eps = 1e-7
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    res = {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }

    if compute_hd:
        res["hd95"] = compute_hausdorff95(pred, gt)

    return res


def compute_bootstrap_ci(
    values: List[float], n_bootstraps: int = 1000, ci_level: float = 0.95, seed: int = 42
) -> Dict[str, Any]:
    """Compute non-parametric bootstrap mean and 95% confidence interval across resamples."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return {
            "mean": None,
            "std": None,
            "ci_lower": None,
            "ci_upper": None,
            "ci_level": ci_level,
            "n_bootstraps": 0,
            "n_observations": 0,
        }
    if len(arr) == 1:
        v = float(arr[0])
        return {
            "mean": v,
            "std": 0.0,
            "ci_lower": v,
            "ci_upper": v,
            "ci_level": ci_level,
            "n_bootstraps": 1,
            "n_observations": 1,
        }

    rng = np.random.RandomState(seed)
    n = len(arr)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha = (1.0 - ci_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100.0))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci_level,
        "n_bootstraps": n_bootstraps,
        "n_observations": n,
    }


def summarize_metric_aggregates(
    metrics_accumulator: Dict[str, List[float]],
    *,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[Dict[str, Optional[float]], Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Summarize finite observations only and retain a denominator per metric."""
    summary_metrics: Dict[str, Optional[float]] = {}
    bootstrap_ci: Dict[str, Dict[str, Any]] = {}
    denominators: Dict[str, int] = {}
    for name, values in metrics_accumulator.items():
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        count = int(len(finite_values))
        denominators[name] = count
        summary_metrics[f"mean_{name}"] = float(np.mean(finite_values)) if count else None
        summary_metrics[f"std_{name}"] = float(np.std(finite_values)) if count else None
        bootstrap_ci[name] = compute_bootstrap_ci(
            finite_values.tolist(), n_bootstraps=n_bootstraps, seed=seed
        )
    return summary_metrics, bootstrap_ci, denominators


def load_reference_cases(
    path: Path, metric_keys: List[str]
) -> tuple[Dict[str, Dict[str, Optional[float]]], Dict[str, Any]]:
    """Load a reference cases JSONL strictly and return content provenance."""
    requested_path = Path(path)
    resolved_path = requested_path.resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"reference cases file does not exist: {resolved_path}")

    raw_bytes = resolved_path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"reference cases file is not valid UTF-8 JSONL: {resolved_path}") from exc

    lines = text.splitlines()
    if not lines:
        raise ValueError(f"reference cases file is empty JSONL: {resolved_path}")

    def reject_nonstandard_json_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant {token}")

    cases: Dict[str, Dict[str, Optional[float]]] = {}
    seen_ids: set[str] = set()
    skipped_ids: set[str] = set()
    skipped_records = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"reference cases contains empty JSONL line {line_number}")
        try:
            entry = json.loads(line, parse_constant=reject_nonstandard_json_constant)
        except ValueError as exc:
            raise ValueError(
                f"reference cases contains invalid JSONL at line {line_number}"
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(f"reference cases line {line_number} must contain a JSON object")

        image_id = entry.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip() or image_id != image_id.strip():
            raise ValueError(
                f"reference cases line {line_number} must have a nonempty image_id without surrounding whitespace"
            )
        if image_id in seen_ids:
            raise ValueError(f"reference cases contains duplicate image_id: {image_id}")
        seen_ids.add(image_id)

        status = entry.get("status", "ok")
        if status == "skipped":
            skipped_ids.add(image_id)
            skipped_records += 1
            continue
        if status != "ok":
            raise ValueError(f"reference cases line {line_number} has unsupported status: {status}")

        metrics: Dict[str, Optional[float]] = {}
        for key in metric_keys:
            if key not in entry:
                continue
            value = entry[key]
            if value is None:
                metrics[key] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(
                    f"reference cases line {line_number} must have a numeric metric {key!r} or null"
                )
            metrics[key] = float(value)
        if not metrics:
            raise ValueError(
                f"reference cases line {line_number} has no parseable requested metric values"
            )
        cases[image_id] = metrics

    provenance = {
        "path_requested": str(path),
        "path_resolved": str(resolved_path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "total_lines": len(lines),
        "valid_records": len(cases),
        "unique_ids": len(seen_ids),
        "skipped_records": skipped_records,
        "_skipped_image_ids": sorted(skipped_ids),
    }
    return cases, provenance


_METRIC_IMPROVEMENT_DIRECTIONS = {
    "dice": "higher",
    "iou": "higher",
    "sensitivity": "higher",
    "specificity": "higher",
    "accuracy": "higher",
    "hd95": "lower",
}


def _validate_metric_case_mapping(
    cases: Dict[str, Dict[str, Any]], label: str
) -> None:
    if not isinstance(cases, dict):
        raise ValueError(f"{label} cases must be a mapping of image_id to metric mapping")
    for image_id, metrics in cases.items():
        if not isinstance(image_id, str) or not image_id.strip() or image_id != image_id.strip():
            raise ValueError(f"{label} cases must have nonempty image_id values without surrounding whitespace")
        if not isinstance(metrics, dict):
            raise ValueError(f"{label} case {image_id} must contain a metric mapping")


def compute_paired_bootstrap(
    sample_a: Dict[str, Dict[str, float]],
    sample_b: Dict[str, Dict[str, float]],
    metric_keys: List[str],
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    *,
    reference_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute descriptive paired bootstrap intervals: Delta = Method_A - Method_B.

    Args:
        sample_a: Dict mapping image_id -> dict of metric values for Method A (proposed).
        sample_b: Dict mapping image_id -> dict of metric values for Method B (reference baseline).
        metric_keys: List of metrics to compare (e.g. ['dice', 'iou']).
        n_bootstraps: Number of paired bootstrap resamples (default: 1000).
        ci_level: Confidence level (default: 0.95).
        seed: Random seed for reproducibility.

    Returns:
        Versioned descriptive contrast with per-metric denominators and exclusions.
    """
    _validate_metric_case_mapping(sample_a, "sample_a")
    _validate_metric_case_mapping(sample_b, "sample_b")
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps must be positive")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be between 0 and 1")

    common_ids = sorted(set(sample_a.keys()).intersection(set(sample_b.keys())))
    skipped_reference_ids = set(
        (reference_provenance or {}).get("_skipped_image_ids", [])
    )
    skipped_common_ids = len(set(sample_a).intersection(skipped_reference_ids))
    sample_a_only_ids = len(set(sample_a) - set(sample_b))
    rng = np.random.RandomState(seed)
    results: Dict[str, Any] = {
        "schema_version": 2,
        "method": "paired_percentile_bootstrap",
        "estimand": "mean_delta=A-B",
        "evaluation_unit": "image_id",
        "interval_interpretation": "descriptive",
        "patient_independence_accredited": False,
        "ci_level": ci_level,
        "n_bootstraps_requested": n_bootstraps,
        "seed": seed,
        "common_cases": len(common_ids),
        "common_ids": len(common_ids) + skipped_common_ids,
        "unique_case_counts": {
            "sample_a": len(sample_a),
            "sample_b": len(sample_b),
        },
        "unmatched_case_counts": {
            "sample_a_only": sample_a_only_ids - skipped_common_ids,
            "sample_a_reference_skipped": skipped_common_ids,
            "sample_b_only": len(set(sample_b) - set(sample_a)),
        },
        "metrics": {},
    }
    if reference_provenance is not None:
        public_reference_provenance = {
            key: value
            for key, value in reference_provenance.items()
            if not key.startswith("_")
        }
        results["reference_cases_provenance"] = {
            **public_reference_provenance,
            "common_ids": len(common_ids) + skipped_common_ids,
            "common_scored_ids": len(common_ids),
            "common_skipped_ids": skipped_common_ids,
            "unmatched_ids": results["unmatched_case_counts"],
        }
    alpha = (1.0 - ci_level) / 2.0

    for key in metric_keys:
        pairs = []
        exclusions = {
            label: {
                reason: 0
                for reason in (
                    "missing_metric_key",
                    "none",
                    "nan",
                    "positive_infinity",
                    "negative_infinity",
                )
            }
            for label in ("sample_a", "sample_b")
        }
        for cid in common_ids:
            values = {}
            valid = True
            for label, sample in (("sample_a", sample_a), ("sample_b", sample_b)):
                case_metrics = sample[cid]
                if key not in case_metrics:
                    exclusions[label]["missing_metric_key"] += 1
                    valid = False
                    continue
                value = case_metrics[key]
                if value is None:
                    exclusions[label]["none"] += 1
                    valid = False
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                    raise ValueError(f"Metric {key!r} for {label} case {cid} must be numeric or null")
                numeric_value = float(value)
                if math.isnan(numeric_value):
                    exclusions[label]["nan"] += 1
                    valid = False
                elif math.isinf(numeric_value):
                    reason = "positive_infinity" if numeric_value > 0 else "negative_infinity"
                    exclusions[label][reason] += 1
                    valid = False
                else:
                    values[label] = numeric_value
            if valid:
                pairs.append((values["sample_a"], values["sample_b"]))

        metric_result: Dict[str, Any] = {
            "status": "ok" if len(pairs) >= 2 else "insufficient_pairs",
            "n_pairs": len(pairs),
            "excluded_pairs": len(common_ids) - len(pairs),
            "exclusions": exclusions,
            "improvement_direction": _METRIC_IMPROVEMENT_DIRECTIONS.get(key, "unspecified"),
            "n_bootstraps_effective": 0,
        }
        if pairs:
            deltas = np.asarray([a - b for a, b in pairs], dtype=np.float64)
            metric_result["mean_delta"] = float(np.mean(deltas))
        else:
            deltas = np.asarray([], dtype=np.float64)
            metric_result["mean_delta"] = None

        metric_result["ci_level"] = ci_level
        metric_result["ci_lower"] = None
        metric_result["ci_upper"] = None
        if len(pairs) >= 2:
            boot_deltas = np.empty(n_bootstraps, dtype=np.float64)
            for bootstrap_index in range(n_bootstraps):
                indices = rng.choice(len(deltas), size=len(deltas), replace=True)
                boot_deltas[bootstrap_index] = np.mean(deltas[indices])
            finite_boot_deltas = boot_deltas[np.isfinite(boot_deltas)]
            metric_result["n_bootstraps_effective"] = int(len(finite_boot_deltas))
            if len(finite_boot_deltas):
                metric_result["ci_lower"] = float(np.percentile(finite_boot_deltas, alpha * 100.0))
                metric_result["ci_upper"] = float(
                    np.percentile(finite_boot_deltas, (1.0 - alpha) * 100.0)
                )
                direction = metric_result["improvement_direction"]
                if direction == "higher":
                    metric_result["interval_favors_a_descriptively"] = (
                        metric_result["ci_lower"] > 0.0
                    )
                elif direction == "lower":
                    metric_result["interval_favors_a_descriptively"] = (
                        metric_result["ci_upper"] < 0.0
                    )

        results["metrics"][key] = metric_result

    return results


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    parser.add_argument(
        "--checkpoint-component",
        choices=("student", "teacher"),
        required=True,
        help="Exact checkpoint component to evaluate.",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        required=True,
        help="Path to evaluation image manifest (e.g. evaluation.manifest.json).",
    )
    parser.add_argument(
        "--full-pair-receipt",
        type=Path,
        default=None,
        help="Receipt for an ISIC2018 full-pair manifest; defaults to <manifest>.receipt.json.",
    )
    parser.add_argument(
        "--evaluation-unit",
        choices=("image", "patient"),
        default="image",
        help="Evaluation unit; the full-pair ISIC2018 manifest supports image-level scoring only.",
    )
    parser.add_argument(
        "--references-manifest",
        type=Path,
        required=True,
        help="Path to references manifest with expert masks (e.g. references.manifest.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="New or empty directory for report and case metrics; nonempty outputs are never overwritten.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device ('auto', 'cpu', 'cuda:1', 'cuda:0').",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binarization probability threshold (default: 0.5).",
    )
    parser.add_argument(
        "--hausdorff",
        action="store_true",
        help="Calculate 95%% Hausdorff distance (HD95).",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save predicted binary PNG masks to out-dir/predicted_masks/.",
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Filter to the receipt-bound provisional cohort; does not establish independent holdout status.",
    )
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        default=None,
        help="Path to the exact clean cohort manifest whose digest is recorded in --clean-receipt.",
    )
    parser.add_argument(
        "--clean-receipt",
        type=Path,
        default=None,
        help="Receipt JSON with clean_eval_manifest_sha256 and, when available, clean_eval_manifest_path.",
    )
    parser.add_argument(
        "--reference-cases",
        type=Path,
        default=None,
        help="Optional strict JSONL path for a descriptive paired percentile bootstrap contrast.",
    )
    parser.add_argument(
        "--hair-removal",
        action="store_true",
        help="Apply black-hat morphological detection + Telea inpainting before model inference.",
    )
    parser.add_argument(
        "--hair-shaped",
        action="store_true",
        help="Keep only elongated hair components during hair mask creation.",
    )
    parser.add_argument(
        "--vignette-mask",
        action="store_true",
        help="Apply circular Field of View (FOV) vignette detection to mask black camera corners.",
    )
    parser.add_argument(
        "--slic-contrast-refinement",
        "--tisage-slic",
        dest="slic_contrast_refinement_enabled",
        action="store_true",
        help="Apply local SLIC superpixel contrast refinement; --tisage-slic remains a deprecated alias.",
    )
    parser.add_argument(
        "--largest-component",
        action="store_true",
        help="Keep only the largest connected component of the predicted lesion mask.",
    )
    parser.add_argument(
        "--center-prior",
        action="store_true",
        help="Weight connected components by proximity to image center (avoids selecting peripheral border artifacts).",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill internal lacunae in the predicted mask (bridges flash glare and central hypopigmentation).",
    )
    parser.add_argument(
        "--adaptive-hair-rescue",
        action="store_true",
        help="Apply directional hair removal when hair fraction exceeds 18 percent.",
    )
    parser.add_argument(
        "--micro-lesion-rescue",
        action="store_true",
        help="Fallback to micro-lesion candidate when predicted mask area is under 0.2 percent.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples for rapid validation.",
    )
    return parser.parse_args(argv)


def load_model(checkpoint_path: Path, device: str, component: str) -> torch.nn.Module:
    """Load exactly the requested model component from a checkpoint."""
    if component not in {"student", "teacher"}:
        raise ValueError("Checkpoint component must be 'student' or 'teacher'")
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = DermLIPSegmentationUNet(in_channels=3, num_classes=1, base_channels=16)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or not ckpt.get("run_id") or not ckpt.get("identity_fingerprint"):
        raise ValueError("Checkpoint is missing a provenance-bound run identity")
    checkpoint_identity = ckpt.get("identity")
    if not isinstance(checkpoint_identity, dict) or identity_fingerprint(checkpoint_identity) != ckpt["identity_fingerprint"]:
        raise ValueError("Checkpoint run identity fingerprint is invalid")

    state_key = f"{component}_state_dict"
    state_dict = ckpt.get(state_key)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"Requested checkpoint component '{component}' is missing")
    recorded_component = ckpt.get("checkpoint_component")
    if recorded_component is not None and recorded_component != component:
        raise ValueError(
            f"Checkpoint component is '{recorded_component}', not requested '{component}'"
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    eval_manifest_path = Path(args.eval_manifest).resolve()
    ref_manifest_path = Path(args.references_manifest).resolve()
    paired_metric_keys = ["dice", "iou", "sensitivity", "specificity", "accuracy"]
    if getattr(args, "hausdorff", False):
        paired_metric_keys.append("hd95")
    reference_cases = None
    reference_cases_provenance = None
    reference_cases_path = getattr(args, "reference_cases", None)
    if reference_cases_path is not None:
        reference_cases, reference_cases_provenance = load_reference_cases(
            reference_cases_path, paired_metric_keys
        )

    eval_doc = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    full_pair_receipt = None
    is_full_pair_manifest = (
        isinstance(eval_doc, dict)
        and eval_doc.get("schema") == "isic2018_task1_full_pair_manifest.v1"
    )
    if is_full_pair_manifest:
        receipt_path = getattr(args, "full_pair_receipt", None)
        if receipt_path is None:
            receipt_path = eval_manifest_path.with_name(f"{eval_manifest_path.stem}.receipt.json")
        eval_doc, _, full_pair_receipt = load_full_pair_manifest(eval_manifest_path, receipt_path)
        evaluation_unit = getattr(args, "evaluation_unit", "image")
        require_full_pair_manifest_authorized(
            eval_doc, full_pair_receipt, evaluation_unit=evaluation_unit
        )
        if getattr(args, "max_samples", None) not in {None, 2594}:
            raise ValueError("Full-pair evaluation requires the fixed 2594-case denominator")
        validate_full_pair_assets(
            eval_doc["records"], eval_manifest_path.parent, ref_manifest_path.parent
        )

    out_dir = require_fresh_output_directory(args.out_dir)
    device = args.device
    if device == "auto":
        device = (
            "cuda:1"
            if (torch.cuda.is_available() and torch.cuda.device_count() > 1)
            else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

    LOGGER.info("Loading model from %s on device: %s", args.checkpoint, device)
    checkpoint_component = getattr(args, "checkpoint_component", None)
    if checkpoint_component not in {"student", "teacher"}:
        raise ValueError("An explicit --checkpoint-component student|teacher is required")
    model = load_model(args.checkpoint, device, checkpoint_component)
    checkpoint_metadata = torch.load(Path(args.checkpoint).resolve(), map_location="cpu", weights_only=False)

    out_dir.mkdir(parents=True, exist_ok=True)

    masks_out_dir = None
    if args.save_masks:
        masks_out_dir = out_dir / "predicted_masks"
        masks_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse evaluation manifests and resolve base directories for relative paths
    eval_dir = eval_manifest_path.parent
    ref_dir = ref_manifest_path.parent

    if is_full_pair_manifest:
        eval_records = [
            {
                "image_id": record["image_id"],
                "image_path": record["image"]["relative_path"],
                "image_sha256": record["image"]["sha256"],
            }
            for record in eval_doc["records"]
        ]
        ref_records = {
            record["image_id"]: {
                "image_id": record["image_id"],
                "mask_path": record["mask"]["relative_path"],
                "mask_sha256": record["mask"]["sha256"],
            }
            for record in eval_doc["records"]
        }
    else:
        ref_doc = json.loads(ref_manifest_path.read_text(encoding="utf-8"))
        ref_records = {r["image_id"]: r for r in ref_doc.get("records", [])}
        eval_records = eval_doc.get("records", [])
    eval_ids = validate_manifest_case_ids(eval_records, "evaluation")
    if is_full_pair_manifest:
        ref_ids = validate_manifest_case_ids(list(ref_records.values()), "reference")
        if eval_ids != ref_ids:
            raise ValueError("Full-pair evaluation image and mask IDs must match exactly")
    else:
        ref_ids = validate_manifest_case_ids(ref_doc.get("records", []), "reference")
    if not eval_ids & ref_ids:
        raise ValueError("Evaluation and reference manifests have no common image IDs")
    input_candidate_count = len(eval_records)
    clean_manifest_digest = None

    if args.clean_only:
        receipt_path = getattr(args, "clean_receipt", None)
        if not args.clean_manifest or not receipt_path:
            raise FileNotFoundError(
                "--clean-only was requested, but verified clean cohort manifest and digest receipt are both required."
            )
        clean_ids, clean_manifest_digest = load_verified_clean_ids(args.clean_manifest, receipt_path)
        eval_records = [r for r in eval_records if r["image_id"] in clean_ids]
        if not eval_records:
            raise ValueError(
                f"--clean-only filtered out all records! Clean cohort has {len(clean_ids)} IDs, but none matched the evaluation manifest."
            )
        LOGGER.info(
            "Clean-only filter applied: %d candidate records retained (from %d clean IDs)",
            len(eval_records),
            len(clean_ids),
        )

    if args.max_samples is not None:
        eval_records = eval_records[:args.max_samples]

    total_samples = len(eval_records)
    LOGGER.info("Evaluating on %d test samples (clean_only=%s)...", total_samples, args.clean_only)

    cases_log_path = out_dir / "cases.jsonl"
    cases_stream = cases_log_path.open("w", encoding="utf-8")

    metrics_accumulator: Dict[str, List[float]] = {
        "dice": [],
        "iou": [],
        "sensitivity": [],
        "specificity": [],
        "accuracy": [],
    }
    if args.hausdorff:
        metrics_accumulator["hd95"] = []

    evaluated_records: List[str] = []
    evaluated_cases_dict: Dict[str, Dict[str, float]] = {}
    skipped_records: List[Dict[str, str]] = []

    def skip_or_abort(image_id: str, reason: str) -> None:
        skipped_records.append({"image_id": image_id, "reason": reason})
        cases_stream.write(json.dumps({"image_id": image_id, "status": "skipped", "reason": reason}) + "\n")
        if is_full_pair_manifest:
            cases_stream.close()
            raise ValueError(
                "Full-pair evaluation aborted to preserve the 2594-case denominator: "
                f"{reason} for {image_id}"
            )

    start_time = time.time()
    for idx, rec in enumerate(eval_records):
        image_id = rec["image_id"]
        raw_img_p = Path(rec["image_path"])
        image_path = raw_img_p if raw_img_p.is_absolute() else (eval_dir / raw_img_p).resolve()

        if image_id not in ref_records:
            LOGGER.warning("Missing reference mask for image_id: %s; skipping.", image_id)
            skip_or_abort(image_id, "missing_reference_mask")
            continue

        ref_record = ref_records[image_id]
        raw_mask_p = Path(ref_record["mask_path"])
        mask_path = raw_mask_p if raw_mask_p.is_absolute() else (ref_dir / raw_mask_p).resolve()
        if not image_path.is_file() or not mask_path.is_file():
            LOGGER.warning("Image or mask file missing for %s; skipping.", image_id)
            reason = "image_file_missing" if not image_path.is_file() else "mask_file_missing"
            skip_or_abort(image_id, reason)
            continue

        image_sha256 = sha256_file(image_path)
        reference_mask_sha256 = sha256_file(mask_path)
        expected_image_sha256 = rec.get("image_sha256")
        expected_mask_sha256 = ref_record.get("mask_sha256")
        if expected_image_sha256 and expected_image_sha256.lower() != image_sha256:
            reason = "evaluation_image_sha256_mismatch"
            skip_or_abort(image_id, reason)
            continue
        if expected_mask_sha256 and expected_mask_sha256.lower() != reference_mask_sha256:
            reason = "reference_mask_sha256_mismatch"
            skip_or_abort(image_id, reason)
            continue

        # Load image and expert mask with explicit per-case failure accounting.
        try:
            with Image.open(image_path) as image_source:
                img_pil = image_source.convert("RGB")
            with Image.open(mask_path) as mask_source:
                mask_values = np.asarray(mask_source.convert("L"))
            orig_w, orig_h = img_pil.size
            if mask_values.shape != (orig_h, orig_w):
                raise ValueError("image_mask_geometry_mismatch")
            gt_arr, gt_fraction = validate_reference_mask(
                mask_values, minimum_coverage=0.0 if is_full_pair_manifest else 0.01
            )
        except Exception as exc:
            reason = str(exc) if isinstance(exc, ValueError) else f"decode_error:{type(exc).__name__}"
            skip_or_abort(image_id, reason)
            continue

        # Preprocess input (resize to 224x224 for ViT / UNet)
        input_pil = img_pil.resize((224, 224), Image.Resampling.BILINEAR)
        arr_u8 = np.asarray(input_pil)
        hair_fraction = 0.0
        if getattr(args, "hair_removal", False):
            if getattr(args, "adaptive_hair_rescue", False):
                from dermssl.segmentation_methods.common.hair import remove_hair, remove_hair_directional
                arr_u8_std, hair_frac = remove_hair(arr_u8, shaped=getattr(args, "hair_shaped", False))
                if hair_frac > 0.18:
                    arr_u8, hair_fraction = remove_hair_directional(arr_u8)
                else:
                    arr_u8, hair_fraction = arr_u8_std, hair_frac
            else:
                from dermssl.segmentation_methods.common.hair import remove_hair
                arr_u8, hair_fraction = remove_hair(arr_u8, shaped=getattr(args, "hair_shaped", False))
        img_arr = arr_u8.astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.transpose(img_arr, (2, 0, 1))).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_t)
            prob_224 = torch.sigmoid(logits).squeeze().cpu().numpy()

        # Resize predicted probability map back to original image resolution
        prob_pil = Image.fromarray((prob_224 * 255).astype(np.uint8), mode="L")
        prob_orig = np.array(
            prob_pil.resize((orig_w, orig_h), Image.Resampling.BILINEAR), dtype=np.float32
        ) / 255.0

        pred_binary = prob_orig >= args.threshold

        if getattr(args, "vignette_mask", False):
            from dermssl.field_roi import field_validity_mask
            valid_fov, _ = field_validity_mask(np.asarray(img_pil))
            pred_binary = pred_binary & valid_fov
        else:
            valid_fov = np.ones((orig_h, orig_w), dtype=bool)

        if getattr(
            args,
            "slic_contrast_refinement_enabled",
            getattr(args, "tisage_slic", False),
        ):
            pred_binary = refine_diffuse_mask_with_slic(
                np.asarray(img_pil), prob_orig, valid_fov
            )

        if getattr(args, "largest_component", False):
            pred_binary = select_best_connected_component(
                pred_binary,
                prob_orig,
                np.asarray(img_pil),
                center_prior=getattr(args, "center_prior", False),
            )

        if getattr(args, "fill_holes", False) and pred_binary.any():
            pred_binary = ndimage.binary_fill_holes(pred_binary)

        if getattr(args, "micro_lesion_rescue", False) and pred_binary.mean() < 0.002:
            from dermssl.segmentation_methods.common.rescue import rescue_micro_lesion_candidate
            micro_cand = rescue_micro_lesion_candidate(prob_orig, valid_fov, min_prob_peak=0.35)
            if micro_cand is not None:
                pred_binary = micro_cand
                if getattr(args, "fill_holes", False):
                    pred_binary = ndimage.binary_fill_holes(pred_binary)

        if masks_out_dir is not None:
            pred_png = Image.fromarray((pred_binary * 255).astype(np.uint8), mode="L")
            pred_png.save(masks_out_dir / f"{image_id}_prediction.png")

        sample_metrics = compute_binary_metrics(pred_binary, gt_arr, compute_hd=args.hausdorff)

        case_entry = {
            "image_id": image_id,
            "status": "ok",
            "image_sha256": image_sha256,
            "reference_mask_sha256": reference_mask_sha256,
            "reference_foreground_fraction": gt_fraction,
            **({"hair_inpaint_fraction": hair_fraction} if getattr(args, "hair_removal", False) else {}),
            **sample_metrics,
        }
        cases_stream.write(json.dumps(case_entry) + "\n")
        evaluated_records.append(image_id)
        evaluated_cases_dict[image_id] = sample_metrics

        for k in metrics_accumulator:
            val = sample_metrics.get(k)
            if val is not None and not math.isnan(val):
                metrics_accumulator[k].append(val)

        if (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
            elapsed = time.time() - start_time
            LOGGER.info("Progress: %d / %d (%.1f%%) — %.1f s", idx + 1, total_samples, (idx + 1)/total_samples * 100, elapsed)

    cases_stream.close()
    duration = time.time() - start_time
    try:
        require_scored_cases(len(evaluated_records), total_samples)
    except ValueError:
        failed_report = {
            "status": "failed_no_cases_scored",
            "checkpoint": str(args.checkpoint),
            "checkpoint_component": checkpoint_component,
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).resolve().read_bytes()).hexdigest(),
            "input_candidates_before_clean_filter": input_candidate_count,
            "total_candidates": total_samples,
            "total_evaluated": len(evaluated_records),
            "total_skipped": len(skipped_records),
            "skip_reasons": {
                reason: sum(item["reason"] == reason for item in skipped_records)
                for reason in sorted({item["reason"] for item in skipped_records})
            },
            "clean_only": args.clean_only,
            "clean_manifest_sha256": clean_manifest_digest,
            "holdout_status": "provisional_disjoint_candidates" if args.clean_only else "independence_not_verified",
            "patient_independence_accredited": False,
            "duration_seconds": round(duration, 2),
        }
        (out_dir / "evaluation_report.json").write_text(json.dumps(failed_report, indent=2), encoding="utf-8")
        raise

    summary_metrics, bootstrap_ci_95, metric_denominators = summarize_metric_aggregates(
        metrics_accumulator, n_bootstraps=1000
    )

    # Optional descriptive paired bootstrap contrast against reference cases
    paired_comparison = None
    if reference_cases is not None:
        LOGGER.info(
            "Computing descriptive paired bootstrap contrast against reference: %s",
            reference_cases_provenance["path_resolved"],
        )
        paired_comparison = compute_paired_bootstrap(
            evaluated_cases_dict,
            reference_cases,
            metric_keys=list(metrics_accumulator.keys()),
            n_bootstraps=1000,
            reference_provenance=reference_cases_provenance,
        )

    report = {
        "status": "completed",
        "checkpoint": str(args.checkpoint),
        "checkpoint_component": checkpoint_component,
        "checkpoint_sha256": sha256_file(Path(args.checkpoint).resolve()),
        "checkpoint_run_id": checkpoint_metadata["run_id"],
        "checkpoint_identity_fingerprint": checkpoint_metadata["identity_fingerprint"],
        "evaluation_manifest_sha256": sha256_file(eval_manifest_path),
        "reference_manifest_sha256": sha256_file(ref_manifest_path),
        "total_candidates": len(eval_records),
        "input_candidates_before_clean_filter": input_candidate_count,
        "total_evaluated": len(evaluated_records),
        "total_skipped": len(skipped_records),
        "skip_reasons": {
            reason: sum(item["reason"] == reason for item in skipped_records)
            for reason in sorted({item["reason"] for item in skipped_records})
        },
        "skipped_cases": skipped_records,
        "metrics": summary_metrics,
        "metric_denominators": metric_denominators,
        **{f"n_{name}": count for name, count in metric_denominators.items()},
        "bootstrap_ci_95": bootstrap_ci_95,
        "paired_bootstrap_comparison": paired_comparison,
        "slic_contrast_refinement_enabled": getattr(
            args,
            "slic_contrast_refinement_enabled",
            getattr(args, "tisage_slic", False),
        ),
        "clean_only": args.clean_only,
        "clean_manifest_sha256": clean_manifest_digest,
        "clean_receipt_hash_match": bool(args.clean_only and clean_manifest_digest),
        "clean_receipt_cryptographic_authenticity_verified": False,
        "holdout_status": "provisional_disjoint_candidates" if args.clean_only else "independence_not_verified",
        "evaluation_unit": "image",
        "patient_independence_accredited": False,
        "patient_independence_claim": "not_made",
        "hair_removal": getattr(args, "hair_removal", False),
        "hair_shaped": getattr(args, "hair_shaped", False),
        "vignette_mask": getattr(args, "vignette_mask", False),
        "largest_component": getattr(args, "largest_component", False),
        "center_prior": getattr(args, "center_prior", False),
        "fill_holes": getattr(args, "fill_holes", False),
        "threshold": args.threshold,
        "duration_seconds": round(duration, 2),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report_path = out_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Evaluation complete! Report written to %s", report_path)
    LOGGER.info(
        "Final Results — Mean Dice: %.4f (n=%d, 95%% CI [%.4f, %.4f]), Mean IoU: %.4f (n=%d, 95%% CI [%.4f, %.4f])",
        summary_metrics["mean_dice"],
        metric_denominators["dice"],
        bootstrap_ci_95["dice"]["ci_lower"],
        bootstrap_ci_95["dice"]["ci_upper"],
        summary_metrics["mean_iou"],
        metric_denominators["iou"],
        bootstrap_ci_95["iou"]["ci_lower"],
        bootstrap_ci_95["iou"]["ci_upper"],
    )
    if paired_comparison and "dice" in paired_comparison["metrics"]:
        d_info = paired_comparison["metrics"]["dice"]
        LOGGER.info(
            "Descriptive paired Delta Dice: %s (bootstrap interval [%s, %s], n_pairs=%d, interval_favors_a=%s)",
            d_info["mean_delta"],
            d_info["ci_lower"],
            d_info["ci_upper"],
            d_info["n_pairs"],
            d_info.get("interval_favors_a_descriptively", "unspecified"),
        )
    if "mean_hd95" in summary_metrics and summary_metrics["mean_hd95"] is not None:
        LOGGER.info(
            "HD95: %.2f px (n=%d, 95%% CI [%.2f, %.2f])",
            summary_metrics["mean_hd95"],
            metric_denominators["hd95"],
            bootstrap_ci_95.get("hd95", {}).get("ci_lower", 0.0),
            bootstrap_ci_95.get("hd95", {}).get("ci_upper", 0.0),
        )

    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    evaluate_dataset(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
