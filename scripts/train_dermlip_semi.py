"""Command-line training script for Semi-Supervised Skin Lesion Segmentation.

Trains a Teacher-Student architecture using:
1. Certified labeled dataset: PH2 (160 train / 40 val) with expert ground-truth masks.
2. Unlabeled dermatoscopic pool: ISIC 2019 (or subsample) with weak/strong augmentations.
3. Multi-scale semantic guidance: DermLIP (PanDerm ViT) dense token matching + SLIC pooling.
4. Entropy-gated consistency regularization (kappa = ln(2)) and EMA teacher updates.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance
from scipy.fft import dctn
import torch
from torch.utils.data import DataLoader, Dataset


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert RGB PIL Image to normalized float tensor [3, H, W] in [0, 1]."""
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_error_message(exc: Exception, *, max_length: int = 500) -> str:
    """Return a bounded diagnostic with common credential forms redacted."""
    message = str(exc).replace("\r", " ").replace("\n", " ")
    message = re.sub(
        r"(?i)(?<![A-Za-z0-9])(?:[A-Z]:[\\/]|\\\\)[^\s\"'<>|,;]+",
        "[LOCAL_PATH]",
        message,
    )
    message = re.sub(
        r"(?<![\w:/])/[A-Za-z0-9._~-]+(?:/[A-Za-z0-9._~-]+)*",
        "[LOCAL_PATH]",
        message,
    )
    message = re.sub(r"(?i)\b(hf_[A-Za-z0-9._-]+)", "[REDACTED]", message)
    message = re.sub(
        r"(?i)\bauthorization\s*:\s*Bearer\s+[^\s,;]+",
        "Authorization=[REDACTED]",
        message,
    )
    message = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [REDACTED]", message)
    message = re.sub(
        r"(?i)\b(api[_-]?key|token|authorization|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        message,
    )
    message = re.sub(r"\s+", " ", message).strip()
    if len(message) > max_length:
        message = message[: max_length - 3].rstrip() + "..."
    return message


def _code_provenance() -> Dict[str, Any]:
    trainer_p = None
    calibrator_p = None
    for candidate_src in [
        Path(__file__).resolve().parent / "src",
        Path(__file__).resolve().parent.parent / "src",
        Path("/home/jovyan/work/proyectonp/repo_v8_canonical/src"),
    ]:
        t = candidate_src / "dermssl/segmentation_methods/dermlip_slic/trainer.py"
        c = candidate_src / "dermssl/segmentation_methods/dermlip_slic/calibrator.py"
        if t.is_file() and c.is_file():
            trainer_p = t
            calibrator_p = c
            break

    files = [Path(__file__).resolve()]
    if trainer_p is not None and calibrator_p is not None:
        files.extend([trainer_p, calibrator_p])
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {"git_revision": revision, "source_sha256": {p.name: _sha256(p) for p in files}}


def _atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    temporary_path = Path(path).with_name(Path(path).name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _write_training_started_receipt(
    path: Path,
    *,
    run_id: str,
    identity_fingerprint: str,
    method: str,
    stage: str,
    output_dir: Path,
    pool_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Atomically record the start boundary without changing gate or preflight inputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing training-start receipt: {path}")
    receipt = {
        "schema": "dermlip_training_started_receipt.v1",
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
        "method": method,
        "stage": stage,
        "output_dir": str(Path(output_dir).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    if pool_provenance is not None:
        receipt["pool_provenance"] = pool_provenance
    temporary_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return receipt


def _write_checkpoint_receipt(
    checkpoint_path: Path,
    *,
    run_id: str,
    identity_fingerprint: str,
    method: str,
    stage: str,
    output_dir: Path,
    epoch: int,
    pool_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    receipt_path = checkpoint_path.with_name("last_checkpoint.receipt.json")
    receipt = {
        "schema": "dermlip_checkpoint_receipt.v1",
        "sha256": _sha256(checkpoint_path),
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
        "method": method,
        "stage": stage,
        "output_dir": str(Path(output_dir).resolve()),
        "epoch": epoch,
    }
    if pool_provenance is not None:
        receipt["pool_provenance"] = pool_provenance
    temporary_path = receipt_path.with_name(f"{receipt_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(receipt_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return receipt


def _validate_resume_checkpoint_structure(
    checkpoint: Any,
    *,
    run_id: str,
    identity_fingerprint: str,
    checkpoint_epoch: int,
) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Resume checkpoint must contain a mapping")
    required = {
        "run_id", "identity_fingerprint", "epoch", "student_state_dict", "teacher_state_dict",
        "optimizer_state_dict", "rng_state", "history", "best_student_val_dice",
        "best_teacher_val_dice",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError("Resume checkpoint structure is incomplete: " + ", ".join(sorted(missing)))
    if (
        checkpoint.get("run_id") != run_id
        or checkpoint.get("identity_fingerprint") != identity_fingerprint
        or checkpoint.get("epoch") != checkpoint_epoch
    ):
        raise ValueError("Resume checkpoint run, identity, or epoch does not match its receipt")
    for key in ("student_state_dict", "teacher_state_dict", "optimizer_state_dict", "rng_state"):
        if not isinstance(checkpoint.get(key), dict):
            raise ValueError(f"Resume checkpoint field {key} must be a mapping")
    if not isinstance(checkpoint.get("history"), list):
        raise ValueError("Resume checkpoint history must be a list")
    rng_state = checkpoint["rng_state"]
    if not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(rng_state):
        raise ValueError("Resume checkpoint RNG state is incomplete")
    for key in ("best_student_val_dice", "best_teacher_val_dice"):
        if not isinstance(checkpoint.get(key), (int, float)):
            raise ValueError(f"Resume checkpoint field {key} must be numeric")
    return checkpoint


def _perceptual_hash(path: Path) -> int:
    """Return a deterministic low-frequency perceptual hash for duplicate screening."""
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(gray, type=2, norm="ortho")[:8, :8]
    values = low.flatten()[1:]
    median = np.median(values)
    bits = values > median
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def _validate_v2_shared_pool_manifest(
    manifest_path: Path,
    manifest: Dict[str, Any],
    records: List[Dict[str, Any]],
    manifest_sha256: str,
) -> Optional[Dict[str, Any]]:
    """Validate a shared v2 segmentation pool against its code pins and adjacent receipt."""
    eligibility = manifest.get("eligibility", {})
    pool_contract = manifest.get("pool_contract", {})
    if not isinstance(eligibility, dict):
        eligibility = {}
    if not isinstance(pool_contract, dict):
        pool_contract = {}

    from dermssl.segmentation_methods.common.protocol import (
        V2_SHARED_POOL_CONTRACT,
        validate_v2_shared_pool_contract,
    )

    pins = V2_SHARED_POOL_CONTRACT
    receipt_path = manifest_path.parent / "receipt.json"
    receipt = None
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            receipt = None
    receipt_indicates_v2 = (
        isinstance(receipt, dict)
        and receipt.get("schema") == "v2_shared_segmentation_pool_bridge.v1"
    )
    v2_contract_fields = {
        "source_exclusions_sha256",
        "evaluation_exclusion_receipt_sha256",
        "ph2_manifest_sha256",
        "ph2_exclusions_sha256",
        "selection_source_manifest_sha256",
        "refit_source_manifest_sha256",
        "selection_cohort_sha256",
        "refit_cohort_sha256",
    }
    v2_eligibility_fields = {
        "excludes_target_test",
        "patient_lesion_groups_preserved",
        "deduplication_preserved",
        "exclusions_sha256",
        "verification_revision",
    }
    is_v2_candidate = (
        pool_contract.get("contract_id") == pins["contract_id"]
        or pool_contract.get("stage") in {"selection", "refit"}
        or eligibility.get("source_receipt_sha256") == pins["source_receipt_sha256"]
        or eligibility.get("source_manifest_sha256") == pins["source_pool_manifest_sha256"]
        or bool(v2_contract_fields.intersection(pool_contract))
        or bool(v2_eligibility_fields.intersection(eligibility))
        or receipt_indicates_v2
    )
    if not is_v2_candidate:
        return None
    if not isinstance(manifest.get("eligibility"), dict) or not isinstance(
        manifest.get("pool_contract"), dict
    ):
        raise ValueError("V2 shared-pool manifest is missing its eligibility or contract object")

    try:
        validate_v2_shared_pool_contract(pool_contract, records=records)
    except ValueError as exc:
        raise ValueError(f"V2 shared-pool contract rejected: {exc}") from exc

    stage = pool_contract["stage"]
    expected_eligibility = {
        "verified": True,
        "excludes_target_test": True,
        "patient_lesion_groups_preserved": True,
        "deduplication_preserved": True,
        "source_manifest_sha256": pins["source_pool_manifest_sha256"],
        "source_receipt_sha256": pins["source_receipt_sha256"],
        "exclusions_sha256": pins["source_exclusions_sha256"],
    }
    if any(eligibility.get(key) != value for key, value in expected_eligibility.items()):
        raise ValueError("V2 shared-pool eligibility does not match the pinned source artifacts")
    if not re.fullmatch(r"[0-9a-f]{64}", str(eligibility.get("verification_revision", ""))):
        raise ValueError("V2 shared-pool eligibility is missing its builder revision hash")
    if (manifest.get("schema_version") != 1
            or manifest.get("partition") != "train"
            or manifest.get("count") != pins[f"{stage}_count"]):
        raise ValueError("V2 shared-pool manifest schema, partition, or count mismatch")

    for index, record in enumerate(records):
        if (not isinstance(record.get("image_id"), str) or not record["image_id"]
                or record.get("dataset") not in {"ISIC2019", "ISIC2020"}
                or not isinstance(record.get("image_path"), str) or not record["image_path"]
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("image_sha256", "")))):
            raise ValueError(f"V2 shared-pool image-only row {index} has invalid identity or provenance")

    if not receipt_path.is_file():
        raise ValueError("V2 shared-pool adjacent receipt.json is required")
    if not isinstance(receipt, dict):
        raise ValueError("V2 shared-pool adjacent receipt.json is unreadable")

    source_hashes = {
        "split_v2_receipt": pins["source_receipt_sha256"],
        "candidate_source_pool_manifest": pins["source_pool_manifest_sha256"],
        "split_v2_exclusion_crosswalk": pins["source_exclusions_sha256"],
        "selection_source_manifest": pins["selection_source_manifest_sha256"],
        "refit_source_manifest": pins["refit_source_manifest_sha256"],
        "isic2018_task1_evaluation_manifest": pins["evaluation_manifest_sha256"],
        "isic2018_task1_exclusion_receipt": pins["evaluation_exclusion_receipt_sha256"],
        "ph2_eligible_manifest": pins["ph2_manifest_sha256"],
        "ph2_curator_exclusions": pins["ph2_exclusions_sha256"],
    }
    source_artifacts = receipt.get("source_artifacts", {}) if isinstance(receipt, dict) else {}
    arms = receipt.get("arms")
    if (not isinstance(arms, list)
            or receipt.get("schema") != "v2_shared_segmentation_pool_bridge.v1"
            or receipt.get("status") != "preflight_bridge_built_execution_disabled"
            or receipt.get("same_underlying_pool_for_both_arms") is not True
            or not {"DermLIP-SLED", "SLED v8"}.issubset(set(arms))
            or any(source_artifacts.get(name, {}).get("sha256") != digest
                   for name, digest in source_hashes.items())):
        raise ValueError("V2 shared-pool receipt schema or source artifact pins mismatch")
    required_false_gates = (
        "execution_enabled", "execution_ready", "launch_ready", "training_started",
        "training_consumable", "pool_manifest_frozen",
    )
    gates = receipt.get("gates", {})
    if any(gates.get(name) is not False for name in required_false_gates):
        raise ValueError("V2 shared-pool receipt must retain all execution gates false")

    output = receipt.get("pools", {}).get(stage, {})
    expected_path = manifest_path.resolve()
    if (Path(str(output.get("path", ""))).resolve() != expected_path
            or output.get("sha256") != manifest_sha256
            or output.get("count") != pins[f"{stage}_count"]
            or output.get("cohort_sha256") != pins[f"{stage}_cohort_sha256"]
            or output.get("dataset_counts") != pins[f"{stage}_dataset_counts"]):
        raise ValueError("V2 shared-pool receipt does not bind this manifest and cohort")

    return {"pool_stage": stage, "target_size": [224, 224], "receipt_path": str(receipt_path)}


def validate_unlabeled_manifest(
    manifest_path: Path,
    labeled_records: List[Dict[str, Any]],
    labeled_base_dir: Path,
    *,
    phash_distance: int = 4,
    failure_ledger_path: Optional[Path] = None,
) -> Tuple[List[Path], str, Dict[str, Any]]:
    """Validate every unlabeled manifest row and reject identity/content overlap."""
    def fail(image_id: str, reason: str, error_type: type[Exception] = ValueError) -> None:
        if failure_ledger_path is not None:
            ledger_path = Path(failure_ledger_path)
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"image_id": image_id, "reason": reason}) + "\n")
        raise error_type(reason)

    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        fail("manifest", "unlabeled_manifest_missing", FileNotFoundError)
    raw_manifest = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    try:
        doc = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail("manifest", f"unlabeled_manifest_invalid_json:{type(exc).__name__}")
    records = doc.get("records")
    if not isinstance(records, list) or not records:
        fail("manifest", "unlabeled_manifest_records_empty_or_invalid")

    try:
        v2_pool_audit = _validate_v2_shared_pool_manifest(
            manifest_path, doc, records, manifest_sha256
        )
    except ValueError as exc:
        fail("manifest", str(exc))

    labeled_ids = {str(r.get("image_id", "")).strip() for r in labeled_records}
    group_keys = ("case_group_id", "lesion_id", "patient_id")
    labeled_groups = {
        key: {str(r[key]).strip() for r in labeled_records if r.get(key) not in (None, "")}
        for key in group_keys
    }
    labeled_hashes = {str(r.get("image_sha256", "")).lower() for r in labeled_records if r.get("image_sha256")}
    labeled_phashes = []
    for record in labeled_records:
        raw_path = record.get("image_path")
        if raw_path:
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = Path(labeled_base_dir) / image_path
            if image_path.is_file():
                labeled_phashes.append((record.get("image_id", "unknown"), _perceptual_hash(image_path)))

    seen_ids: set[str] = set()
    paths: List[Path] = []
    pool_phashes = []
    pool_hashes = set()
    is_certified = v2_pool_audit is not None or (
        doc.get("eligibility", {}).get("verified") is True
        and doc.get("pool_contract") is not None
    )
    for index, record in enumerate(records):
        image_id = str(record.get("image_id", "")).strip()
        digest = str(record.get("image_sha256", "")).lower()
        raw_path = str(record.get("image_path", "")).strip()
        if not image_id or not raw_path or len(digest) != 64:
            fail(image_id or f"row_{index}", "required_image_id_image_path_or_image_sha256_missing")
        if image_id in seen_ids:
            fail(image_id, "duplicate_image_id_in_unlabeled_manifest")
        seen_ids.add(image_id)
        path = Path(raw_path)
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        group_overlaps = [
            key
            for key in group_keys
            if record.get(key) not in (None, "") and str(record[key]).strip() in labeled_groups[key]
        ]
        if image_id in labeled_ids or group_overlaps:
            fail(image_id, "unlabeled_pool_identity_overlap_with_labeled_data")
        if digest in labeled_hashes:
            fail(image_id, "unlabeled_pool_sha256_overlap_with_labeled_data")
        if digest in pool_hashes:
            fail(image_id, "duplicate_exact_image_sha256_within_unlabeled_pool")
        pool_hashes.add(digest)
        paths.append(path)

        if not is_certified:
            if not path.is_file():
                fail(image_id, "unlabeled_image_file_missing", FileNotFoundError)
            observed_digest = _sha256(path)
            if observed_digest != digest:
                fail(image_id, "unlabeled_image_sha256_mismatch")
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                fail(image_id, f"unlabeled_image_decode:{type(exc).__name__}")
            image_phash = _perceptual_hash(path)
            pool_phashes.append((image_id, image_phash))

    if is_certified and paths:
        # Fast integrity probe on boundary and midpoint files
        for probe_idx in (0, len(paths) // 2, len(paths) - 1):
            if not paths[probe_idx].is_file():
                fail(records[probe_idx].get("image_id", f"idx_{probe_idx}"), "unlabeled_image_file_missing", FileNotFoundError)

    # Include labeled validation rows in the supplied labeled_records list; reject near duplicates too.
    for image_id, image_hash in pool_phashes:
        for labeled_id, labeled_hash in labeled_phashes:
            if (image_hash ^ labeled_hash).bit_count() <= phash_distance:
                fail(image_id, f"unlabeled_pool_perceptual_hash_overlap_with_labeled_case:{labeled_id}")
    audit = {
        "manifest_sha256": manifest_sha256,
        "records_validated": len(paths),
        "image_ids_checked": len(seen_ids),
        "exact_sha256_hashes_checked": len(pool_hashes),
        "perceptual_hash_method": "8x8_dct_median_hash" if not is_certified else "certified_pool_freeze_receipt",
        "perceptual_hash_distance_threshold": phash_distance,
        "perceptual_hash_comparisons": len(pool_phashes) * len(labeled_phashes) if not is_certified else 0,
        "labeled_cases_with_perceptual_hash": len(labeled_phashes),
        "patient_id_metadata_available_in_labeled": any(r.get("patient_id") for r in labeled_records),
        "patient_id_metadata_available_in_unlabeled": any(r.get("patient_id") for r in records),
        "patient_independence_accredited": False,
        "overlap_checks_passed": ["image_id", "case_group_id", "lesion_id", "patient_id_if_present", "sha256"] + (["perceptual_hash"] if not is_certified else ["certified_fresh_pool"]),
    }
    if v2_pool_audit is not None:
        audit.update(v2_pool_audit)
    return paths, manifest_sha256, audit

# Ensure src is on sys.path
for _candidate in [
    Path(__file__).resolve().parent / "src",
    Path(__file__).resolve().parent.parent / "src",
]:
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from dermssl.segmentation_methods.dermlip_slic.calibrator import DermLIPSuperpixelCalibrator
from dermssl.segmentation_methods.dermlip_slic.trainer import (
    TeacherStudentTrainer,
    build_segmentation_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)
PH2_HF_REPOSITORY = "ahishamm/ph2_masks"
PH2_HF_REVISION = "24c39c79b6543aa6ffed7c80617b9f335dfe77a9"
PH2_HF_PATH = "data/train-00000-of-00001-9b48dfef44b35212.parquet"
PH2_HF_SHA256 = "0c7c7a2d5cf06537f0e440f0a36ba7f3c94452c97da5f8a0c17279d2e3dd77b1"
PH2_HF_PARQUET_SIZE_BYTES = 20547347
PH2_HF_URL = (
    f"https://huggingface.co/datasets/{PH2_HF_REPOSITORY}/resolve/{PH2_HF_REVISION}/{PH2_HF_PATH}"
)
PH2_REVIEW_FOREGROUND_COVERAGE = 0.90
PH2_MAX_FOREGROUND_COVERAGE = 0.95
PH2_CURATOR_EXCLUDED_IDS = {"IMD240", "IMD411", "IMD413", "IMD417", "IMD424"}
PH2_CURATOR_EXCLUSION_REASON = (
    "Curator-designated incorrect ground-truth mask; excluded from PH2 supervision and PH2 mask-based scoring. "
    "This is an explicit curation decision, not an automated quality classification."
)
DERMLIP_MODEL_REPOSITORY = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
DERMLIP_MODEL_REVISION = "b455d4ca3b39a742ae9a8ea7d3246216c6d4bf4f"
DERMLIP_TOKENIZER_REPOSITORY = "NeuML/pubmedbert-base-embeddings"
DERMLIP_TOKENIZER_REVISION = "b79526d6ef3645e0df4530322e266f24c829f5ef"
DERMLIP_LICENSE = "CC-BY-NC-ND-4.0"
DERMLIP_MODEL_NAME = "DermLIP PanDerm-base-w-PubMed-256"
DERMLIP_CONFIG_SOURCE_SHA256 = "0d9dcaeebcd6fe219392578f3da73ab025d471d4c7289698b56788172ddc2ba1"
DERMLIP_CHECKPOINT_SOURCE_SHA256 = "c01315cc8bda77decb7bfaaafaf4668979d065d0ded33c1cb42108cd8d56fedf"
DERMLIP_CACHE_FILES = {
    "open_clip_config.json": "model",
    "open_clip_model.safetensors": "model",
}
DERMLIP_INPUT_SIZE = (224, 224)


def verify_dermlip_hf_cache(
    identity: Dict[str, Any], cache_root: Optional[Path] = None
) -> Dict[str, Any]:
    """Read-only verification of the active HF refs and files OpenCLIP will consume."""
    if cache_root is None:
        if os.environ.get("HF_HUB_CACHE"):
            cache_root = Path(os.environ["HF_HUB_CACHE"])
        elif os.environ.get("HF_HOME"):
            cache_root = Path(os.environ["HF_HOME"]) / "hub"
        else:
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_root = Path(cache_root).expanduser().resolve()
    repositories = {
        "model": (identity["model_repository"], identity["model_revision"]),
        "tokenizer": (identity["tokenizer_repository"], identity["tokenizer_revision"]),
    }
    snapshots: Dict[str, Path] = {}
    refs: Dict[str, str] = {}
    for label, (repository, approved_revision) in repositories.items():
        repo_cache = cache_root / f"models--{repository.replace('/', '--')}"
        if not repo_cache.is_dir():
            target_name = f"models--{repository.replace('/', '--')}".lower()
            for candidate in cache_root.iterdir():
                if candidate.is_dir() and candidate.name.lower() == target_name:
                    repo_cache = candidate
                    break
        ref_path = repo_cache / "refs" / "main"
        if not ref_path.is_file():
            raise FileNotFoundError(f"DermLIP {label} active HF cache ref is missing: {ref_path}")
        active_revision = ref_path.read_text(encoding="utf-8").strip().lower()
        if active_revision != approved_revision:
            raise ValueError(
                f"DermLIP {label} active HF cache ref does not match approved revision: "
                f"expected {approved_revision}, found {active_revision or '<empty>'}"
            )
        snapshot = repo_cache / "snapshots" / active_revision
        if not snapshot.is_dir():
            raise FileNotFoundError(f"DermLIP {label} HF cache snapshot is missing: {snapshot}")
        refs[label] = active_revision
        snapshots[label] = snapshot

    declared_files = identity.get("files")
    if not isinstance(declared_files, dict):
        raise ValueError("DermLIP identity is missing its locally declared cache file hashes")
    cache_hashes: Dict[str, str] = {}
    for filename, label in DERMLIP_CACHE_FILES.items():
        expected_hash = declared_files.get(filename)
        if not expected_hash:
            raise ValueError(f"DermLIP manifest does not declare the consumed cache file: {filename}")
        path = snapshots[label] / filename
        if not path.is_file():
            raise FileNotFoundError(f"DermLIP {label} HF cache file is missing: {filename}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"DermLIP cache SHA-256 mismatch for {filename}")
        cache_hashes[filename] = actual_hash

    tokenizer_files = {
        name for name in declared_files
        if Path(name).name.lower() in {
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt",
        }
    }
    if not tokenizer_files:
        raise ValueError("DermLIP manifest does not declare tokenizer cache files")
    actual_tokenizer_files = {
        path.name for path in snapshots["tokenizer"].iterdir()
        if path.is_file() and path.name.lower() in {
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt",
        }
    }
    if actual_tokenizer_files != {Path(name).name for name in tokenizer_files}:
        raise ValueError(
            "DermLIP tokenizer cache file inventory differs from the declared hashes: "
            f"unlisted={sorted(actual_tokenizer_files - {Path(name).name for name in tokenizer_files})}, "
            f"missing={sorted({Path(name).name for name in tokenizer_files} - actual_tokenizer_files)}"
        )
    for filename in sorted(tokenizer_files):
        path = snapshots["tokenizer"] / filename
        if Path(filename).parent != Path(".") or not path.is_file():
            raise FileNotFoundError(f"DermLIP tokenizer cache file is missing or not at its root: {filename}")
        actual_hash = _sha256(path)
        if actual_hash != declared_files[filename]:
            raise ValueError(f"DermLIP tokenizer cache SHA-256 mismatch for {filename}")
        cache_hashes[filename] = actual_hash

    return {
        "verified": True,
        "cache_root": str(cache_root),
        "model_ref": refs["model"],
        "tokenizer_ref": refs["tokenizer"],
        "model_config_sha256": cache_hashes["open_clip_config.json"],
        "model_weights_sha256": cache_hashes["open_clip_model.safetensors"],
        "tokenizer_file_sha256": {
            name: cache_hashes[name] for name in sorted(tokenizer_files)
        },
        "tokenizer_source_hashes_verified": False,
        "tokenizer_cache_integrity_verified": True,
        "cache_mutated": False,
    }


def load_dermlip_model_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Verify the immutable local DermLIP bundle and return its run identity."""
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"DermLIP model manifest not found: {manifest_path}")
    raw_manifest = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw_manifest)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"DermLIP model manifest is invalid JSON: {type(exc).__name__}") from exc

    if manifest.get("schema_version") != 1:
        raise ValueError("DermLIP model manifest schema_version must be 1")
    model = manifest.get("model", {})
    tokenizer = manifest.get("tokenizer", {})
    if model.get("repository") != DERMLIP_MODEL_REPOSITORY:
        raise ValueError("DermLIP model repository does not match the approved model identity")
    if tokenizer.get("repository") != DERMLIP_TOKENIZER_REPOSITORY:
        raise ValueError("DermLIP tokenizer repository does not match the approved tokenizer identity")
    pinned_revisions = {
        "model": DERMLIP_MODEL_REVISION,
        "tokenizer": DERMLIP_TOKENIZER_REVISION,
    }
    for label, item in (("model", model), ("tokenizer", tokenizer)):
        revision = str(item.get("revision", "")).lower()
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError(f"DermLIP {label} revision must be a full 40-character commit SHA")
        if revision != pinned_revisions[label]:
            raise ValueError(f"DermLIP {label} revision must match the pinned source revision")
    if manifest.get("license") != DERMLIP_LICENSE:
        raise ValueError(f"DermLIP bundle license must be recorded as {DERMLIP_LICENSE}")
    if manifest.get("usage") != {
        "commercial_use": "prohibited",
        "weight_modification": "prohibited",
    }:
        raise ValueError("DermLIP bundle must preserve noncommercial, no-weight-modification restrictions")
    raw_bundle_path = Path(str(manifest.get("bundle_path", "")))
    bundle_path = (manifest_path.parent / raw_bundle_path).resolve() if not raw_bundle_path.is_absolute() else raw_bundle_path.resolve()
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"DermLIP local bundle directory not found: {bundle_path}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("DermLIP manifest must contain a nonempty path-to-SHA-256 file map")

    verified_files: Dict[str, str] = {}
    for relative_name, expected_hash in sorted(files.items()):
        relative_path = Path(str(relative_name))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"DermLIP bundle file path must stay within the bundle: {relative_name}")
        expected_hash = str(expected_hash).lower()
        if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
            raise ValueError(f"DermLIP bundle file has an invalid SHA-256: {relative_name}")
        file_path = (bundle_path / relative_path).resolve()
        if bundle_path not in file_path.parents or not file_path.is_file():
            raise FileNotFoundError(f"DermLIP bundle file is missing or outside bundle: {relative_name}")
        if _sha256(file_path) != expected_hash:
            raise ValueError(f"DermLIP bundle file SHA-256 mismatch: {relative_name}")
        verified_files[relative_path.as_posix()] = expected_hash

    actual_inventory = {
        path.relative_to(bundle_path).as_posix()
        for path in bundle_path.rglob("*")
        if path.is_file()
    }
    declared_inventory = set(verified_files)
    if actual_inventory != declared_inventory:
        omitted = sorted(actual_inventory - declared_inventory)
        missing = sorted(declared_inventory - actual_inventory)
        raise ValueError(
            "DermLIP bundle file inventory must exactly match the SHA-256 manifest; "
            f"unlisted={omitted}, missing={missing}"
        )

    file_names = {Path(name).name.lower() for name in verified_files}
    if not file_names.intersection({"open_clip_config.json", "config.json"}):
        raise ValueError("DermLIP bundle is missing its OpenCLIP/model config")
    if not any(Path(name).suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"} for name in verified_files):
        raise ValueError("DermLIP bundle is missing its checkpoint weights")
    tokenizer_files = {
        name
        for name in verified_files
        if Path(name).name.lower() in {
            "tokenizer.json", "tokenizer_config.json", "vocab.txt", "vocab.json", "merges.txt"
        }
    }
    if not tokenizer_files:
        raise ValueError("DermLIP bundle is missing tokenizer files")
    if not any(Path(name).parent == Path(".") for name in tokenizer_files):
        raise ValueError("DermLIP tokenizer files must be at the bundle root")
    if verified_files.get("open_clip_config.json") != DERMLIP_CONFIG_SOURCE_SHA256:
        raise ValueError("DermLIP open_clip_config.json source SHA-256 mismatch")
    if verified_files.get("open_clip_model.safetensors") != DERMLIP_CHECKPOINT_SOURCE_SHA256:
        raise ValueError("DermLIP open_clip_model.safetensors source SHA-256 mismatch")

    terms_acceptance = manifest.get("terms_acceptance", {})
    if (
        not isinstance(terms_acceptance, dict)
        or terms_acceptance.get("status") != "user_attested"
        or terms_acceptance.get("verification_method") != "manual_user_attestation"
    ):
        raise ValueError("DermLIP terms acceptance requires a user_attested human_attestation")
    attestation = terms_acceptance.get("human_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("DermLIP human_attestation receipt is required")
    expected_attestation = {
        "repository": DERMLIP_MODEL_REPOSITORY,
        "model": DERMLIP_MODEL_NAME,
        "model_revision": DERMLIP_MODEL_REVISION,
        "license": DERMLIP_LICENSE,
    }
    for key, expected in expected_attestation.items():
        if attestation.get(key) != expected:
            raise ValueError(f"DermLIP human_attestation {key} does not match the approved source")
    for key in ("terms_reference", "terms_version", "accepted_by"):
        if not isinstance(attestation.get(key), str) or not attestation[key].strip():
            raise ValueError(f"DermLIP human_attestation requires nonempty {key}")
    if attestation["terms_version"] != DERMLIP_LICENSE:
        raise ValueError("DermLIP human_attestation terms_version must match the approved license")
    accepted_on = attestation.get("accepted_on")
    try:
        parsed_acceptance_date = date.fromisoformat(str(accepted_on))
    except ValueError as exc:
        raise ValueError("DermLIP human_attestation accepted_on must be an ISO date") from exc
    if parsed_acceptance_date.isoformat() != accepted_on:
        raise ValueError("DermLIP human_attestation accepted_on must use YYYY-MM-DD")
    attested_at = attestation.get("attested_at")
    try:
        parsed_timestamp = datetime.fromisoformat(str(attested_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("DermLIP human_attestation attested_at must be an ISO-8601 timestamp") from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise ValueError("DermLIP human_attestation attested_at must include a timezone")
    evidence_path_value = str(attestation.get("receipt_path", ""))
    evidence_hash = str(attestation.get("receipt_sha256", "")).lower()
    if (
        not evidence_path_value
        or len(evidence_hash) != 64
        or any(char not in "0123456789abcdef" for char in evidence_hash)
    ):
        raise ValueError("DermLIP human_attestation receipt requires a path and SHA-256")
    evidence_path = (manifest_path.parent / evidence_path_value).resolve()
    if (
        manifest_path.parent not in evidence_path.parents
        or not evidence_path.is_file()
        or evidence_path.stat().st_size == 0
    ):
        raise FileNotFoundError("DermLIP human_attestation receipt is missing or outside the manifest directory")
    if _sha256(evidence_path) != evidence_hash:
        raise ValueError("DermLIP human_attestation receipt SHA-256 mismatch")

    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "model_repository": model["repository"],
        "model_revision": str(model["revision"]).lower(),
        "tokenizer_repository": tokenizer["repository"],
        "tokenizer_revision": str(tokenizer["revision"]).lower(),
        "bundle_path": str(bundle_path),
        "model_name": f"hf-hub:{model['repository']}",
        "license": DERMLIP_LICENSE,
        "terms_acceptance_status": "user_attested",
        "terms_acceptance_verification_method": "manual_user_attestation",
        "terms_human_attestation": {
            key: value for key, value in attestation.items() if key != "receipt_path"
        },
        "terms_acceptance_evidence_path": str(evidence_path),
        "terms_acceptance_evidence_sha256": evidence_hash,
        "usage": manifest["usage"],
        "files": verified_files,
        "model_source_hashes_verified": True,
        "tokenizer_source_hashes_verified": False,
        "tokenizer_bundle_integrity_verified": True,
        "source_provenance_status": "model_files_verified_tokenizer_hashes_unverified",
    }


class PH2Dataset(Dataset):
    """PyTorch Dataset for certified PH2 labeled dermoscopic images and expert masks."""

    def __init__(
        self,
        manifest_path: Path,
        split: str = "train",
        target_size: Tuple[int, int] = DERMLIP_INPUT_SIZE,
        max_samples: Optional[int] = None,
        hair_removal: bool = False,
        augment: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.base_dir = self.manifest_path.parent
        self.split = split
        self.target_size = target_size
        self.hair_removal = hair_removal
        self.augment = bool(augment and split == "train")

        doc = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        source_kind = doc.get("source_kind")
        if source_kind not in {
            "official_local_extracted_tree",
            "huggingface_pinned_parquet_user_authorized",
        }:
            raise ValueError("PH2 manifest source is not an accepted pinned/local input")
        terms = doc.get("usage_terms", {})
        if source_kind == "official_local_extracted_tree":
            if terms.get("commercial_use") != "prohibited" or terms.get("redistribution") != "prohibited":
                raise ValueError("PH2 manifest is missing its restricted noncommercial usage terms")
        else:
            authorization = doc.get("user_authorization", {})
            license_metadata = doc.get("license_metadata", {})
            if (
                doc.get("source_repository") != PH2_HF_REPOSITORY
                or doc.get("source_revision") != PH2_HF_REVISION
                or doc.get("source_path") != PH2_HF_PATH
                or doc.get("source_url") != PH2_HF_URL
                or doc.get("source_parquet_sha256") != PH2_HF_SHA256
                or doc.get("source_artifact_sha256") != PH2_HF_SHA256
                or license_metadata.get("status") != "unverified"
                or authorization.get("status") != "user_confirmed"
                or authorization.get("confirmed_date") != "2026-10-21"
                or terms.get("commercial_use") != "prohibited_pending_upstream_terms_verification"
                or terms.get("redistribution") != "prohibited_pending_upstream_terms_verification"
            ):
                raise ValueError("PH2 Hugging Face manifest is not the authorized pinned snapshot with conservative terms")
            if doc.get("source_parquet_size_bytes") != PH2_HF_PARQUET_SIZE_BYTES:
                raise ValueError("PH2 Hugging Face source parquet size does not match pinned snapshot metadata")
        if doc.get("patient_independence_accredited") is not False:
            raise ValueError("PH2 patient independence must remain unaccredited without verified patient IDs")
        all_records = doc.get("records", [])
        source_inventory = doc.get("source_case_inventory", [])
        if (
            doc.get("schema_version") != 3
            or len(all_records) != 195
            or doc.get("total_count") != 195
            or doc.get("source_count") != 200
            or doc.get("expected_count") != 200
            or len(source_inventory) != 200
        ):
            raise ValueError("Training requires a complete 200-case PH2 source snapshot and 195 eligible cases")
        if (
            doc.get("split_unit") != "PH2_case_and_lesion_id"
            or doc.get("train_ratio") != 0.8
        ):
            raise ValueError("PH2 manifest must use the frozen 80/20 case-disjoint split")
        if (
            doc.get("source_train_count") != 160
            or doc.get("source_val_count") != 40
            or sum(record.get("split") == "train" for record in source_inventory) != 160
            or sum(record.get("split") == "val" for record in source_inventory) != 40
            or doc.get("train_count") + doc.get("val_count") != 195
            or doc.get("train_count") != sum(record.get("split") == "train" for record in all_records)
            or doc.get("val_count") != sum(record.get("split") == "val" for record in all_records)
        ):
            raise ValueError("PH2 manifest must preserve the original 160/40 split roles before exclusions")
        source_digest = str(doc.get("source_artifact_sha256", "")).lower()
        if len(source_digest) != 64 or any(char not in "0123456789abcdef" for char in source_digest):
            raise ValueError("PH2 manifest must bind a valid source_artifact_sha256")
        all_ids = [r.get("image_id") for r in all_records]
        source_ids = [r.get("image_id") for r in source_inventory]
        if (
            any(not value for value in all_ids + source_ids)
            or len(set(all_ids)) != 195
            or len(set(source_ids)) != 200
        ):
            raise ValueError("PH2 manifest must bind unique eligible IDs and a complete 200-case source inventory")
        receipt_reference = doc.get("exclusion_receipt", {})
        receipt_path = self.base_dir / str(receipt_reference.get("path", ""))
        if not receipt_path.is_file() or not receipt_reference.get("sha256"):
            raise ValueError("PH2 manifest requires a hash-bound curator exclusion receipt")
        if _sha256(receipt_path) != receipt_reference["sha256"]:
            raise ValueError("PH2 curator exclusion receipt SHA-256 mismatch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        excluded_cases = receipt.get("excluded_cases", [])
        excluded_ids = {case.get("image_id") for case in excluded_cases}
        if (
            receipt.get("schema_version") != 1
            or receipt.get("dataset") != "PH2"
            or receipt.get("decision_type") != "curator_directed_exclusion"
            or receipt.get("decision_source") != "project_curator_instruction"
            or receipt.get("reason") != PH2_CURATOR_EXCLUSION_REASON
            or receipt.get("source_artifact_sha256") != source_digest
            or receipt.get("excluded_count") != 5
            or len(excluded_cases) != 5
            or excluded_ids != PH2_CURATOR_EXCLUDED_IDS
            or doc.get("excluded_case_ids") != sorted(PH2_CURATOR_EXCLUDED_IDS)
        ):
            raise ValueError("PH2 curator exclusion receipt does not match the approved five-case policy")
        if any(record.get("split") not in {"train", "val"} for record in source_inventory) or any(
            case.get("split") not in {"train", "val"} for case in excluded_cases
        ):
            raise ValueError("PH2 source inventory and exclusion receipt must preserve train/validation roles")
        expected_eligible_train = sum(record["split"] == "train" for record in source_inventory) - sum(
            case["split"] == "train" for case in excluded_cases
        )
        expected_eligible_val = sum(record["split"] == "val" for record in source_inventory) - sum(
            case["split"] == "val" for case in excluded_cases
        )
        observed_eligible_train = sum(record.get("split") == "train" for record in all_records)
        observed_eligible_val = sum(record.get("split") == "val" for record in all_records)
        if (
            doc.get("train_count") != expected_eligible_train
            or doc.get("val_count") != expected_eligible_val
            or observed_eligible_train != expected_eligible_train
            or observed_eligible_val != expected_eligible_val
        ):
            raise ValueError(
                "PH2 expected eligible PH2 role counts: "
                f"{expected_eligible_train} train and {expected_eligible_val} val"
            )
        inventory_by_id = {record["image_id"]: record for record in source_inventory}
        if set(all_ids) | excluded_ids != set(source_ids) or set(all_ids) & excluded_ids:
            raise ValueError("PH2 eligible and explicitly excluded records do not cover the original source snapshot")
        if any(
            record.get("image_id") not in inventory_by_id
            or any(
                record.get(field) != inventory_by_id[record["image_id"]].get(field)
                for field in ("source_row_index", "source_row_id", "split", "source_image_sha256", "source_mask_sha256")
            )
            for record in all_records
        ):
            raise ValueError("PH2 eligible manifest records disagree with the original source snapshot")
        if any(
            case.get("image_id") not in inventory_by_id
            or case.get("exclusion_reason") != PH2_CURATOR_EXCLUSION_REASON
            or any(
                case.get(field) != inventory_by_id[case["image_id"]].get(field)
                for field in ("source_row_index", "source_row_id", "split", "source_image_sha256", "source_mask_sha256")
            )
            for case in excluded_cases
        ):
            raise ValueError("PH2 exclusion receipt does not match original source image/mask hashes and roles")
        for record in source_inventory:
            if (
                record.get("split") not in {"train", "val"}
                or any(
                    len(str(record.get(field, ""))) != 64
                    or any(char not in "0123456789abcdef" for char in str(record[field]).lower())
                    for field in ("source_image_sha256", "source_mask_sha256")
                )
                or record.get("source_row_index") is None
                or not record.get("source_row_id")
            ):
                raise ValueError("PH2 source snapshot inventory is missing row identity, split, or image/mask hashes")
        if any(
            not r.get("source_row_id")
            or r.get("source_row_index") is None
            or len(str(r.get("source_image_sha256", ""))) != 64
            or len(str(r.get("source_mask_sha256", ""))) != 64
            for r in all_records
        ):
            raise ValueError("PH2 manifest records must bind source row IDs and source image/mask hashes")
        train_groups = {r.get("case_group_id") for r in all_records if r.get("split") == "train"}
        val_groups = {r.get("case_group_id") for r in all_records if r.get("split") == "val"}
        if not train_groups or not val_groups or None in train_groups or None in val_groups or train_groups & val_groups:
            raise ValueError("PH2 train/validation case groups must be present, nonempty, and disjoint")
        for record in all_records:
            for field in ("image_path", "mask_path"):
                path = self.base_dir / record[field]
                if not path.is_file():
                    raise FileNotFoundError(f"PH2 {field} missing for {record['image_id']}: {path}")
                expected_hash = record.get(field.replace("_path", "_sha256"))
                if not expected_hash or _sha256(path) != expected_hash:
                    raise ValueError(f"PH2 {field} SHA-256 mismatch for {record['image_id']}")
            with Image.open(self.base_dir / record["image_path"]) as image:
                image_size = image.size
                image.verify()
            with Image.open(self.base_dir / record["mask_path"]) as mask_image:
                mask = np.asarray(mask_image.convert("L"))
            if image_size != (mask.shape[1], mask.shape[0]):
                raise ValueError(f"PH2 image/mask geometry mismatch for {record['image_id']}")
            coverage = float(np.mean(mask > 0))
            if coverage < 0.01:
                raise ValueError(f"PH2 mask for {record['image_id']} is empty or below minimum coverage")
            if coverage > PH2_REVIEW_FOREGROUND_COVERAGE:
                adjudicated_ids = set(doc.get("coverage_adjudication", {}).get("adjudicated_case_ids", []))
                if record["image_id"] in adjudicated_ids or record.get("adjudicated") is True:
                    LOGGER.info(
                        "Case %s has high coverage (%.4f) but is recognized as manually adjudicated",
                        record["image_id"],
                        coverage,
                    )
                else:
                    reasons = ["near_full_mask_requires_manual_lesion_vs_fov_review"]
                    if coverage > PH2_MAX_FOREGROUND_COVERAGE:
                        reasons.append("foreground_coverage_hard_gate_exceeded")
                    raise ValueError(
                        f"PH2 mask for {record['image_id']} requires QC review: "
                        f"foreground_fraction={coverage:.6f}, reason_codes={reasons}"
                    )
        records = [r for r in doc["records"] if r["split"] == split]
        if split not in {"train", "val"} or not records:
            raise ValueError(f"PH2 split {split!r} is missing or empty")
        if max_samples is not None:
            records = records[:max_samples]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img_path = self.base_dir / rec["image_path"]
        mask_path = self.base_dir / rec["mask_path"]

        img = Image.open(img_path).convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)
        mask = Image.open(mask_path).convert("L").resize(self.target_size, Image.Resampling.NEAREST)

        if self.hair_removal:
            from dermssl.segmentation_methods.common.hair import remove_hair
            arr_u8 = np.asarray(img, dtype=np.uint8)
            clean_u8, _ = remove_hair(arr_u8, shaped=False)
            img = Image.fromarray(clean_u8)

        if self.augment:
            if np.random.rand() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rot_k = int(np.random.randint(0, 4))
            if rot_k == 1:
                img = img.transpose(Image.Transpose.ROTATE_90)
                mask = mask.transpose(Image.Transpose.ROTATE_90)
            elif rot_k == 2:
                img = img.transpose(Image.Transpose.ROTATE_180)
                mask = mask.transpose(Image.Transpose.ROTATE_180)
            elif rot_k == 3:
                img = img.transpose(Image.Transpose.ROTATE_270)
                mask = mask.transpose(Image.Transpose.ROTATE_270)
            factor_b = float(np.random.uniform(0.88, 1.12))
            factor_c = float(np.random.uniform(0.88, 1.12))
            img = ImageEnhance.Brightness(img).enhance(factor_b)
            img = ImageEnhance.Contrast(img).enhance(factor_c)

        # To normalized tensor [0, 1]
        img_t = pil_to_tensor(img)
        # Binary mask tensor [0, 1] with shape [1, H, W]
        mask_arr = (np.array(mask) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_arr).unsqueeze(0)

        return img_t, mask_t


class UnlabeledDermDataset(Dataset):
    """Dataset producing paired weak and strong augmentations for consistency training."""

    def __init__(
        self,
        image_paths: List[Path],
        target_size: Tuple[int, int] = (224, 224),
        calibrator: Optional[DermLIPSuperpixelCalibrator] = None,
        max_samples: Optional[int] = None,
        failure_ledger_path: Optional[Path] = None,
        hair_removal: bool = False,
        return_slic: bool = False,
        n_segments: int = 150,
        compactness: float = 10.0,
    ) -> None:
        self.image_paths = image_paths[:max_samples] if max_samples is not None else image_paths
        self.target_size = target_size
        self.calibrator = calibrator
        self.failure_ledger_path = Path(failure_ledger_path) if failure_ledger_path else None
        self.hair_removal = hair_removal
        self.return_slic = return_slic
        self.n_segments = n_segments
        self.compactness = compactness

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_strong_aug(self, img: Image.Image) -> Image.Image:
        """Apply photometric perturbations for student branch."""
        # Random brightness and contrast jitter
        factor_b = float(np.random.uniform(0.8, 1.2))
        factor_c = float(np.random.uniform(0.8, 1.2))
        img = ImageEnhance.Brightness(img).enhance(factor_b)
        img = ImageEnhance.Contrast(img).enhance(factor_c)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.image_paths[idx]
        try:
            with Image.open(path) as source_image:
                source_image.draft("RGB", (self.target_size[0] * 2, self.target_size[1] * 2))
                pil_img = source_image.convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)
        except Exception as exc:
            self._record_failure(path, f"image_decode:{type(exc).__name__}")
            raise RuntimeError(f"Unlabeled image decode failed for {path.name}: {type(exc).__name__}") from exc

        if self.hair_removal:
            from dermssl.segmentation_methods.common.hair import remove_hair
            arr_u8 = np.asarray(pil_img, dtype=np.uint8)
            clean_u8, _ = remove_hair(arr_u8, shaped=False)
            pil_img = Image.fromarray(clean_u8)

        # Optional DermLIP semantic guidance extraction
        if self.calibrator is not None:
            try:
                res = self.calibrator.calibrate_image(pil_img, target_size=self.target_size)
                if "probability_map" not in res:
                    raise ValueError("calibrator output has no probability_map")
                sem_prob = torch.from_numpy(res["probability_map"]).unsqueeze(0).float()
                if sem_prob.shape != (1, self.target_size[0], self.target_size[1]) or not torch.isfinite(sem_prob).all():
                    raise ValueError("calibrator probability_map has invalid shape or non-finite values")
            except Exception as exc:
                self._record_failure(path, f"dermlip_guidance:{type(exc).__name__}")
                raise RuntimeError(f"DermLIP guidance failed for {path.name}: {type(exc).__name__}") from exc
        else:
            sem_prob = torch.full((1, self.target_size[0], self.target_size[1]), 0.5, dtype=torch.float32)

        # Synchronized spatial transformation shared across teacher and student views
        flip_h = np.random.rand() > 0.5
        if flip_h:
            pil_img = pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            sem_prob = torch.flip(sem_prob, dims=[-1])

        # 1. Weak view: spatially transformed base image
        weak_img = pil_img.copy()

        # 2. Strong view: photometric perturbations on identical spatial geometry
        strong_img = self._apply_strong_aug(pil_img)

        weak_t = pil_to_tensor(weak_img)
        strong_t = pil_to_tensor(strong_img)

        if self.return_slic:
            from skimage.segmentation import slic
            arr_rgb = np.asarray(weak_img, dtype=np.uint8)
            slic_map = slic(
                arr_rgb,
                n_segments=self.n_segments,
                compactness=self.compactness,
                start_label=0,
                max_num_iter=5,
            )
            slic_t = torch.from_numpy(slic_map).long()
            return weak_t, strong_t, sem_prob, slic_t

        return weak_t, strong_t, sem_prob

    def _record_failure(self, path: Path, reason: str) -> None:
        if self.failure_ledger_path is None:
            return
        self.failure_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.failure_ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"image_id": Path(path).stem, "reason": reason}) + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=None,
        help="Hash-bound DermLIP selection runtime config; applies the pinned recipe and input paths.",
    )
    parser.add_argument(
        "--ph2-manifest",
        type=Path,
        default=Path("data/ph2/manifest.json"),
        help="Path to PH2 dataset manifest.json.",
    )
    parser.add_argument(
        "--unlabeled-manifest",
        type=Path,
        default=None,
        help="Required independent unlabeled JSON manifest (except explicit smoke mode).",
    )
    parser.add_argument(
        "--pool-use-attestation",
        type=Path,
        default=None,
        help="Versioned method-scoped v2 pool-consumption attestation; required for v2 manifests.",
    )
    parser.add_argument(
        "--execution-gate",
        type=Path,
        default=None,
        help="Separate stage-specific execution gate; required with --execute for v2 manifests.",
    )
    parser.add_argument(
        "--execution-claim-path",
        type=Path,
        default=None,
        help="Unique claim path bound by the v2 stage gate; consumed once for an authorized new run.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute a v2 training stage only when its independently enabled gate matches all pins.",
    )
    parser.add_argument(
        "--allow-legacy-pool",
        action="store_true",
        help="Compatibility flag; legacy manifests remain rejected for fresh training.",
    )
    parser.add_argument(
        "--dermlip-model-manifest",
        type=Path,
        default=None,
        help="Required verified local DermLIP model/tokenizer bundle identity manifest when guidance is enabled.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/dermlip_semi_pilot"),
        help="Output directory for checkpoints, metrics, and logs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Total training epochs (default: 2 for pilot validation).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per step (default: 4).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
        help="Exponential moving average decay for teacher weights (default: 0.99).",
    )
    parser.add_argument(
        "--gamma-semantic",
        type=float,
        default=0.5,
        help="Weight of DermLIP semantic logits in teacher fusion (default: 0.5).",
    )
    parser.add_argument(
        "--superpixel-affinity-weight",
        type=float,
        default=0.0,
        help="Weight for differentiable superpixel affinity loss (L_sp) during student training (default: 0.0).",
    )
    parser.add_argument(
        "--pseudo-label-threshold-high",
        type=float,
        default=0.65,
        help="Upper probability threshold for confident foreground pseudo-labels (default: 0.65).",
    )
    parser.add_argument(
        "--pseudo-label-threshold-low",
        type=float,
        default=0.35,
        help="Lower probability threshold for confident background pseudo-labels (default: 0.35).",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="scratch",
        choices=["scratch", "resnet34", "resnet50"],
        help="Student/Teacher segmentation encoder-decoder backbone (default: 'scratch').",
    )
    parser.add_argument(
        "--soft-kl-weight",
        type=float,
        default=0.0,
        help="Weight for continuous entropy-weighted Soft KL / Bernoulli distillation (default: 0.0).",
    )
    parser.add_argument(
        "--log-linear-fusion",
        action="store_true",
        help="Use log-linear Product-of-Experts fusion between teacher and semantic prior.",
    )
    parser.add_argument(
        "--copy-paste-prob",
        type=float,
        default=0.0,
        help="Probability of Scale-Jittered Copy-Paste / BCP augmentation per labeled image (default: 0.0).",
    )
    parser.add_argument(
        "--fp-dropout",
        action="store_true",
        help="Enable UniMatch feature-level perturbation dropout consistency stream.",
    )
    parser.add_argument(
        "--calibrate-ph2-prototypes",
        action="store_true",
        help="Calibrate DermLIP foreground/background prototypes using PH2 labeled visual tokens.",
    )
    parser.add_argument(
        "--ph2-geom-aug",
        action="store_true",
        help="Apply synchronized random flips, 90-degree rotations, and photometric jitter on PH2 train split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cpu', 'cuda:0', 'cuda:1').",
    )
    parser.add_argument(
        "--pilot-samples",
        type=int,
        default=None,
        help="Limit number of samples for rapid validation (e.g. 10).",
    )
    parser.add_argument(
        "--enable-dermlip",
        action="store_true",
        default=True,
        help="Enable online DermLIP semantic guidance extraction (the default).",
    )
    parser.add_argument(
        "--disable-dermlip",
        action="store_false",
        dest="enable_dermlip",
        help="Run explicit no-DermLIP ablation; this is not the proposed guided method.",
    )
    parser.add_argument(
        "--allow-self-consistency-pilot",
        action="store_true",
        help="Allow running a self-consistency smoke test using labeled images without masks when unlabeled manifest is absent.",
    )
    parser.add_argument(
        "--kappa-uncertainty",
        type=float,
        default=0.693147,
        help="Uncertainty scaling factor kappa for entropy weighting: alpha = exp(-H / kappa). Default: ln(2) = 0.693147.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    parser.add_argument(
        "--hair-removal",
        action="store_true",
        help="Apply black-hat morphological detection + Telea inpainting before model inference and training.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume only from a checkpoint whose stored run identity exactly matches current inputs and settings.",
    )
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(actual_argv)
    args._provided_cli_options = {
        item.split("=", 1)[0] for item in actual_argv if item.startswith("--")
    }
    return args


_RUNTIME_PROFILE_FIELDS = {
    "epochs": ("epochs", "--epochs"),
    "batch_size": ("batch_size", "--batch-size"),
    "learning_rate": ("lr", "--lr"),
    "ema_decay": ("ema_decay", "--ema-decay"),
    "gamma_semantic": ("gamma_semantic", "--gamma-semantic"),
    "kappa_uncertainty": ("kappa_uncertainty", "--kappa-uncertainty"),
    "seed": ("seed", "--seed"),
    "device": ("device", "--device"),
    "guidance_enabled": ("enable_dermlip", "--enable-dermlip"),
    "pilot_samples": ("pilot_samples", "--pilot-samples"),
    "allow_self_consistency_pilot": (
        "allow_self_consistency_pilot", "--allow-self-consistency-pilot"
    ),
}


def _read_runtime_json(path: Path, description: str) -> Dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is missing or unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value


def _apply_runtime_config_profile(
    args: argparse.Namespace,
    config: Path | Dict[str, Any],
    *,
    provided_options: set[str],
) -> Dict[str, Any]:
    """Apply the exact recipe, rejecting any explicitly conflicting CLI value."""
    document = _read_runtime_json(config, "DermLIP runtime config") if isinstance(config, Path) else config
    profile = document.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("DermLIP runtime config profile must be a JSON object")
    expected_fields = set(_RUNTIME_PROFILE_FIELDS) | {
        "execution_claim_path", "stage"
    }
    allowed_fields = expected_fields | {"hair_removal"}
    if not (expected_fields <= set(profile) <= allowed_fields):
        raise ValueError("DermLIP runtime config profile fields do not match the pinned schema")
    if profile.get("stage") != "selection":
        raise ValueError("DermLIP runtime config must bind the selection stage")
    for field, (attribute, cli_flag) in _RUNTIME_PROFILE_FIELDS.items():
        configured = profile[field]
        if cli_flag == "--enable-dermlip":
            cli_supplied = "--disable-dermlip" in provided_options
            cli_value = False if cli_supplied else getattr(args, attribute)
        else:
            cli_supplied = cli_flag in provided_options
            cli_value = getattr(args, attribute)
        if cli_supplied and cli_value != configured:
            raise ValueError(f"CLI {field} conflicts with DermLIP runtime config")
        setattr(args, attribute, configured)
    if "hair_removal" in profile:
        configured_hair = profile["hair_removal"]
        if "--hair-removal" in provided_options and getattr(args, "hair_removal") != configured_hair:
            raise ValueError("CLI hair_removal conflicts with DermLIP runtime config")
        setattr(args, "hair_removal", configured_hair)
    return document


def _load_selection_runtime_config(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """Bind the disabled selection config, gate, CLI inputs, and on-disk SHA pins."""
    runtime_config_path = getattr(args, "runtime_config", None)
    if runtime_config_path is None:
        return None
    runtime_config_path = Path(runtime_config_path).resolve()
    provided_options = getattr(args, "_provided_cli_options", set())
    config = _apply_runtime_config_profile(
        args, runtime_config_path, provided_options=provided_options
    )
    if (
        config.get("schema") != "dermlip_selection_runtime_config.v1"
        or config.get("method") != "dermlip_semi_supervised"
        or config.get("stage") != "selection"
    ):
        raise ValueError("DermLIP runtime config schema, stage, or disabled flags mismatch")
    execute = bool(getattr(args, "execute", False))
    if config.get("status") == "execution_authorized":
        if config.get("verified_for_use") is not False:
            raise ValueError("DermLIP execution config requires verified_for_use=false")
        if config.get("patient_independence_accredited") is not False:
            raise ValueError(
                "DermLIP execution config requires patient_independence_accredited=false"
            )
    disabled_config = (
        config.get("status") == "preparation_only_execution_disabled"
        and config.get("execution_enabled") is False
        and config.get("launch_ready") is False
        and config.get("training_started") is False
        and config.get("verified_for_use") is False
    )
    enabled_config = (
        config.get("status") == "execution_authorized"
        and config.get("execution_ready") is True
        and config.get("execution_enabled") is True
        and config.get("launch_ready") is True
        and config.get("training_started") is False
        and config.get("verified_for_use") is False
        and config.get("patient_independence_accredited") is False
    )
    if disabled_config and execute:
        raise ValueError("DermLIP runtime config is execution-disabled and cannot be launched")
    if enabled_config and not execute:
        raise ValueError("Enabled DermLIP runtime config requires explicit --execute")
    if not disabled_config and not enabled_config:
        raise ValueError("DermLIP runtime config schema, stage, or execution flags mismatch")
    bindings = config.get("bindings")
    paths = config.get("paths")
    if not isinstance(bindings, dict) or not isinstance(paths, dict):
        raise ValueError("DermLIP runtime config is missing bindings or paths")
    from dermssl.segmentation_methods.common.pool_use import (
        V2_POOL_USE_ATTESTATION_SHA256,
        V2_POOL_USE_BINDINGS,
    )

    expected_eval = V2_POOL_USE_BINDINGS["source_artifacts"]["evaluation_manifest_sha256"]
    expected_eval_receipt = V2_POOL_USE_BINDINGS["source_artifacts"][
        "evaluation_manifest_receipt_sha256"
    ]
    expected_selection = V2_POOL_USE_BINDINGS["stages"]["selection"]
    expected_source_artifacts = V2_POOL_USE_BINDINGS["source_artifacts"]
    if (
        bindings.get("pool_use_attestation", {}).get("sha256")
        != V2_POOL_USE_ATTESTATION_SHA256
        or bindings.get("pool_manifest", {}).get("sha256")
        != expected_selection["sha256"]
        or bindings.get("pool_manifest", {}).get("path") != expected_selection["path"]
        or bindings.get("pool_manifest", {}).get("record_count")
        != expected_selection["record_count"]
        or bindings.get("pool_manifest", {}).get("stage") != "selection"
        or bindings.get("evaluation_manifest", {}).get("sha256") != expected_eval
        or bindings.get("evaluation_manifest", {}).get("receipt_sha256") != expected_eval_receipt
        or bindings.get("dermlip_model_manifest", {}).get("sha256")
        != expected_source_artifacts["dermlip_model_manifest_sha256"]
        or bindings.get("terms_acceptance", {}).get("sha256")
        != expected_source_artifacts["terms_acceptance_sha256"]
        or bindings.get("ph2_manifest", {}).get("sha256")
        != expected_source_artifacts["ph2_manifest_sha256"]
    ):
        raise ValueError("DermLIP runtime config pool or evaluation manifest pin mismatch")

    gate_path = Path(paths.get("selection_gate", "")).resolve()
    configured_paths = {
        "ph2_manifest": "ph2_manifest",
        "pool_manifest": "unlabeled_manifest",
        "pool_use_attestation": "pool_use_attestation",
        "dermlip_model_manifest": "dermlip_model_manifest",
        "output_root": "out_dir",
    }
    if disabled_config:
        configured_paths["selection_gate"] = "execution_gate"
    else:
        configured_paths["execution_gate"] = "execution_gate"
        configured_paths["execution_claim_path"] = "execution_claim_path"
    for config_key, argument in configured_paths.items():
        value = paths.get(config_key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"DermLIP runtime config path {config_key!r} is missing")
        cli_flag = {
            "ph2_manifest": "--ph2-manifest",
            "pool_manifest": "--unlabeled-manifest",
            "pool_use_attestation": "--pool-use-attestation",
            "dermlip_model_manifest": "--dermlip-model-manifest",
            "output_root": "--out-dir",
            "selection_gate": "--execution-gate",
            "execution_gate": "--execution-gate",
            "execution_claim_path": "--execution-claim-path",
        }[config_key]
        supplied = cli_flag in provided_options
        observed_arg = getattr(args, argument, None)
        if supplied and observed_arg is not None and Path(observed_arg).resolve() != Path(value).resolve():
            raise ValueError(f"CLI {argument} path conflicts with DermLIP runtime config")
        setattr(args, argument, Path(value))

    if enabled_config:
        claim_text = config["profile"].get("execution_claim_path")
        configured_claim_path = paths.get("execution_claim_path")
        if (
            not isinstance(claim_text, str)
            or not Path(claim_text).is_absolute()
            or not isinstance(configured_claim_path, str)
            or Path(claim_text).resolve() != Path(configured_claim_path).resolve()
        ):
            raise ValueError("DermLIP runtime config execution claim path binding mismatch")
        claim_path = Path(claim_text).resolve()
        if "--execution-claim-path" in provided_options and Path(
            getattr(args, "execution_claim_path")
        ).resolve() != claim_path:
            raise ValueError("CLI execution_claim_path path conflicts with DermLIP runtime config")
        args.execution_claim_path = claim_path
        if claim_path.exists():
            raise FileExistsError(f"Execution claim path is already consumed: {claim_path}")

    binding_paths = {
        "ph2_manifest": "ph2_manifest",
        "pool_manifest": "pool_manifest",
        "pool_use_attestation": "pool_use_attestation",
        "dermlip_model_manifest": "dermlip_model_manifest",
        "terms_acceptance": "terms_acceptance",
    }
    for binding_name, path_name in binding_paths.items():
        if Path(bindings[binding_name].get("path", "")).resolve() != Path(
            paths[path_name]
        ).resolve():
            raise ValueError(f"DermLIP runtime config {binding_name} path binding mismatch")

    gate = _read_runtime_json(gate_path, "DermLIP selection gate")
    config_sha256 = _sha256(runtime_config_path)
    expected_gate_flags = (
        {
            "execution_ready": True,
            "execution_enabled": True,
            "launch_ready": True,
            "training_started": False,
            "verified_for_use": False,
            "patient_independence_accredited": False,
        }
        if enabled_config
        else {
            "execution_enabled": False,
            "launch_ready": False,
            "training_started": False,
            "verified_for_use": False,
        }
    )
    if (
        gate.get("schema") != "dermlip_selection_runtime_gate.v1"
        or gate.get("method") != "dermlip_semi_supervised"
        or gate.get("stage") != "selection"
        or gate.get("runtime_config_relative_path") != config.get("artifact_path")
        or gate.get("runtime_config_sha256") != config_sha256
        or gate.get("bindings") != bindings
        or gate.get("profile") != config.get("profile")
        or any(gate.get(flag) is not value for flag, value in expected_gate_flags.items())
    ):
        mode = "enabled" if enabled_config else "disabled"
        raise ValueError(f"DermLIP selection gate does not match the {mode} runtime config")
    if enabled_config and (
        gate.get("execution_claim_path") != str(Path(config["profile"]["execution_claim_path"]).resolve())
    ):
        raise ValueError("DermLIP runtime gate execution claim path does not match runtime config")
    if config.get("evaluation_candidate") != bindings.get("evaluation_manifest"):
        raise ValueError("DermLIP runtime config evaluation candidate binding mismatch")

    for binding_name, binding in bindings.items():
        if binding_name == "evaluation_manifest":
            manifest = binding
            files = [
                (manifest.get("path"), manifest.get("sha256"), "evaluation manifest"),
                (manifest.get("receipt_path"), manifest.get("receipt_sha256"), "evaluation receipt"),
            ]
        else:
            files = [(binding.get("path"), binding.get("sha256"), binding_name)]
        for path_text, expected_sha256, description in files:
            path = Path(path_text).resolve()
            if not path.is_file() or _sha256(path) != expected_sha256:
                raise ValueError(f"DermLIP runtime config {description} SHA-256 pin mismatch")
    if enabled_config:
        stage_gate_path = Path(paths["execution_gate"]).resolve()
        if _sha256(stage_gate_path) != paths.get("execution_gate_sha256"):
            raise ValueError("DermLIP stage execution gate SHA-256 pin mismatch")
        from dermssl.segmentation_methods.common.pool_use import validate_execution_gate

        stage_gate = validate_execution_gate(
            stage_gate_path,
            method="dermlip_semi_supervised",
            stage="selection",
            profile_sha256=_json_sha256(config["profile"]),
            profile=config["profile"],
            manifest_path=Path(bindings["pool_manifest"]["path"]),
            attestation_sha256=bindings["pool_use_attestation"]["sha256"],
        )
        if stage_gate.get("execution_ready") is not True:
            raise ValueError("DermLIP stage execution gate requires execution_ready=true")
        if (
            stage_gate.get("verified_for_use") is not False
            or stage_gate.get("patient_independence_accredited") is not False
        ):
            raise ValueError(
                "DermLIP stage execution gate requires verified_for_use=false and "
                "patient_independence_accredited=false"
            )
        if stage_gate.get("execution_claim_path") != str(
            Path(config["profile"]["execution_claim_path"]).resolve()
        ):
            raise ValueError("DermLIP stage execution gate claim path does not match runtime config")
    if config.get("ph2", {}).get("manifest_sha256") != bindings.get("ph2_manifest", {}).get("sha256"):
        raise ValueError("DermLIP runtime config PH2 binding mismatch")
    if config.get("dermlip_model", {}).get("manifest_sha256") != bindings.get(
        "dermlip_model_manifest", {}
    ).get("sha256"):
        raise ValueError("DermLIP runtime config model binding mismatch")
    return config


def get_device(device_str: str) -> str:
    if device_str != "auto":
        return device_str
    if torch.cuda.is_available():
        return "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
    return "cpu"


def _execution_profile(
    args: argparse.Namespace,
    stage: str,
    runtime_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return every training/sampling choice that must be pinned by the stage gate."""
    profile = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "ema_decay": args.ema_decay,
        "gamma_semantic": args.gamma_semantic,
        "kappa_uncertainty": args.kappa_uncertainty,
        "seed": args.seed,
        "device": args.device,
        "guidance_enabled": bool(args.enable_dermlip),
        "pilot_samples": getattr(args, "pilot_samples", None),
        "allow_self_consistency_pilot": bool(getattr(args, "allow_self_consistency_pilot", False)),
        "execution_claim_path": (
            str(Path(args.execution_claim_path).resolve())
            if getattr(args, "execution_claim_path", None) is not None
            else None
        ),
        "stage": stage,
    }
    if runtime_config is not None and "hair_removal" in runtime_config.get("profile", {}):
        profile["hair_removal"] = bool(getattr(args, "hair_removal", False))
    elif getattr(args, "hair_removal", False):
        profile["hair_removal"] = True
    return profile


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    runtime_config = _load_selection_runtime_config(args)
    random.seed(args.seed)
    torch.random.default_generator.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir).resolve()
    failure_ledger_path = None if runtime_config is not None else out_dir / "preflight_failures.jsonl"
    from dermssl.segmentation_methods.common.pool_use import (
        V2_POOL_USE_BINDINGS,
        V2_POOL_USE_ATTESTATION_SHA256,
        claim_execution_gate,
        is_v2_pool_manifest,
        validate_execution_gate,
        validate_pool_use_attestation,
        validate_v2_resume_contract,
        write_resume_attempt_receipt,
        build_v2_pool_provenance,
    )

    unlabeled_manifest_arg = getattr(args, "unlabeled_manifest", None)
    pool_use_attestation_arg = getattr(args, "pool_use_attestation", None)
    execution_gate_arg = getattr(args, "execution_gate", None)
    execution_claim_path_arg = getattr(args, "execution_claim_path", None)
    execute = bool(getattr(args, "execute", False))
    v2_pool = is_v2_pool_manifest(unlabeled_manifest_arg)
    allow_legacy_pool = bool(getattr(args, "allow_legacy_pool", False))
    v2_stage = None
    v2_attestation_sha256 = None
    v2_pool_provenance = None
    execution_profile = None
    if execute and not v2_pool:
        raise ValueError("--execute is reserved for v2 manifests with separate stage gates")
    if unlabeled_manifest_arg is not None and not v2_pool:
        raise ValueError(
            "Fresh training requires an attested v2 selection/refit manifest; --allow-legacy-pool cannot bypass this requirement"
        )
    if allow_legacy_pool and v2_pool:
        raise ValueError("--allow-legacy-pool cannot be used to bypass v2 attestation")
    if v2_pool:
        if unlabeled_manifest_arg is None:
            raise ValueError("V2 pool use requires --unlabeled-manifest")
        manifest_doc = json.loads(Path(unlabeled_manifest_arg).read_text(encoding="utf-8"))
        v2_stage = manifest_doc.get("pool_contract", {}).get("stage")
        if v2_stage is None:
            v2_stage = {
                "selection.manifest.json": "selection",
                "refit.manifest.json": "refit",
            }.get(Path(unlabeled_manifest_arg).name)
        if v2_stage not in {"selection", "refit"}:
            raise ValueError("V2 pool manifest must declare the pinned selection or refit stage")
        if runtime_config is not None:
            if v2_stage != runtime_config.get("stage"):
                raise ValueError("V2 pool stage does not match the DermLIP runtime config")
            execution_profile = _execution_profile(args, v2_stage, runtime_config=runtime_config)
            if execution_profile != runtime_config.get("profile"):
                raise ValueError("Resolved training profile does not match the DermLIP runtime config")
        if pool_use_attestation_arg is None:
            raise ValueError("V2 pool use requires --pool-use-attestation")
        receipt_path = Path(unlabeled_manifest_arg).resolve().parent / "receipt.json"
        attestation = validate_pool_use_attestation(
            pool_use_attestation_arg,
            expected_sha256=V2_POOL_USE_ATTESTATION_SHA256,
            expected_bindings=V2_POOL_USE_BINDINGS,
            manifest_path=unlabeled_manifest_arg,
            receipt_path=receipt_path,
            method="dermlip_semi_supervised",
            stage=v2_stage,
        )
        v2_attestation_sha256 = V2_POOL_USE_ATTESTATION_SHA256
        v2_pool_provenance = build_v2_pool_provenance(
            attestation_path=pool_use_attestation_arg,
            attestation=attestation,
            attestation_sha256=v2_attestation_sha256,
            manifest_path=unlabeled_manifest_arg,
            stage=v2_stage,
        )
        if execute:
            if execution_gate_arg is None:
                raise ValueError("V2 --execute requires a separate --execution-gate")
            if execution_claim_path_arg is None:
                raise ValueError("V2 --execute requires the gate-bound --execution-claim-path")
            execution_profile = _execution_profile(args, v2_stage, runtime_config=runtime_config)
            validate_execution_gate(
                execution_gate_arg,
                method="dermlip_semi_supervised",
                stage=v2_stage,
                profile_sha256=_json_sha256(execution_profile),
                profile=execution_profile,
                manifest_path=unlabeled_manifest_arg,
                attestation_sha256=v2_attestation_sha256,
            )
        if getattr(args, "resume", None) is not None and not execute:
            raise ValueError("V2 resume requires explicit --execute and the same enabled stage gate")
    elif (
        execution_gate_arg is not None
        or pool_use_attestation_arg is not None
        or execution_claim_path_arg is not None
    ):
        raise ValueError("Pool-use attestations and execution gates apply only to v2 manifests")

    if v2_pool and args.resume is None and out_dir.exists():
        raise FileExistsError(
            f"V2 fresh runs require a new output namespace; preserve and review existing artifacts: {out_dir}"
        )
    if args.resume is None and not v2_pool and out_dir.exists() and any(p.name not in {"train.log", ".lock"} for p in out_dir.iterdir()):
        raise FileExistsError(f"Refusing to append to nonempty run directory without a matching --resume: {out_dir}")

    device = get_device(args.device)
    LOGGER.info("Starting Semi-Supervised Training on device: %s", device)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    dermlip_model_identity = None
    if args.enable_dermlip:
        if args.dermlip_model_manifest is None:
            raise ValueError(
                "DermLIP guidance requires --dermlip-model-manifest with pinned model/tokenizer revisions "
                "a verifiable terms-acceptance receipt, and a read-only verified Hugging Face cache."
            )
        dermlip_model_identity = load_dermlip_model_manifest(args.dermlip_model_manifest)
        dermlip_model_identity["cache_verification"] = verify_dermlip_hf_cache(dermlip_model_identity)

    # 1. Load PH2 labeled subsets
    ph2_manifest_path = Path(args.ph2_manifest).resolve()
    if not ph2_manifest_path.is_file():
        raise FileNotFoundError(f"PH2 manifest not found: {args.ph2_manifest}")
    ph2_doc = json.loads(ph2_manifest_path.read_text(encoding="utf-8"))

    try:
        ds_train_l = PH2Dataset(
            ph2_manifest_path,
            split="train",
            max_samples=args.pilot_samples,
            hair_removal=args.hair_removal,
            augment=bool(getattr(args, "ph2_geom_aug", False)),
        )
        ds_val_l = PH2Dataset(
            ph2_manifest_path,
            split="val",
            max_samples=args.pilot_samples,
            hair_removal=args.hair_removal,
            augment=False,
        )
    except Exception as exc:
        records = ph2_doc.get("records", [])
        message = str(exc)
        image_id = next(
            (str(record["image_id"]) for record in records if record.get("image_id") and str(record["image_id"]) in message),
            "ph2_manifest",
        )
        if failure_ledger_path is not None:
            failure_ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "image_id": image_id,
                    "reason": f"ph2_preflight:{type(exc).__name__}",
                    "error_type": type(exc).__name__,
                    "error_message": _safe_error_message(exc),
                    "ph2_manifest_sha256": _sha256(ph2_manifest_path),
                }) + "\n")
        raise

    loader_train_l = DataLoader(ds_train_l, batch_size=args.batch_size, shuffle=True, drop_last=True)
    loader_val_l = DataLoader(ds_val_l, batch_size=args.batch_size, shuffle=False)

    LOGGER.info("Loaded Labeled PH2: %d train, %d val", len(ds_train_l), len(ds_val_l))

    # 2. Load and validate the complete independent unlabeled pool.
    unlabeled_manifest_path = None
    unlabeled_manifest_sha256 = None
    all_labeled = ph2_doc["records"]
    if args.unlabeled_manifest is not None:
        unlabeled_manifest_path = Path(args.unlabeled_manifest).resolve()
        unlabeled_paths, unlabeled_manifest_sha256, unlabeled_audit = validate_unlabeled_manifest(
            unlabeled_manifest_path,
            all_labeled,
            ph2_manifest_path.parent,
            failure_ledger_path=failure_ledger_path,
        )
        run_type = (
            "dermlip_guided_semi_supervised_case_disjoint_patient_unverified"
            if args.enable_dermlip
            else "no_dermlip_ablation_case_disjoint_patient_unverified"
        )
    elif args.allow_self_consistency_pilot:
        unlabeled_audit = {
            "records_validated": len(ds_train_l.records),
            "pool_type": "labeled_cases_reused_for_self_consistency_smoke_test",
            "overlap_checks_passed": [],
            "patient_independence_accredited": False,
        }
        unlabeled_paths = [ph2_manifest_path.parent / r["image_path"] for r in ds_train_l.records]
        LOGGER.warning(
            "ATTENTION: Running in SELF-CONSISTENCY SMOKE TEST mode. Unlabeled pool is populated from labeled images without masks. "
            "This does NOT measure independent semi-supervised benefit."
        )
        run_type = "self_consistency_smoke_test_dermlip_guided" if args.enable_dermlip else "self_consistency_smoke_test_no_dermlip_ablation"
    else:
        raise ValueError(
            "Missing required --unlabeled-manifest. The proposed semi-supervised method requires an independent "
            "unlabeled pool (e.g. ISIC 2019). For a non-scientific self-consistency smoke test, explicitly pass "
            "--allow-self-consistency-pilot."
        )

    calibrator = None
    if args.enable_dermlip:
        LOGGER.info("Instantiating DermLIP calibrator...")
        calibrator = DermLIPSuperpixelCalibrator(
            model_name=dermlip_model_identity["model_name"],
            device=device,
            calibration_mode="dense",
            source_verification=dermlip_model_identity,
        )

    ds_unlabeled = UnlabeledDermDataset(
        unlabeled_paths,
        target_size=DERMLIP_INPUT_SIZE,
        calibrator=calibrator,
        max_samples=args.pilot_samples,
        failure_ledger_path=(
            None if runtime_config is not None else Path(args.out_dir).resolve() / "input_failures.jsonl"
        ),
        hair_removal=args.hair_removal,
        return_slic=(getattr(args, "superpixel_affinity_weight", 0.0) > 0.0),
        n_segments=150,
    )
    if len(ds_train_l) < args.batch_size or len(ds_unlabeled) < args.batch_size:
        raise ValueError("Labeled and unlabeled datasets must each contain at least one full training batch")
    loader_unlabeled = DataLoader(ds_unlabeled, batch_size=args.batch_size, shuffle=True, drop_last=True)
    LOGGER.info("Loaded Unlabeled pool: %d images (run_type: %s)", len(ds_unlabeled), run_type)

    identity_payload = {
        "run_type": run_type,
        "ph2_manifest_sha256": _sha256(ph2_manifest_path),
        "ph2_source_artifact_sha256": ph2_doc.get("source_artifact_sha256"),
        "train_case_ids_sha256": _json_sha256(sorted(r["case_group_id"] for r in ds_train_l.records)),
        "val_case_ids_sha256": _json_sha256(sorted(r["case_group_id"] for r in ds_val_l.records)),
        "unlabeled_manifest_sha256": unlabeled_manifest_sha256,
        "unlabeled_validation_audit_sha256": _json_sha256(unlabeled_audit),
        "pool_provenance": v2_pool_provenance,
        "dermlip_model": dermlip_model_identity,
        "code": _code_provenance(),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "ema_decay": args.ema_decay,
            "gamma_semantic": args.gamma_semantic,
            "kappa_uncertainty": args.kappa_uncertainty,
            "seed": args.seed,
            "device": device,
            "dermlip_guidance": bool(args.enable_dermlip),
            "hair_removal": bool(args.hair_removal),
            "pilot_samples": args.pilot_samples,
            "allow_self_consistency_pilot": bool(args.allow_self_consistency_pilot),
            "perceptual_hash_distance_threshold": 4,
        },
    }
    if v2_pool and not execute:
        return {
            "status": "preflight_validated_execution_disabled",
            "method": "dermlip_semi_supervised",
            "stage": v2_stage,
            "pool_use_attestation_sha256": v2_attestation_sha256,
            "pool_manifest_sha256": unlabeled_manifest_sha256,
            "runtime_config_sha256": (
                _sha256(Path(args.runtime_config).resolve()) if runtime_config is not None else None
            ),
            "pool_provenance": v2_pool_provenance,
            "ph2_manifest_sha256": identity_payload["ph2_manifest_sha256"],
            "dermlip_model_manifest_sha256": (
                dermlip_model_identity["manifest_sha256"] if dermlip_model_identity is not None else None
            ),
            "patient_independence_accredited": False,
            "checkpoint_selection_sources": ["PH2 validation Dice only"],
            "training_started": False,
        }
    identity_fingerprint = _json_sha256(identity_payload)
    identity_path = out_dir / "run_identity.json"
    resume_checkpoint = None
    resume_contract = None
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        if v2_pool:
            if not execute or execution_profile is None:
                raise ValueError("V2 resume requires explicit --execute with its enabled stage gate")
            if not identity_path.is_file():
                raise FileNotFoundError(f"Resume run identity is missing: {identity_path}")
            run_identity_doc = json.loads(identity_path.read_text(encoding="utf-8"))
            resume_contract = validate_v2_resume_contract(
                gate_path=execution_gate_arg,
                manifest_path=unlabeled_manifest_arg,
                profile_sha256=_json_sha256(execution_profile),
                profile=execution_profile,
                attestation_sha256=v2_attestation_sha256,
                claim_path=execution_claim_path_arg,
                out_dir=out_dir,
                resume_path=resume_path,
                run_id=run_identity_doc.get("run_id"),
                identity_fingerprint=identity_fingerprint,
                pool_provenance=v2_pool_provenance,
            )
            validation_path = out_dir / "input_validation.json"
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            if (
                validation.get("ph2_manifest_sha256") != identity_payload["ph2_manifest_sha256"]
                or validation.get("ph2_source_artifact_sha256") != identity_payload["ph2_source_artifact_sha256"]
            ):
                raise ValueError("Resume PH2 input receipt does not match current source hashes")
            if _json_sha256(validation.get("unlabeled_pool")) != identity_payload["unlabeled_validation_audit_sha256"]:
                raise ValueError("Existing input validation audit does not match resume checkpoint identity")
            if validation.get("dermlip_model") != dermlip_model_identity:
                raise ValueError("Existing DermLIP model validation receipt does not match resume checkpoint identity")
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_contract is not None:
            _validate_resume_checkpoint_structure(
                resume_checkpoint,
                run_id=resume_contract["run_id"],
                identity_fingerprint=identity_fingerprint,
                checkpoint_epoch=resume_contract["checkpoint_epoch"],
            )
        required_resume_fields = {
            "run_id", "identity_fingerprint", "epoch", "student_state_dict", "teacher_state_dict",
            "optimizer_state_dict", "rng_state", "history",
            "best_student_val_dice", "best_teacher_val_dice",
        }
        missing_resume_fields = required_resume_fields - set(resume_checkpoint)
        if missing_resume_fields:
            raise ValueError(
                "Resume checkpoint is not a complete resumable checkpoint; missing: "
                + ", ".join(sorted(missing_resume_fields))
            )
        if resume_checkpoint.get("identity_fingerprint") != identity_fingerprint:
            raise ValueError("Resume checkpoint identity does not match current data, code, or configuration")
        run_id = resume_checkpoint.get("run_id")
        if not run_id or not identity_path.is_file():
            raise ValueError("Resume requires a run_identity.json and run_id from the original run")
        stored_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if stored_identity.get("identity_fingerprint") != identity_fingerprint or stored_identity.get("run_id") != run_id:
            raise ValueError("Existing output directory identity does not match resume checkpoint")
        for component in ("student", "teacher"):
            best_path = out_dir / f"best_{component}_model.pt"
            if not best_path.is_file():
                raise ValueError(f"Resume requires the original best {component} checkpoint")
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            if (
                best_checkpoint.get("run_id") != run_id
                or best_checkpoint.get("identity_fingerprint") != identity_fingerprint
                or best_checkpoint.get("checkpoint_component") != component
            ):
                raise ValueError(f"Existing best {component} checkpoint does not match resume run identity")
    else:
        run_id = str(uuid.uuid4())
        if v2_pool and execute:
            claim_execution_gate(
                execution_gate_arg,
                method="dermlip_semi_supervised",
                stage=v2_stage,
                profile_sha256=_json_sha256(execution_profile),
                profile=execution_profile,
                manifest_path=unlabeled_manifest_arg,
                attestation_sha256=v2_attestation_sha256,
                run_id=run_id,
                identity_fingerprint=identity_fingerprint,
                out_dir=out_dir,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        identity_path.write_text(json.dumps({
            "run_id": run_id,
            "identity_fingerprint": identity_fingerprint,
            "output_dir": str(out_dir),
            "identity": identity_payload,
            "patient_independence_accredited": False,
        }, indent=2), encoding="utf-8")
        (out_dir / "input_validation.json").write_text(json.dumps({
            "ph2_manifest_sha256": identity_payload["ph2_manifest_sha256"],
            "ph2_source_artifact_sha256": identity_payload["ph2_source_artifact_sha256"],
            "unlabeled_pool": unlabeled_audit,
            "dermlip_model": dermlip_model_identity,
            "dermlip_model_manifest_sha256": (
                dermlip_model_identity["manifest_sha256"] if dermlip_model_identity is not None else None
            ),
            "patient_independence_accredited": False,
        }, indent=2), encoding="utf-8")

    # 3. Model & Trainer Initialization
    if v2_pool and execute and resume_contract is not None:
        write_resume_attempt_receipt(
            out_dir,
            gate_sha256=resume_contract["gate_sha256"],
            run_id=run_id,
            identity_fingerprint=identity_fingerprint,
            method="dermlip_semi_supervised",
            stage=v2_stage,
            checkpoint_sha256=resume_contract["checkpoint_sha256"],
        )

    if calibrator is not None and getattr(args, "calibrate_ph2_prototypes", False):
        LOGGER.info("Calibrating DermLIP visual-textual prototypes on PH2 train split...")
        prev_aug = ds_train_l.augment
        ds_train_l.augment = False
        try:
            proto_stats = calibrator.fit_labeled_prototypes(ds_train_l)
            LOGGER.info("PH2 prototype calibration completed: %s", proto_stats)
        finally:
            ds_train_l.augment = prev_aug

    backbone_name = getattr(args, "backbone", "scratch")
    student_net = build_segmentation_model(
        backbone=backbone_name,
        in_channels=3,
        num_classes=1,
        base_channels=16,
        pretrained=True,
    )
    trainer = TeacherStudentTrainer(
        student=student_net,
        device=device,
        ema_decay=args.ema_decay,
        gamma_semantic=args.gamma_semantic,
        kappa_uncertainty=args.kappa_uncertainty,
        superpixel_affinity_weight=getattr(args, "superpixel_affinity_weight", 0.0),
        pseudo_label_threshold_high=getattr(args, "pseudo_label_threshold_high", 0.65),
        pseudo_label_threshold_low=getattr(args, "pseudo_label_threshold_low", 0.35),
        soft_kl_weight=getattr(args, "soft_kl_weight", 0.0),
        use_log_linear_fusion=getattr(args, "log_linear_fusion", False),
        copy_paste_prob=getattr(args, "copy_paste_prob", 0.0),
        use_fp_dropout=getattr(args, "fp_dropout", False),
    )
    if hasattr(trainer.student, "get_parameter_groups"):
        optimizer = torch.optim.AdamW(
            trainer.student.get_parameter_groups(args.lr, weight_decay=1e-4)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.lr * 0.05
        )
    else:
        optimizer = torch.optim.AdamW(trainer.student.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = None

    best_student_val_dice = -1.0
    best_teacher_val_dice = -1.0
    history = []
    start_epoch = 1
    if resume_checkpoint is not None:
        trainer.student.load_state_dict(resume_checkpoint["student_state_dict"])
        trainer.teacher.load_state_dict(resume_checkpoint["teacher_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        history = list(resume_checkpoint.get("history", []))
        best_student_val_dice = float(resume_checkpoint["best_student_val_dice"])
        best_teacher_val_dice = float(resume_checkpoint["best_teacher_val_dice"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        if "rng_state" not in resume_checkpoint:
            raise ValueError("Resume checkpoint is missing reproducible RNG state")
        rng_state = resume_checkpoint["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"])
        if torch.cuda.is_available() and rng_state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    metrics_path = out_dir / "metrics.jsonl"
    if resume_checkpoint is not None:
        with metrics_path.open("w", encoding="utf-8") as rebuilt_metrics:
            for entry in history:
                rebuilt_metrics.write(json.dumps(entry) + "\n")
    log_stream = metrics_path.open("a" if resume_checkpoint is not None else "x", encoding="utf-8")

    # 4. Training Loop
    LOGGER.info("Beginning training for %d epochs...", args.epochs)
    start_time = time.time()

    if v2_pool and execute and args.resume is None:
        _write_training_started_receipt(
            out_dir / "training_started.json",
            run_id=run_id,
            identity_fingerprint=identity_fingerprint,
            method="dermlip_semi_supervised",
            stage=v2_stage,
            output_dir=out_dir,
            pool_provenance=v2_pool_provenance,
        )

    u_iter = iter(loader_unlabeled)
    max_steps_arg = getattr(args, "max_steps", None)
    steps_per_epoch = int(max_steps_arg) if max_steps_arg is not None else len(loader_train_l)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        trainer.student.train()
        losses_sup, losses_unsup, losses_sp, alphas = [], [], [], []

        # Ramp up unsupervised consistency weight from 0 to 1 over first 5 epochs
        lambda_u = float(np.clip(epoch / 5.0, 0.0, 1.0))

        l_iter = iter(loader_train_l)
        for batch_idx in range(steps_per_epoch):
            try:
                labeled_batch = next(l_iter)
            except StopIteration:
                l_iter = iter(loader_train_l)
                labeled_batch = next(l_iter)

            try:
                unlabeled_batch = next(u_iter)
            except StopIteration:
                u_iter = iter(loader_unlabeled)
                unlabeled_batch = next(u_iter)

            if len(unlabeled_batch) == 4:
                images_u_weak, images_u_strong, sem_probs_u, slic_u = unlabeled_batch
                unlabeled_batch_core = (images_u_weak, images_u_strong, sem_probs_u)
                step_res = trainer.training_step(
                    labeled_batch,
                    unlabeled_batch_core,
                    lambda_u=lambda_u,
                    unlabeled_slic_labels=slic_u.to(device),
                )
            else:
                step_res = trainer.training_step(labeled_batch, unlabeled_batch, lambda_u=lambda_u)

            optimizer.zero_grad()
            step_res["total_loss"].backward()
            optimizer.step()

            global_step = (epoch - 1) * steps_per_epoch + batch_idx
            trainer.update_teacher(step=global_step)

            losses_sup.append(step_res["loss_sup"].item())
            losses_unsup.append(step_res["loss_unsup"].item())
            losses_sp.append(step_res["loss_sp"].item() if "loss_sp" in step_res else 0.0)
            alphas.append(step_res["mean_alpha"].item())

        if scheduler is not None:
            scheduler.step()

        # Validation step
        val_metrics = trainer.evaluate(loader_val_l)
        epoch_dur = time.time() - epoch_start

        mean_sup = float(np.mean(losses_sup)) if losses_sup else 0.0
        mean_unsup = float(np.mean(losses_unsup)) if losses_unsup else 0.0
        mean_sp = float(np.mean(losses_sp)) if losses_sp else 0.0
        mean_alpha = float(np.mean(alphas)) if alphas else 0.0

        entry = {
            "epoch": epoch,
            "loss_sup": round(mean_sup, 4),
            "loss_unsup": round(mean_unsup, 4),
            "loss_sp": round(mean_sp, 4),
            "mean_alpha": round(mean_alpha, 4),
            "student_dice": round(val_metrics["student_dice"], 4),
            "student_iou": round(val_metrics["student_iou"], 4),
            "teacher_dice": round(val_metrics["teacher_dice"], 4),
            "teacher_iou": round(val_metrics["teacher_iou"], 4),
            "ph2_validation_cases": val_metrics.get("ph2_validation_cases", len(ds_val_l)),
            "lambda_u": round(lambda_u, 3),
            "duration_seconds": round(epoch_dur, 2),
        }
        log_stream.write(json.dumps(entry) + "\n")
        log_stream.flush()
        history.append(entry)

        # Track the best epoch independently for each component using PH2 validation Dice.
        for component in ("student", "teacher"):
            component_dice = val_metrics[f"{component}_dice"]
            if component == "student":
                is_best = component_dice > best_student_val_dice
                if is_best:
                    best_student_val_dice = component_dice
            else:
                is_best = component_dice > best_teacher_val_dice
                if is_best:
                    best_teacher_val_dice = component_dice
            if is_best:
                best_ckpt = out_dir / f"best_{component}_model.pt"
                _atomic_torch_save(
                    {
                        "run_id": run_id,
                        "identity_fingerprint": identity_fingerprint,
                        "identity": identity_payload,
                        "checkpoint_component": component,
                        "backbone": backbone_name,
                        "epoch": epoch,
                        f"{component}_state_dict": getattr(trainer, component).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "ph2_validation_dice": component_dice,
                        "ph2_validation_iou": val_metrics[f"{component}_iou"],
                        "ph2_validation_cases": val_metrics.get("ph2_validation_cases", len(ds_val_l)),
                    },
                    best_ckpt,
                )

        LOGGER.info(
            "Epoch %d/%d — sup_loss: %.4f, unsup_loss: %.4f, student_dice: %.4f, teacher_dice: %.4f, best_student: %.4f, best_teacher: %.4f, ph2_validation_cases: %d (%.1fs)",
            epoch,
            args.epochs,
            mean_sup,
            mean_unsup,
            val_metrics["student_dice"],
            val_metrics["teacher_dice"],
            best_student_val_dice,
            best_teacher_val_dice,
            val_metrics.get("ph2_validation_cases", len(ds_val_l)),
            epoch_dur,
        )

        last_ckpt = out_dir / "last_checkpoint.pt"
        _atomic_torch_save(
            {
                "run_id": run_id,
                "identity_fingerprint": identity_fingerprint,
                "identity": identity_payload,
                "backbone": backbone_name,
                "epoch": epoch,
                "teacher_state_dict": trainer.teacher.state_dict(),
                "student_state_dict": trainer.student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rng_state": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch_cpu": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                "best_student_val_dice": best_student_val_dice,
                "best_teacher_val_dice": best_teacher_val_dice,
                "history": history,
            },
            last_ckpt,
        )
        if v2_pool and execute:
            _write_checkpoint_receipt(
                last_ckpt,
                run_id=run_id,
                identity_fingerprint=identity_fingerprint,
                method="dermlip_semi_supervised",
                stage=v2_stage,
                output_dir=out_dir,
                epoch=epoch,
                pool_provenance=v2_pool_provenance,
            )

    log_stream.close()
    total_time = time.time() - start_time
    LOGGER.info(
        "Training completed in %.1f seconds. Best PH2 validation Dice: student %.4f, teacher %.4f",
        total_time,
        best_student_val_dice,
        best_teacher_val_dice,
    )

    summary = {
        "status": "completed",
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
        "run_type": run_type,
        "patient_independence_accredited": False,
        "ph2_manifest_sha256": identity_payload["ph2_manifest_sha256"],
        "ph2_source_artifact_sha256": identity_payload["ph2_source_artifact_sha256"],
        "unlabeled_manifest_sha256": identity_payload["unlabeled_manifest_sha256"],
        "pool_provenance": v2_pool_provenance,
        "total_epochs": args.epochs,
        "best_student_ph2_validation_dice": best_student_val_dice,
        "best_teacher_ph2_validation_dice": best_teacher_val_dice,
        "ph2_validation_cases": val_metrics.get("ph2_validation_cases", len(ds_val_l)) if "val_metrics" in locals() else 0,
        "final_student_dice": val_metrics["student_dice"] if "val_metrics" in locals() else None,
        "final_teacher_dice": val_metrics["teacher_dice"] if "val_metrics" in locals() else None,
        "kappa_uncertainty": args.kappa_uncertainty,
        "gamma_semantic": args.gamma_semantic,
        "ema_decay": args.ema_decay,
        "history": history,
        "out_dir": str(out_dir),
        "best_student_checkpoint_sha256": _sha256(out_dir / "best_student_model.pt") if (out_dir / "best_student_model.pt").is_file() else None,
        "best_teacher_checkpoint_sha256": _sha256(out_dir / "best_teacher_model.pt") if (out_dir / "best_teacher_model.pt").is_file() else None,
        "last_checkpoint_sha256": _sha256(out_dir / "last_checkpoint.pt") if (out_dir / "last_checkpoint.pt").is_file() else None,
        "total_duration_seconds": round(total_time, 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
