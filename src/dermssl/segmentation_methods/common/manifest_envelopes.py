"""Materialize hash-bound, image-only runner envelopes from frozen manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .protocol import POOL_V2_RETROSPECTIVE_CONTRACT


IMAGE_FIELDS = {"image_id", "dataset", "image_path", "image_sha256"}
MASK_FIELDS = {"mask", "mask_path", "mask_sha256", "expert_mask"}


def _read_json(path: Path) -> tuple[dict, bytes]:
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value, raw


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_image_rows(source: dict, source_path: Path | PurePosixPath,
                      partition: str) -> list[dict]:
    """Project source records onto the exact image-only runner row schema."""
    records = source.get("records")
    if not isinstance(records, list):
        raise ValueError("Source manifest records must be a list")
    if partition == "selection":
        if any(not isinstance(row, dict) or set(row) != IMAGE_FIELDS for row in records):
            raise ValueError("Selection source rows must be exact image-only records")
        return records
    dataset = source.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("Evaluation source dataset is required")
    dataset = dataset.replace("_Task1_Training", "")
    rows = []
    for record in records:
        image = record.get("image") if isinstance(record, dict) else None
        if (not isinstance(record, dict) or not isinstance(image, dict)
                or not isinstance(image.get("relative_path"), str)
                or not isinstance(image.get("sha256"), str)):
            raise ValueError("Evaluation source requires original image metadata")
        path_type = PurePosixPath if isinstance(source_path, PurePosixPath) else Path
        rel_path = path_type(image["relative_path"])
        image_path = rel_path if rel_path.is_absolute() else source_path.parent / rel_path
        rows.append({"image_id": record.get("image_id"), "dataset": dataset,
                     "image_path": str(image_path), "image_sha256": image["sha256"]})
    return rows


def validate_source_binding(document: dict) -> dict:
    """Verify source bytes and original image-row identity for a bound envelope."""
    reference = document.get("source_manifest")
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference.get("path"), str)
            or re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", "")) is None):
        raise ValueError("Source manifest path and SHA-256 are required")
    source_path = Path(reference["path"])
    source, raw = _read_json(source_path)
    if sha256_bytes(raw) != reference["sha256"]:
        raise ValueError("Source manifest SHA-256 mismatch")
    partition = document.get("partition")
    role = "selection" if partition == "train" else "evaluation" if partition == "evaluation" else None
    if role is None:
        raise ValueError("Source-bound envelope partition must be train or evaluation")
    original_rows = source_image_rows(source, source_path, role)
    if document.get("records") != original_rows:
        raise ValueError("Envelope records differ from source manifest original image rows")
    contract = document.get("pool_contract")
    if not isinstance(contract, dict) or contract != POOL_V2_RETROSPECTIVE_CONTRACT:
        raise ValueError("Source-bound manifest must match the exact pool-v2 retrospective contract")
    expected_hash = (contract["selection_manifest_sha256"] if role == "selection"
                     else contract["evaluation_manifest_sha256"])
    if reference["sha256"] != expected_hash:
        raise ValueError("Source manifest SHA-256 does not match the pool-v2 contract")
    if role == "selection":
        eligibility = document.get("eligibility", {})
        pins = {
            "source_manifest_sha256": contract["source_pool_manifest_sha256"],
            "source_receipt_sha256": contract["source_receipt_sha256"],
            "exclusions_sha256": contract["source_exclusions_sha256"],
        }
        if any(eligibility.get(key) != value for key, value in pins.items()):
            raise ValueError("Selection eligibility source, receipt, or exclusions pin mismatch")
    else:
        if any(MASK_FIELDS.intersection(row) for row in document["records"]):
            raise ValueError("Expert masks must remain outside evaluation learning records")
        if (len(original_rows) != contract["evaluation_count"]
                or _cohort_digest(original_rows) != contract["evaluation_cohort_sha256"]):
            raise ValueError("Evaluation image cohort does not match the pool-v2 contract")
    return source


def _cohort_digest(rows: list[dict]) -> str:
    normalized = sorted((row["image_id"], row["image_sha256"]) for row in rows)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def materialize_manifest_envelope(source_manifest: Path, protocol_path: Path,
                                  protocol_sha256: str, pool_contract: dict,
                                  partition: str, *,
                                  source_reference_path: str | None = None,
                                  protocol_reference_path: str | None = None) -> dict:
    """Build a draft envelope without changing its source manifest or opening gates."""
    if partition not in {"selection", "evaluation"}:
        raise ValueError("Partition must be selection or evaluation")
    if (source_reference_path is not None
            and (not isinstance(source_reference_path, str)
                 or not PurePosixPath(source_reference_path).is_absolute())):
        raise ValueError("Source reference path must be an absolute POSIX path")
    if (protocol_reference_path is not None
            and (not isinstance(protocol_reference_path, str)
                 or not PurePosixPath(protocol_reference_path).is_absolute())):
        raise ValueError("Protocol reference path must be an absolute POSIX path")
    if pool_contract != POOL_V2_RETROSPECTIVE_CONTRACT:
        raise ValueError("Pool contract differs from the pinned pool-v2 retrospective contract")
    source_path = Path(source_manifest).resolve()
    source, raw = _read_json(source_path)
    source_sha256 = sha256_bytes(raw)
    expected_source_sha256 = (pool_contract["selection_manifest_sha256"] if partition == "selection"
                              else pool_contract["evaluation_manifest_sha256"])
    if source_sha256 != expected_source_sha256:
        raise ValueError("Source manifest SHA-256 does not match the pool-v2 contract")
    protocol, protocol_raw = _read_json(Path(protocol_path))
    if sha256_bytes(protocol_raw) != protocol_sha256:
        raise ValueError("Retrospective protocol SHA-256 mismatch")
    if protocol.get("learning_pool_contract") != pool_contract:
        raise ValueError("Protocol learning pool contract mismatch")
    if partition == "evaluation":
        _validate_pair_receipt(protocol, pool_contract)
    rows = source_image_rows(source, source_path, partition)
    contract = pool_contract
    common = {
        "schema_version": 1,
        "partition": "train" if partition == "selection" else "evaluation",
        "count": len(rows),
        "pool_contract": contract,
        "source_manifest": {"path": str(source_path), "sha256": source_sha256},
        "evaluation_protocol": {"protocol_id": protocol["protocol_id"],
                                 "path": str(Path(protocol_path).resolve()),
                                 "sha256": protocol_sha256},
        "records": rows,
        "execution_enabled": False,
        "execution_ready": False,
        "launch_ready": False,
        "training_started": False,
        "patient_independence_accredited": False,
    }
    if partition == "selection":
        eligibility = dict(source.get("eligibility", {}))
        if contract.get("patient_independence_accredited") is not False:
            raise ValueError("Selection pool must not claim accredited patient independence")
        eligibility["patient_independence_accredited"] = False
        common["eligibility"] = eligibility
        if any(common["eligibility"].get(key) != value for key, value in {
            "source_manifest_sha256": contract["source_pool_manifest_sha256"],
            "source_receipt_sha256": contract["source_receipt_sha256"],
            "exclusions_sha256": contract["source_exclusions_sha256"],
        }.items()):
            raise ValueError("Selection source eligibility pins differ from the protocol contract")
    else:
        common["evaluation_scope"] = {
            "purpose": "retrospective_descriptive_image_level_scoring_only",
            "selection_or_threshold_tuning": False,
            "holdout_novelty": False,
            "patient_independence_accredited": False,
        }
        if (len(rows) != contract["evaluation_count"]
                or _cohort_digest(rows) != contract["evaluation_cohort_sha256"]):
            raise ValueError("Evaluation source image cohort does not match the protocol contract")
        if any(MASK_FIELDS.intersection(row) for row in rows):
            raise ValueError("Expert masks must remain outside evaluation learning records")
    validate_source_binding(common)
    if source_reference_path is not None:
        common["source_manifest"]["path"] = source_reference_path
        if partition == "evaluation":
            common["records"] = source_image_rows(
                source, PurePosixPath(source_reference_path), partition)
    if protocol_reference_path is not None:
        common["evaluation_protocol"]["path"] = protocol_reference_path
    return common


def _validate_pair_receipt(protocol: dict, contract: dict) -> dict:
    reference = protocol.get("evaluation_pair_receipt")
    if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
            or reference.get("sha256") != contract["evaluation_pair_receipt_sha256"]):
        raise ValueError("Evaluation pair receipt reference differs from the pool-v2 contract")
    receipt, raw = _read_json(Path(reference["path"]))
    if sha256_bytes(raw) != reference["sha256"]:
        raise ValueError("Evaluation pair receipt SHA-256 mismatch")
    membership = receipt.get("official_zip_membership", {})
    checks = receipt.get("integrity_checks", {})
    overlaps = receipt.get("overlaps", {})
    if (receipt.get("manifest_sha256") != contract["evaluation_pair_manifest_sha256"]
            or receipt.get("schema") != "isic2018_task1_full_pair_manifest_receipt.v1"
            or receipt.get("verified_for_use") is not False
            or receipt.get("patient_independence_accredited") not in (None, False)
            or membership.get("expected_count") != contract["evaluation_count"]
            or membership.get("image_count") != contract["evaluation_count"]
            or membership.get("mask_count") != contract["evaluation_count"]
            or membership.get("pairing") != "one_to_one_by_canonical_full_image_id"
            or set(checks) != {"all_images_readable_jpeg", "all_masks_readable_binary_png",
                               "all_pair_dimensions_match", "extracted_bytes_match_official_zip_members"}
            or any(value is not True for value in checks.values())
            or set(overlaps) != {"selection", "refit"}):
        raise ValueError("Evaluation pair receipt does not match the reviewed 2,594-case cohort")
    for stage in ("selection", "refit"):
        evidence = overlaps[stage]
        if any(evidence.get(key) != 0 for key in
               ("exact_id_overlap_count", "base_id_overlap_count", "image_sha256_overlap_count")):
            raise ValueError("Evaluation pair receipt reports overlap with the learning pool")
    return receipt
