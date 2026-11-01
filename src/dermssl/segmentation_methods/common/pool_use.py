"""Method-scoped consumption attestations and separate execution gates."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from datetime import datetime, timezone


IMAGE_ONLY_FIELDS = ["image_id", "dataset", "image_path", "image_sha256"]
ALLOWED_METHODS = {"dermlip_semi_supervised", "sled"}
EXECUTION_FLAGS = ("execution_enabled", "launch_ready")
V2_POOL_USE_ATTESTATION_SHA256 = "aeaa24beae7857472ad07ca6b5e9e92478b7bbc66d1eb98ad7be1bb7c6f5385b"
V2_POOL_USE_BINDINGS = {
    "pool_root": "/home/jovyan/work/dermlip_slic_v8/preflight/shared-segmentation-pools-v2-20261022-001/",
    "preflight_receipt": {
        "path": "/home/jovyan/work/dermlip_slic_v8/preflight/shared-segmentation-pools-v2-20261022-001/receipt.json",
        "sha256": "dafaaa005b6352c0b786f91e1cdf9575b74e816f172fbd7bcd3b7c5e9929d265",
    },
    "source_artifacts": {
        "selection_source_manifest_sha256": "331457804f6bde1b65c4f184a1a529d9a6e4802b0349f6aebc9f2ab6d1945dc8",
        "refit_source_manifest_sha256": "21903961c680cf8cc3bdffeb8be394e2e32aec2ca93b295eea47eeb859d0bea0",
        "ph2_manifest_sha256": "d2a159bb1f846e0ee2195406268d457d3c1b5a26a650c201b31e9f815371b9bf",
        "ph2_exclusions_sha256": "b3253b3c1748f4057b497b8e2ceea6bfb4aa91344bba932b4c63d22253e9c09e",
        "dermlip_model_manifest_sha256": "f1bcc6e2f5c5eb102f2565bbb920c0298e0356cc1db575e5dc5f7203fa6ec45f",
        "terms_acceptance_sha256": "5f6ef7d4cf7e810bddb053058bcd5dfaa9c56181e97ffcd62eb9356c47c598a9",
        "evaluation_manifest_sha256": "53009a68073d6ae1048063a9328a6e7bc6b9ef34cdf0096d64ebaa866fe0f179",
        "evaluation_manifest_receipt_sha256": "25f42c626ba6c4db8583b0220dcbd2f0450665fbbe3c62133741859988dc35a8",
    },
    "stages": {
        "selection": {
            "path": "/home/jovyan/work/dermlip_slic_v8/preflight/shared-segmentation-pools-v2-20261022-001/selection.manifest.json",
            "sha256": "222fb1eabe5813ce5e5de328d91cba6c6ee2225a36c552f83e96b61f4d627809",
            "record_count": 34330,
            "dataset_counts": {"ISIC2019": 14411, "ISIC2020": 19919},
            "cohort_sha256": "469c275f2a22bd21a2cf496f4a85701f53776adbaea21f07aac1827de69810c2",
        },
        "refit": {
            "path": "/home/jovyan/work/dermlip_slic_v8/preflight/shared-segmentation-pools-v2-20261022-001/refit.manifest.json",
            "sha256": "8abd989da36072607db81845928dbd6af338536243b6474e9873c3b613b9e088",
            "record_count": 42977,
            "dataset_counts": {"ISIC2019": 18053, "ISIC2020": 24924},
            "cohort_sha256": "ab65cb7273ed5402ee9e8a7509fd3fe7c1f417b0f0f14af2ebd994577bb75cba",
        },
    },
    "ph2": {"eligible_count": 195, "train_count": 156, "validation_count": 39, "excluded_count": 5},
    "model": {"license": "CC-BY-NC-ND-4.0"},
    "holdout_novelty": False,
    "patient_independence_accredited": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is missing or unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return value


def validate_pool_use_attestation(
    attestation_path: Path,
    *,
    expected_sha256: str,
    expected_bindings: dict[str, Any],
    manifest_path: Path,
    receipt_path: Path,
    method: str,
    stage: str,
) -> dict[str, Any]:
    """Verify a separately pinned method/stage consumption claim and its inputs."""
    if method not in ALLOWED_METHODS:
        raise ValueError(f"Pool-use attestation does not authorize method {method!r}")
    path = Path(attestation_path)
    if not path.is_file():
        raise ValueError("Pool-use attestation is missing")
    if _sha256(path) != expected_sha256:
        raise ValueError("Pool-use attestation SHA-256 pin mismatch")
    attestation = _read_json(path, description="Pool-use attestation")
    if attestation.get("schema") != "v2_method_pool_use_attestation.v1":
        raise ValueError("Pool-use attestation schema mismatch")
    if attestation.get("status") != "pool_consumption_attested_execution_disabled":
        raise ValueError("Pool-use attestation status mismatch")
    if method not in attestation.get("method_scopes", []):
        raise ValueError(f"Pool-use attestation does not include method {method!r}")
    if attestation.get("stage") not in {stage, "selection_and_refit"}:
        raise ValueError("Pool-use attestation stage mismatch")
    if attestation.get("bindings") != expected_bindings:
        raise ValueError("Pool-use attestation artifact pins mismatch")
    if attestation.get("preflight_receipt") != expected_bindings.get("preflight_receipt"):
        raise ValueError("Pool-use attestation preflight receipt pin mismatch")
    if attestation.get("schema_fields") != IMAGE_ONLY_FIELDS:
        raise ValueError("Pool-use attestation schema is not the exact image-only schema")
    if attestation.get("pool_manifest_frozen") is not True:
        raise ValueError("Pool-use attestation must pin the pool manifest SHA identity")
    if attestation.get("freeze_semantics") != "sha_identity_pinned_filesystem_immutability_not_claimed":
        raise ValueError("Pool-use attestation freeze semantics mismatch")
    if attestation.get("training_consumable") is not True:
        raise ValueError("Pool-use attestation does not authorize pool consumption")
    if attestation.get("patient_independence_accredited") is not False:
        raise ValueError("Pool-use attestation must not accredit patient independence")
    if attestation.get("holdout_novelty") is not False:
        raise ValueError("Pool-use attestation must not claim holdout novelty")

    manifest = attestation.get("stage_manifests", {}).get(stage)
    if not isinstance(manifest, dict):
        raise ValueError("Pool-use attestation is missing the requested stage manifest")
    manifest_path = Path(manifest_path).resolve()
    receipt_path = Path(receipt_path).resolve()
    pool_root_value = expected_bindings.get("pool_root")
    if not isinstance(pool_root_value, str) or not Path(pool_root_value).is_absolute():
        raise ValueError("Pool-use attestation must bind an absolute v2 pool root")
    pool_root = Path(pool_root_value).resolve()
    try:
        manifest_path.relative_to(pool_root)
    except ValueError as exc:
        raise ValueError("Pool-use attestation manifest is outside the bound v2 pool root") from exc
    if manifest_path.name != f"{stage}.manifest.json":
        raise ValueError("Pool-use attestation requires the explicit stage manifest filename")
    if manifest.get("path") != str(manifest_path):
        raise ValueError("Pool-use attestation manifest path mismatch")
    expected_stage = expected_bindings.get("stages", {}).get(stage)
    if not isinstance(expected_stage, dict) or any(
        manifest.get(key) != expected_stage.get(key)
        for key in ("path", "sha256", "record_count", "dataset_counts", "cohort_sha256")
    ):
        raise ValueError("Pool-use attestation stage manifest pins mismatch")
    if not manifest_path.is_file() or _sha256(manifest_path) != manifest.get("sha256"):
        raise ValueError("Pool-use attestation manifest SHA-256 mismatch")
    try:
        manifest_document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Pool-use attestation manifest is missing or unreadable") from exc
    if not isinstance(manifest_document, dict):
        raise ValueError("Pool-use attestation manifest must be a JSON object")
    pool_contract = manifest_document.get("pool_contract")
    if (
        manifest_document.get("schema_version") != 1
        or manifest_document.get("partition") != "train"
        or not isinstance(pool_contract, dict)
        or pool_contract.get("contract_id") != "classification_split_master_v2"
        or pool_contract.get("stage") != stage
    ):
        raise ValueError("Pool-use attestation requires the matching v2 stage manifest schema")
    rows = manifest_document.get("records")
    if (
        not isinstance(rows, list)
        or not rows
        or type(manifest_document.get("count")) is not int
        or manifest_document["count"] != len(rows)
        or len(rows) != manifest.get("record_count")
    ):
        raise ValueError("Pool-use attestation manifest count or records mismatch")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    dataset_counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(IMAGE_ONLY_FIELDS):
            raise ValueError("Pool-use attestation manifest rows must use the exact image-only schema")
        image_id = row.get("image_id")
        dataset = row.get("dataset")
        image_path = row.get("image_path")
        image_sha256 = row.get("image_sha256")
        if (
            not isinstance(image_id, str) or not image_id
            or not isinstance(dataset, str) or not dataset
            or not isinstance(image_path, str) or not Path(image_path).is_absolute()
            or not isinstance(image_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", image_sha256)
            or image_id in seen_ids or image_sha256 in seen_hashes
        ):
            raise ValueError("Pool-use attestation manifest contains invalid or duplicate image identities")
        seen_ids.add(image_id)
        seen_hashes.add(image_sha256)
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    if dataset_counts != manifest.get("dataset_counts"):
        raise ValueError("Pool-use attestation manifest dataset counts mismatch")
    from .protocol import cohort_digest
    if cohort_digest(rows) != manifest.get("cohort_sha256"):
        raise ValueError("Pool-use attestation manifest cohort SHA-256 mismatch")
    receipt = attestation.get("preflight_receipt")
    if not isinstance(receipt, dict) or receipt.get("path") != str(receipt_path):
        raise ValueError("Pool-use attestation preflight receipt path mismatch")
    if not receipt_path.is_file() or _sha256(receipt_path) != receipt.get("sha256"):
        raise ValueError("Pool-use attestation preflight receipt SHA-256 mismatch")
    return attestation


def build_v2_pool_provenance(
    *,
    attestation_path: Path,
    attestation: dict[str, Any],
    attestation_sha256: str,
    manifest_path: Path,
    stage: str,
) -> dict[str, Any]:
    """Build receipt-ready provenance from one validated v2 pool stage."""
    attestation_path = Path(attestation_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    if stage not in {"selection", "refit"}:
        raise ValueError("V2 pool provenance requires selection or refit stage")
    if not attestation_path.is_file() or _sha256(attestation_path) != attestation_sha256:
        raise ValueError("V2 pool provenance attestation hash mismatch")
    manifest_pin = attestation.get("stage_manifests", {}).get(stage)
    receipt_pin = attestation.get("preflight_receipt")
    bindings = attestation.get("bindings", {})
    if (
        not isinstance(manifest_pin, dict)
        or manifest_pin.get("path") != str(manifest_path)
        or manifest_pin.get("sha256") != _sha256(manifest_path)
        or not isinstance(receipt_pin, dict)
        or receipt_pin.get("sha256") != bindings.get("preflight_receipt", {}).get("sha256")
        or not isinstance(bindings.get("source_artifacts"), dict)
    ):
        raise ValueError("V2 pool provenance is not bound to the validated manifest and receipt")
    return {
        "schema": "dermlip_v2_pool_provenance.v1",
        "pool_stage": stage,
        "pool_root": str(Path(str(bindings.get("pool_root", ""))).resolve()),
        "pool_manifest_path": str(manifest_path),
        "pool_manifest_sha256": manifest_pin["sha256"],
        "pool_record_count": manifest_pin["record_count"],
        "pool_dataset_counts": manifest_pin["dataset_counts"],
        "pool_cohort_sha256": manifest_pin["cohort_sha256"],
        "pool_use_attestation_path": str(attestation_path),
        "pool_use_attestation_sha256": attestation_sha256,
        "preflight_receipt_path": receipt_pin["path"],
        "preflight_receipt_sha256": receipt_pin["sha256"],
        "source_artifact_sha256": dict(bindings["source_artifacts"]),
        "fresh_model_initialization": True,
        "legacy_checkpoint_reused": False,
        "legacy_feature_cache_reused": False,
        "legacy_pseudomasks_reused": False,
    }


def is_v2_pool_manifest(manifest_path: Path | None) -> bool:
    """Identify v2 manifests without changing behavior for legacy inputs."""
    if manifest_path is None or not Path(manifest_path).is_file():
        return False
    try:
        document = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    contract = document.get("pool_contract", {}) if isinstance(document, dict) else {}
    receipt_path = Path(manifest_path).resolve().parent / "receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        receipt = {}
    v2_plan_path = any(
        parent.name == "shared-segmentation-pools-v2-20261022-001"
        for parent in Path(manifest_path).resolve().parents
    ) and Path(manifest_path).name in {"selection.manifest.json", "refit.manifest.json"}
    return v2_plan_path or (
        isinstance(contract, dict)
        and contract.get("stage") in {"selection", "refit"}
    ) or (isinstance(receipt, dict) and receipt.get("schema") == "v2_shared_segmentation_pool_bridge.v1")


def validate_execution_gate(
    gate_path: Path,
    *,
    method: str,
    stage: str,
    profile_sha256: str,
    profile: dict[str, Any],
    manifest_path: Path,
    attestation_sha256: str,
) -> dict[str, Any]:
    """Require a separately enabled gate pinned to the exact method inputs/profile."""
    gate = _read_json(Path(gate_path), description="Execution gate")
    if gate.get("schema") != "dermlip_stage_execution_gate.v1":
        raise ValueError("Execution gate schema mismatch")
    if gate.get("method") != method or gate.get("stage") != stage:
        raise ValueError("Execution gate method or stage mismatch")
    if gate.get("profile_sha256") != profile_sha256:
        raise ValueError("Execution gate profile SHA-256 pin mismatch")
    gate_profile = gate.get("profile")
    if not isinstance(gate_profile, dict):
        raise ValueError("Execution gate profile must be a JSON object")
    if gate_profile.get("pilot_samples") is None and profile.get("pilot_samples") is not None:
        raise ValueError("Execution gate forbids pilot_samples for this profile")
    if (
        gate_profile.get("allow_self_consistency_pilot") is False
        and profile.get("allow_self_consistency_pilot") is True
    ):
        raise ValueError("Execution gate forbids allow_self_consistency_pilot for this profile")
    if gate_profile != profile:
        raise ValueError("Execution gate profile content does not match requested profile")
    profile_digest = hashlib.sha256(
        json.dumps(gate_profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if profile_digest != profile_sha256:
        raise ValueError("Execution gate profile content does not match its SHA-256 pin")
    pool = gate.get("pool_manifest", {})
    manifest_path = Path(manifest_path).resolve()
    if (
        not isinstance(pool, dict)
        or pool.get("path") != str(manifest_path)
        or pool.get("sha256") != _sha256(manifest_path)
    ):
        raise ValueError("Execution gate pool manifest SHA-256 pin mismatch")
    if gate.get("pool_use_attestation_sha256") != attestation_sha256:
        raise ValueError("Execution gate pool-use attestation SHA-256 pin mismatch")
    for flag in EXECUTION_FLAGS:
        if gate.get(flag) is not True:
            raise ValueError(f"Execution gate requires {flag}=true before --execute")
    if gate.get("training_started") is not False:
        raise ValueError("New-run execution gate requires training_started=false")
    if gate.get("pool_manifest_frozen") is not True or gate.get("training_consumable") is not True:
        raise ValueError("Execution gate requires an attested, SHA-pinned consumable pool")
    claim_path = profile.get("execution_claim_path")
    if (
        not isinstance(claim_path, str)
        or not Path(claim_path).is_absolute()
        or gate.get("execution_claim_path") != str(Path(claim_path).resolve())
        or stage not in Path(claim_path).name
    ):
        raise ValueError("Execution gate must bind a unique absolute, stage-specific execution_claim_path")
    return gate


def claim_execution_gate(
    gate_path: Path,
    *,
    method: str,
    stage: str,
    profile_sha256: str,
    profile: dict[str, Any],
    manifest_path: Path,
    attestation_sha256: str,
    run_id: str,
    identity_fingerprint: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Validate a new-run gate and atomically consume its one-time execution claim."""
    gate = validate_execution_gate(
        gate_path,
        method=method,
        stage=stage,
        profile_sha256=profile_sha256,
        profile=profile,
        manifest_path=manifest_path,
        attestation_sha256=attestation_sha256,
    )
    claim_path = Path(profile["execution_claim_path"]).resolve()
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dermlip_execution_claim.v1",
        "gate_sha256": _sha256(Path(gate_path)),
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
        "method": method,
        "stage": stage,
        "output_dir": str(Path(out_dir).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        descriptor = os.open(
            claim_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise FileExistsError(f"Execution claim already exists; this gate cannot be reused: {claim_path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def validate_v2_resume_contract(
    *,
    gate_path: Path,
    manifest_path: Path,
    profile_sha256: str,
    profile: dict[str, Any],
    attestation_sha256: str,
    claim_path: Path,
    out_dir: Path,
    resume_path: Path,
    run_id: str,
    identity_fingerprint: str,
    pool_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every persisted artifact required to resume one exact v2 run."""
    method = "dermlip_semi_supervised"
    stage = str(profile.get("stage", ""))
    gate = validate_execution_gate(
        gate_path,
        method=method,
        stage=stage,
        profile_sha256=profile_sha256,
        profile=profile,
        manifest_path=manifest_path,
        attestation_sha256=attestation_sha256,
    )
    gate_sha256 = _sha256(Path(gate_path))
    claim_path = Path(claim_path).resolve()
    expected_claim_path = Path(str(gate.get("execution_claim_path", ""))).resolve()
    if claim_path != expected_claim_path:
        raise ValueError("Resume execution claim path does not match the gate")
    claim = _read_json(claim_path, description="Execution claim")
    expected_identity = {
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
        "method": method,
        "stage": stage,
        "output_dir": str(Path(out_dir).resolve()),
    }
    if claim.get("schema") != "dermlip_execution_claim.v1":
        raise ValueError("Resume execution claim schema mismatch")
    if claim.get("gate_sha256") != gate_sha256:
        raise ValueError("Resume claim gate SHA-256 mismatch")
    if any(claim.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Resume execution claim identity or method/stage mismatch")

    out_dir = Path(out_dir).resolve()
    checkpoint_path = out_dir / "last_checkpoint.pt"
    if Path(resume_path).resolve() != checkpoint_path:
        raise ValueError("V2 resume must use the last checkpoint under the original output directory")
    identity_path = out_dir / "run_identity.json"
    start_receipt_path = out_dir / "training_started.json"
    input_validation_path = out_dir / "input_validation.json"
    checkpoint_receipt_path = out_dir / "last_checkpoint.receipt.json"
    for required_path, description in (
        (identity_path, "run identity"),
        (start_receipt_path, "training-start receipt"),
        (input_validation_path, "input-validation receipt"),
        (checkpoint_path, "last checkpoint"),
        (checkpoint_receipt_path, "last-checkpoint receipt"),
    ):
        if not required_path.is_file():
            raise FileNotFoundError(f"Resume {description} is missing: {required_path}")

    stored_identity = _read_json(identity_path, description="Run identity")
    stored_run_id = stored_identity.get("run_id")
    if not isinstance(stored_run_id, str) or not stored_run_id:
        raise ValueError("Resume run identity is missing its run_id")
    if run_id != stored_run_id:
        raise ValueError("Resume run identity run_id mismatch")
    expected_identity["run_id"] = stored_run_id
    if (
        stored_identity.get("identity_fingerprint") != identity_fingerprint
        or stored_identity.get("output_dir") != str(out_dir)
    ):
        raise ValueError("Resume run identity does not match the current data/model/profile identity")
    start_receipt = _read_json(start_receipt_path, description="Training-start receipt")
    if start_receipt.get("schema") != "dermlip_training_started_receipt.v1":
        raise ValueError("Resume training-start receipt schema mismatch")
    if any(start_receipt.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("Resume training-start receipt identity or method/stage mismatch")
    if pool_provenance is not None and start_receipt.get("pool_provenance") != pool_provenance:
        raise ValueError("Resume training-start receipt pool provenance mismatch")

    checkpoint_receipt = _read_json(checkpoint_receipt_path, description="Last-checkpoint receipt")
    observed_checkpoint_sha256 = _sha256(checkpoint_path)
    checkpoint_epoch = checkpoint_receipt.get("epoch")
    if (
        checkpoint_receipt.get("schema") != "dermlip_checkpoint_receipt.v1"
        or checkpoint_receipt.get("sha256") != observed_checkpoint_sha256
        or any(checkpoint_receipt.get(key) != value for key, value in expected_identity.items())
        or (pool_provenance is not None and checkpoint_receipt.get("pool_provenance") != pool_provenance)
        or not isinstance(checkpoint_epoch, int)
        or checkpoint_epoch < 1
    ):
        raise ValueError("Last-checkpoint hash or receipt identity/structure is inconsistent")
    return {
        "run_id": stored_run_id,
        "identity_fingerprint": identity_fingerprint,
        "gate_sha256": gate_sha256,
        "checkpoint_sha256": observed_checkpoint_sha256,
        "checkpoint_epoch": checkpoint_epoch,
    }


def write_resume_attempt_receipt(
    out_dir: Path,
    *,
    gate_sha256: str,
    run_id: str,
    identity_fingerprint: str,
    method: str,
    stage: str,
    checkpoint_sha256: str,
) -> Path:
    """Record one explicitly requested resume without overwriting the start claim."""
    attempts_dir = Path(out_dir).resolve() / "resume_attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    attempt_id = os.urandom(16).hex()
    attempt_path = attempts_dir / f"{attempt_id}.json"
    payload = {
        "schema": "dermlip_resume_attempt.v1",
        "gate_sha256": gate_sha256,
        "run_id": run_id,
        "identity_fingerprint": identity_fingerprint,
        "method": method,
        "stage": stage,
        "checkpoint_sha256": checkpoint_sha256,
        "output_dir": str(Path(out_dir).resolve()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    descriptor = os.open(attempt_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return attempt_path


# --- v3: source-cohort-disjoint pool (additive; v2 above is unchanged) ---
# Attestation: /home/jovyan/work/dermlip_slic_v8/runtime_poolv2_20261022_001/preflight/pool-use-attestation.v3.20261101.006.json
# Derivation: mechanical re-derivation of v2, policy unchanged;
# patient_independence_accredited stays False.
V3_POOL_USE_ATTESTATION_SHA256 = "9bb5503ef40975aad99d006fe7d7ea0977d8613c7f0a078fac558b862a6ec8ad"
V3_POOL_USE_BINDINGS = {
    "holdout_novelty": False,
    "model": {
        "license": "CC-BY-NC-ND-4.0"
    },
    "patient_independence_accredited": False,
    "ph2": {
        "eligible_count": 195,
        "excluded_count": 5,
        "train_count": 156,
        "validation_count": 39
    },
    "pool_root": "/home/jovyan/work/dermlip_slic_v8/preflight/source-disjoint-pools-v3-20261101-003/",
    "preflight_receipt": {
        "path": "/home/jovyan/work/dermlip_slic_v8/preflight/source-disjoint-pools-v3-20261101-003/receipt.json",
        "sha256": "933b5af709f56e3fbd72a1593b3f2f50f5619e12fcd32edc63136d7c2e63ded4"
    },
    "source_artifacts": {
        "dermlip_model_manifest_sha256": "f1bcc6e2f5c5eb102f2565bbb920c0298e0356cc1db575e5dc5f7203fa6ec45f",
        "evaluation_manifest_receipt_sha256": "25f42c626ba6c4db8583b0220dcbd2f0450665fbbe3c62133741859988dc35a8",
        "evaluation_manifest_sha256": "53009a68073d6ae1048063a9328a6e7bc6b9ef34cdf0096d64ebaa866fe0f179",
        "ph2_exclusions_sha256": "b3253b3c1748f4057b497b8e2ceea6bfb4aa91344bba932b4c63d22253e9c09e",
        "ph2_manifest_sha256": "d2a159bb1f846e0ee2195406268d457d3c1b5a26a650c201b31e9f815371b9bf",
        "refit_source_manifest_sha256": "21903961c680cf8cc3bdffeb8be394e2e32aec2ca93b295eea47eeb859d0bea0",
        "selection_source_manifest_sha256": "331457804f6bde1b65c4f184a1a529d9a6e4802b0349f6aebc9f2ab6d1945dc8",
        "terms_acceptance_sha256": "5f6ef7d4cf7e810bddb053058bcd5dfaa9c56181e97ffcd62eb9356c47c598a9"
    },
    "stages": {
        "refit": {
            "cohort_sha256": "baff6ca6a6c6ab784f426f8dcaba2bf638f6387644161af659fad6c6e02c5c53",
            "dataset_counts": {
                "ISIC2019": 17804,
                "ISIC2020": 24924
            },
            "path": "/home/jovyan/work/dermlip_slic_v8/preflight/source-disjoint-pools-v3-20261101-003/refit.manifest.json",
            "record_count": 42728,
            "sha256": "3b9992aba46dd78df3f1b4094a3a03a812985098f02fd98065a8521df5e89e80"
        },
        "selection": {
            "cohort_sha256": "326f7316fcf2095a39122e3a198c6b0189a2bdafc43d7f137f6f5872f2aa2ae8",
            "dataset_counts": {
                "ISIC2019": 14196,
                "ISIC2020": 19919
            },
            "path": "/home/jovyan/work/dermlip_slic_v8/preflight/source-disjoint-pools-v3-20261101-003/selection.manifest.json",
            "record_count": 34115,
            "sha256": "0bb8be1d7d1eaa9848e7b554b8e20b6d9ee69687fecf3d2d375acbb027140a5d"
        }
    }
}
