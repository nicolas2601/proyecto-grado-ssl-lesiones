"""Pure stage-ledger contracts for the complete Kaggle experiment flow."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SCHEMA_VERSION = "1.0"
VALID_STATUSES = frozenset({"PASS", "FAIL", "COMPLETE", "SKIPPED", "SMOKE"})
CANONICAL_STAGE_NAMES = frozenset(
    {"fst", "tone_transforms", "pretrain", "stage1", "stage2"}
)
_STATUS_ALIASES = {"completed": "COMPLETE", "skipped": "SKIPPED"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_STAGE_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "input_sha256",
        "output_sha256",
        "status",
        "valid_for_thesis",
    }
)


def _sha256_artifact(value: Path | str | None) -> str:
    if value is None:
        payload = b""
    else:
        path = Path(value)
        payload = path.read_bytes() if path.is_file() else str(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_status(status: str) -> str:
    return _STATUS_ALIASES.get(status, status)


def new_ledger(run_id: str, code_sha256: str) -> dict:
    """Create an empty ledger bound to an immutable code revision."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "code_sha256": code_sha256,
        "stages": [],
    }


def record_stage(
    ledger: dict,
    *,
    name: str,
    status: str,
    artifacts: dict[str, Path | str],
) -> dict:
    """Append one hashable stage result and return its serialized record."""
    if name not in CANONICAL_STAGE_NAMES:
        raise ValueError(f"nombre de etapa no canónico: {name}")
    status = _normalize_status(status)
    if status not in VALID_STATUSES:
        raise ValueError(f"estado de etapa no válido: {status}")

    stage = {
        "name": name,
        "schema_version": SCHEMA_VERSION,
        "run_id": ledger["run_id"],
        "input_sha256": _sha256_artifact(artifacts.get("input")),
        "output_sha256": _sha256_artifact(artifacts.get("output")),
        "status": status,
        "valid_for_thesis": status not in {"SMOKE", "SKIPPED"},
    }
    ledger.setdefault("stages", []).append(stage)
    return stage


def validate_ledger(ledger: dict) -> None:
    """Reject malformed ledgers and Stage 2 executions without a Stage 1 FAIL."""
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("el ledger tiene una versión de esquema no compatible")
    if not ledger.get("run_id"):
        raise ValueError("el ledger requiere run_id")
    if not _SHA256_RE.fullmatch(str(ledger.get("code_sha256", ""))):
        raise ValueError("el ledger requiere code_sha256 SHA-256")

    stages = ledger.get("stages")
    if not isinstance(stages, list):
        raise ValueError("el ledger requiere una lista de etapas")

    names = set()
    stage1_failed = False
    for stage in stages:
        if not isinstance(stage, dict) or _REQUIRED_STAGE_KEYS - stage.keys():
            raise ValueError("una etapa no cumple el esquema requerido")
        if stage["schema_version"] != SCHEMA_VERSION or stage["run_id"] != ledger["run_id"]:
            raise ValueError("una etapa no coincide con el ledger")
        if not _SHA256_RE.fullmatch(str(stage["input_sha256"])):
            raise ValueError("una etapa tiene input_sha256 inválido")
        if not _SHA256_RE.fullmatch(str(stage["output_sha256"])):
            raise ValueError("una etapa tiene output_sha256 inválido")
        normalized_status = _normalize_status(stage["status"])
        if normalized_status not in VALID_STATUSES:
            raise ValueError("una etapa tiene un estado no válido")
        if not isinstance(stage["valid_for_thesis"], bool):
            raise ValueError("una etapa debe declarar valid_for_thesis")
        if normalized_status in {"SMOKE", "SKIPPED"} and stage["valid_for_thesis"]:
            raise ValueError("una etapa smoke no es válida para tesis")
        name = stage.get("name")
        if name not in CANONICAL_STAGE_NAMES:
            raise ValueError("una etapa tiene un nombre no canónico")
        if name in names:
            raise ValueError("el ledger no admite etapas duplicadas")
        names.add(name)
        if name == "stage2" and not stage1_failed:
            raise ValueError("Stage 2 solo puede ejecutarse tras Stage 1 con estado FAIL")
        if name == "stage1" and normalized_status == "FAIL":
            stage1_failed = True
