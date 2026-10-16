from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def describe_model(manifest_path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        detail = manifest["backbone_detail"]
        digest = detail["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            return None
        return {
            "model_id": manifest["arm"],
            "release": detail["release"],
            "backbone_sha256": digest,
            "sha256_verified": False,
            "feature_dim": manifest["feature_extraction"]["dim"],
            "preprocess": manifest["preprocess"],
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def verify_model(manifest_path: Path, weights_path: Path) -> dict[str, Any]:
    identity = describe_model(manifest_path)
    if identity is None:
        raise ValueError("invalid model identity manifest")
    with weights_path.open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    if digest != identity["backbone_sha256"]:
        raise ValueError("backbone checksum does not match classifier manifest")
    return {**identity, "sha256_verified": True}
