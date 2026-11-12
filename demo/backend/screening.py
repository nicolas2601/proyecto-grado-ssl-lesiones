"""Melanoma-priority screening flag next to the shipped prediction. It never changes ``pred``.

The policy file (schema dermssl.screening-policy.v1, pinned by SHA256 from the environment) holds one
log offset per head, chosen on development out-of-fold data only. The screening prediction is
``argmax(log p + log_offset * [class == target])``. A missing, invalid or mismatched policy makes
the block ``unavailable``; it never fails the request.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from pathlib import Path
from typing import Any

SCHEMA = "dermssl.screening-policy.v1"
POLICY_NAME = "screening_policy.json"
MAX_LOG_OFFSET = 3.0
SHA256_PATTERN = r"[a-f0-9]{64}"
TARGETS = {"MOCO_HAM7": "mel", "MOCO_PAD6": "MEL"}


class _PolicyError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(SHA256_PATTERN, value) is not None


def _valid_head(model_key: str, head: Any) -> bool:
    if not isinstance(head, dict) or head.get("target") != TARGETS.get(model_key):
        return False
    offset = head.get("log_offset")
    if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(offset):
        return False
    return 0 <= offset <= MAX_LOG_OFFSET and _is_sha(head.get("head_sha256"))


def _validate(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict) or policy.get("schema") != SCHEMA or not _is_sha(policy.get("model_sha256")):
        raise _PolicyError("policy_invalid")
    heads = policy.get("heads")
    if not isinstance(heads, dict) or not heads or not isinstance(policy.get("rule_id"), str):
        raise _PolicyError("policy_invalid")
    if any(not _valid_head(key, head) for key, head in heads.items()):
        raise _PolicyError("policy_invalid")
    return policy


def _screening_pred(classes: list[str], probs: dict[str, float], target: str, offset: float) -> str:
    def score(label: str) -> float:
        p = float(probs[label])
        base = math.log(p) if p > 0 else -math.inf
        return base + offset if label == target else base

    scores = [score(label) for label in classes]
    return classes[scores.index(max(scores))]  # first maximum, like numpy argmax


class ScreeningPolicy:
    def __init__(self, policy_dir: str | Path, expected_sha256: str | None):
        self._path = Path(policy_dir) / POLICY_NAME
        self._pin = expected_sha256
        self._lock = threading.Lock()
        self._policy: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        with self._lock:
            if self._policy is not None:
                return self._policy
            if self._pin is None:
                raise _PolicyError("policy_not_configured")
            try:
                content = self._path.read_bytes()
            except OSError as exc:
                raise _PolicyError("policy_not_configured") from exc
            if not _is_sha(self._pin) or hashlib.sha256(content).hexdigest() != self._pin:
                raise _PolicyError("policy_invalid")
            try:
                self._policy = _validate(json.loads(content.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise _PolicyError("policy_invalid") from exc
            return self._policy

    def status(self) -> dict[str, Any]:
        try:
            policy = self._load()
        except _PolicyError as exc:
            return {"ready": False, "reason": exc.reason, "rule_id": None, "sha256": None}
        return {"ready": True, "reason": "ready", "rule_id": policy["rule_id"], "sha256": self._pin}

    def annotate(self, prediction: dict[str, Any]) -> dict[str, Any]:
        try:
            policy = self._load()
        except _PolicyError as exc:
            return _unavailable(exc.reason)
        head = policy["heads"].get(prediction.get("model_key"))
        if head is None:
            return _unavailable("model_not_covered")
        identity = prediction.get("identity") or {}
        if identity.get("model_sha256") != policy["model_sha256"] or identity.get("head_sha256") != head["head_sha256"]:
            return _unavailable("classifier_mismatch")
        try:
            pred = _screening_pred(prediction["classes"], prediction["probs"], head["target"], float(head["log_offset"]))
        except (KeyError, TypeError, ValueError):
            return _unavailable("prediction_invalid")
        return {
            "status": "ok",
            "reason": None,
            "rule_id": policy["rule_id"],
            "target": head["target"],
            "log_offset": float(head["log_offset"]),
            "pred": pred,
            "alert": pred == head["target"],
            "differs_from_pred": pred != prediction.get("pred"),
            "development": head.get("development"),
            "note": (
                "Operating point chosen on development data to prioritise melanoma recall; it raises "
                "more false alerts on benign lesions. Not evaluated on the sealed test sets."
            ),
        }
