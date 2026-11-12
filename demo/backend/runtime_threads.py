"""ONNX intra-op thread count. Production keeps 2; the cluster evidence run may raise it via env."""

from __future__ import annotations

import os

DEFAULT_INTRA_THREADS = 2
ENV_VAR = "DERM_ORT_INTRA_THREADS"
MAX_INTRA_THREADS = 8


def intra_op_threads() -> int:
    raw = os.getenv(ENV_VAR)
    if raw is None or raw == "":
        return DEFAULT_INTRA_THREADS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENV_VAR} must be an integer") from exc
    if not 1 <= value <= MAX_INTRA_THREADS:
        raise ValueError(f"{ENV_VAR} must be between 1 and {MAX_INTRA_THREADS}")
    return value
