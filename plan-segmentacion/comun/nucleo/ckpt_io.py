"""Atomic checkpoint I/O (resume after a kill) and state-dict key helpers.

``torch`` is imported lazily so the pure key helpers stay testable without it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

WRAPPER_KEYS = ("state_dict", "model", "model_state_dict")


def save_checkpoint(path: Path, state: Mapping) -> None:
    """Write to a temporary file and rename, so a kill never leaves a torn file."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict:
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=True)


def unwrap_state_dict(loaded: Mapping) -> dict:
    """Return the tensor dict inside common checkpoint wrappers."""
    for key in WRAPPER_KEYS:
        if key in loaded and isinstance(loaded[key], Mapping):
            return dict(loaded[key])
    return dict(loaded)


def strip_state_dict_prefixes(state: Mapping, prefixes: Sequence[str]) -> dict:
    """Remove leading prefixes (``module.``, ``encoder_q.``...) from every key."""
    cleaned = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        cleaned[key] = value
    return cleaned
