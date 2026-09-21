"""Argument presets stored as JSON and translated into command-line flags."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def load_preset(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def flags_from_mapping(mapping: Mapping[str, object]) -> list[str]:
    """Translate ``{"batch_size": 8, "flag": True}`` into CLI tokens.

    ``True`` becomes a bare flag, ``False``/``None`` are skipped and lists repeat
    the flag once per item.
    """
    tokens: list[str] = []
    for key, value in mapping.items():
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            tokens.append(flag)
        elif isinstance(value, (list, tuple)):
            for item in value:
                tokens.extend([flag, str(item)])
        else:
            tokens.extend([flag, str(value)])
    return tokens
