"""Safe manifest and raster readers shared by qualitative S1/T5 figures."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
BINARY_MASK_VALUES = {0, 1, 255}


def read_manifest(manifest_path: str | Path) -> tuple[Path, list[dict], Path]:
    manifest = Path(manifest_path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"input manifest not found: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    cases = payload if isinstance(payload, list) else payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases or any(not isinstance(case, dict) for case in cases):
        raise ValueError("manifest must contain a non-empty list of case objects")
    return manifest.parent.resolve(), cases, manifest


def resolve_input_path(root: str | Path, relative_path: str) -> Path:
    base = Path(root).resolve()
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("manifest input paths must be non-empty relative paths")
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"manifest input path must be relative: {relative_path}")
    resolved = (base / relative).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(f"manifest input path must remain inside its root: {relative_path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest input file not found: {relative_path}")
    return resolved


def resolve_case_path(root: Path, case: dict, filename: str) -> Path:
    routes = case.get("rutas")
    route = routes.get(filename) if isinstance(routes, dict) else case.get(filename)
    return resolve_input_path(root, route)


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB image has unexpected shape: {path}")
    return rgb


def read_binary_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
    values = set(int(value) for value in np.unique(grayscale))
    if not values.issubset(BINARY_MASK_VALUES):
        raise ValueError(f"mask must be binary black/white (0/1/255): {path}")
    return (grayscale > 0).astype(np.uint8)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_sha256(path: str | Path, expected: str, label: str) -> str:
    if not isinstance(expected, str) or not SHA256_PATTERN.fullmatch(expected):
        raise ValueError(f"{label} SHA-256 must contain 64 hexadecimal characters")
    actual = sha256_file(path)
    if actual.casefold() != expected.casefold():
        raise ValueError(f"{label} SHA-256 mismatch: {path}")
    return actual
