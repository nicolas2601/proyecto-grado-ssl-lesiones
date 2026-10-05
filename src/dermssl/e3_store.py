"""Files, locks and pins that keep E3 selection and the validation read apart (docs/33 section 4).

* :class:`CellStore`: one JSON per cell, written atomically and skipped when it already
  exists with the same ``key_sha256`` (the sha256 of the cell's identity: ids, protocol,
  mask sha, code revision). A cell with another key is moved aside as ``.stale-<sha8>``
  and recomputed, never mixed in. A failing cell is retried :data:`RETRIES` times, then
  listed in ``failures`` while the rest goes on.
* :func:`require_selection`: ``final`` needs the selection JSON AND its sha256 pinned on
  the command line; without either it aborts before touching any validation data.
* :class:`ValLock`: validation is read once. The lock file is created with ``O_EXCL`` and
  is never removed by code; a second read refuses.
* :class:`FeatureReader`: the only way ``select`` opens feature arrays. It is built with
  the forbidden splits and raises on them, so the property "select never opens val" is a
  checked behaviour, not a promise.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

RETRIES = 2
LOCK_NAME = "val_read.lock"
SPLITS = ("fit", "holdout", "train", "val")


class E3Error(RuntimeError):
    """Base of the E3 refusals."""


class SelectionMissingError(E3Error):
    """``final`` was asked to run without a selection file."""


class SelectionPinError(E3Error):
    """The selection file is not the pinned one, or is not a selection."""


class ValAlreadyReadError(E3Error):
    """Validation was already read (the lock exists)."""


class ValAccessError(E3Error):
    """A stage that must not see this split tried to open its arrays."""


class MissingFeaturesError(E3Error):
    """The feature array a stage needs was not extracted."""


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=_plain).encode("utf-8")).hexdigest()


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} no es serializable a JSON")


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=_plain, allow_nan=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


class CellStore:
    """One resumable JSON per cell under ``root``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.failures: list[tuple[str, str]] = []

    def path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def read(self, name: str) -> dict:
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def _is_current(self, name: str, key_sha: str) -> bool:
        path = self.path(name)
        if not path.is_file():
            return False
        try:
            return self.read(name).get("key_sha256") == key_sha
        except ValueError:
            return False

    def _set_aside(self, name: str) -> None:
        path = self.path(name)
        if path.is_file():
            path.replace(path.with_name(f"{path.name}.stale-{file_sha256(path)[:8]}"))

    def run(self, name: str, key: dict, fn) -> str:
        """``"skipped"``, ``"done"`` or ``"failed"``; ``KeyboardInterrupt`` is never swallowed."""
        key_sha = canonical_sha256(key)
        if self._is_current(name, key_sha):
            return "skipped"
        error = ""
        for _ in range(1 + RETRIES):
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - a failed cell is listed, the run goes on
                error = f"{type(exc).__name__}: {exc}"
                continue
            self._set_aside(name)
            write_json_atomic(self.path(name), {"cell": name, "key": key, "key_sha256": key_sha,
                                                "result": result})
            return "done"
        self.failures.append((name, error))
        return "failed"


def require_selection(path, pinned_sha256: str) -> tuple[dict, str]:
    """The selection document and its sha256, or an :class:`E3Error`; never reads validation."""
    path = Path(path)
    if not path.is_file():
        raise SelectionMissingError(f"no existe la seleccion en {path}: corre `select` y fija su sha256 antes de `final`")
    sha = file_sha256(path)
    if not pinned_sha256 or sha != pinned_sha256:
        raise SelectionPinError(f"el sha256 de {path.name} es {sha[:12]}, no el fijado ({str(pinned_sha256)[:12]!r})")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "decision" not in doc:
        raise SelectionPinError(f"{path.name} no es una seleccion de E3 (sin 'decision')")
    return doc, sha


class ValLock:
    """The single read of validation, as a file created with ``O_EXCL``."""

    def __init__(self, root: Path):
        self.path = Path(root) / LOCK_NAME

    def taken(self) -> bool:
        return self.path.exists()

    def acquire(self, selection_sha256: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise ValAlreadyReadError(
                f"val ya se leyo ({self.path}): una segunda lectura invalida la unica evaluacion") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"selection_sha256": selection_sha256, "pid": os.getpid()}, handle)


class FeatureReader:
    """Opens ``<split>__<condition>.npy`` arrays, refusing the splits it was built to exclude."""

    def __init__(self, root: Path, forbidden=()):
        self.root, self.forbidden = Path(root), frozenset(forbidden)

    def path(self, split: str, condition: str) -> Path:
        return self.root / f"{split}__{condition}.npy"

    def load(self, split: str, condition: str) -> np.ndarray:
        if split in self.forbidden:
            raise ValAccessError(f"esta etapa no puede abrir el split {split!r}")
        path = self.path(split, condition)
        if not path.is_file():
            raise MissingFeaturesError(f"falta {path.name}: corre `extract` para {split}/{condition}")
        return np.load(path, mmap_mode="r")


def save_array(root: Path, split: str, condition: str, array: np.ndarray) -> Path:
    path = FeatureReader(root).path(split, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)
    return path
