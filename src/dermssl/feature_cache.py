"""Persistent cache for extracted features.

Why this module exists
----------------------
Running 10 000 images through a ViT-B is the expensive part of the evaluation -
minutes of GPU time. Fitting a probe on the resulting vectors is milliseconds.
So we extract features **once**, write them to disk, and then run as many
probes, label regimes and seeds as we like for free. That is what turns "5 seeds
x 6 regimes x 3 probes" from an impossible bill into a rounding error, and it is
what lets a Kaggle session survive a restart without redoing work.

The safety property that matters: a cache must never return features that belong
to a different configuration. The key therefore includes a fingerprint of the
*ordered* list of image ids, because features are stored aligned to that order.
Change the split, and the key changes, and the stale cache is ignored instead of
silently corrupting every downstream number.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

import numpy as np

SCHEMA_VERSION = 1


def fingerprint_ids(ids: Sequence[str]) -> str:
    """Short, order-sensitive digest of an id list."""
    digest = hashlib.sha256()
    for i in ids:
        digest.update(str(i).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def cache_key(
    backbone: str,
    split: str,
    resolution: int,
    ids: Sequence[str],
    transform_tag: str = "eval",
) -> str:
    """Filesystem-safe key identifying one exact extraction configuration."""
    safe_backbone = backbone.replace("/", "-")
    return (
        f"v{SCHEMA_VERSION}__{safe_backbone}__{split}__r{resolution}"
        f"__{transform_tag}__{fingerprint_ids(ids)}"
    )


@dataclass
class CachedFeatures:
    key: str
    features: np.ndarray
    labels: np.ndarray
    meta: dict


def _digest(array: np.ndarray) -> str:
    """sha256 of the array's bytes AND its shape/dtype.

    The raw buffer alone is not enough: the same bytes reshaped are a
    different array, and a labels file re-saved with another dtype would hash
    identically while loading as something else.
    """
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a unique temporary and rename into place.

    ``Path.replace`` is atomic within a filesystem, so a reader sees either
    the old file or the new one, never a partial. The temporary name carries
    a uuid because two processes writing the same key would otherwise share
    one scratch file and interleave into it.

    ``np.save`` appends ``.npy`` when the path lacks it, so the temporary
    keeps that suffix and the real name is restored by the rename.
    """
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp{path.suffix}")
    try:
        write(temporary)
        written = temporary
        if not written.exists():
            # np.save("x.tmp.npy") is fine, but a caller that passed a path
            # without the suffix gets it appended; find what actually landed.
            candidate = written.with_name(written.name + ".npy")
            if candidate.exists():
                written = candidate
        written.replace(path)
    finally:
        for leftover in path.parent.glob(f".{path.name}.*.tmp*"):
            leftover.unlink(missing_ok=True)


class FeatureCache:
    """Directory-backed store of ``(features, labels, meta)`` triples."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def paths_for(self, key: str) -> tuple[Path, Path, Path]:
        base = self.root / key
        return (
            base.with_suffix(".features.npy"),
            base.with_suffix(".labels.npy"),
            base.with_suffix(".meta.json"),
        )

    def has(self, key: str) -> bool:
        return all(p.is_file() for p in self.paths_for(key))

    def save(
        self,
        key: str,
        features: np.ndarray,
        labels: np.ndarray,
        meta: dict | None = None,
    ) -> CachedFeatures:
        """Write one entry. Rejects mismatched lengths instead of storing garbage."""
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels)
        if len(features) != len(labels):
            raise ValueError(
                f"features/labels length mismatch: {len(features)} vs {len(labels)}"
            )
        f_path, l_path, m_path = self.paths_for(key)
        full_meta = {
            "key": key,
            "schema": SCHEMA_VERSION,
            "n_samples": int(len(features)),
            "dim": int(features.shape[1]) if features.ndim == 2 else None,
            # What was written, so a later read can tell whether the three
            # files belong to the SAME write. See the docstring below.
            "features_sha256": _digest(features),
            "labels_sha256": _digest(labels),
            **(meta or {}),
        }
        # Each file lands by rename, and the meta lands LAST because it is what
        # makes the entry visible to `has`. A reader therefore never sees a
        # half-written array, and never sees an entry whose meta arrived before
        # its data.
        _atomic_write(f_path, lambda path: np.save(path, features))
        _atomic_write(l_path, lambda path: np.save(path, labels))
        _atomic_write(
            m_path,
            lambda path: path.write_text(json.dumps(full_meta, indent=2), encoding="utf-8"),
        )
        return CachedFeatures(key, features, labels, full_meta)

    def load(self, key: str) -> CachedFeatures | None:
        """Return the entry, or None when absent or inconsistent.

        Returning None (rather than raising) is deliberate: the caller's natural
        reaction is "then extract it", which is always correct.
        """
        f_path, l_path, m_path = self.paths_for(key)
        if not all(p.is_file() for p in (f_path, l_path, m_path)):
            return None
        try:
            features = np.load(f_path)
            labels = np.load(l_path)
            meta = json.loads(m_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if len(features) != len(labels):
            return None
        if meta.get("schema") != SCHEMA_VERSION:
            return None
        if meta.get("n_samples") not in (None, len(features)):
            return None

        # Do the three files come from the SAME write? A crash between the two
        # np.save calls leaves the new features next to the previous entry's
        # labels, and every other check here passes: the files exist, both
        # arrays load, the lengths agree, the schema matches. The cache would
        # then serve one run's features with another run's answer key.
        #
        # Entries written before these digests existed (the v6 bank cache on
        # the pod) have no digest and are accepted: refusing them would
        # re-extract the whole published bank and make it look corrupt.
        for name, array in (("features_sha256", features), ("labels_sha256", labels)):
            recorded = meta.get(name)
            if recorded is not None and recorded != _digest(array):
                return None

        return CachedFeatures(key, features, labels, meta)

    def get_or_extract(
        self,
        key: str,
        extract_fn,
        meta: dict | None = None,
    ) -> CachedFeatures:
        """Load ``key`` if cached, otherwise call ``extract_fn() -> (X, y)``.

        This single call is what makes the notebook resume-safe: a session that
        dies after extracting the training features re-uses them on restart.
        """
        cached = self.load(key)
        if cached is not None:
            return cached
        features, labels = extract_fn()
        return self.save(key, features, labels, meta=meta)

    def manifest(self) -> list[dict]:
        """Every entry currently on disk. Printed as an audit trail."""
        entries = []
        for m_path in sorted(self.root.glob("*.meta.json")):
            try:
                entries.append(json.loads(m_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return entries
