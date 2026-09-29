"""T7 (docs/27): luminance terciles per cache row and the fixed-budget oversampled epoch.

The bins are ``tone_tercile`` of the published manifest: terciles of the skin
median L* over the pool, luminance and not skin tone. The dark tercile is
oversampled by an integer factor ``k`` without changing the budget: each
epoch is the multiset of the training rows plus ``k - 1`` extra copies of the
dark ones, permuted with ``default_rng([seed, epoch])`` and cut to the number
of samples A2 drew. With ``k = 1`` the permutation is the one
``sampling.shard_batches`` draws itself, so A2's batches are reproduced
exactly (tests pin it). Pure numpy and csv.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from dermssl import pad_ufes_guard as pg

REGISTRATION_COMMIT = "d24a0f252e6c06a6bce5c85cf01de52139a64ae5"
REGISTRATION_DOC = "docs/27-plan-v7-mejoras-ssl.md"
BLOCK_START = "<!-- registro-t7-reentreno-luminancia:inicio -->"
BLOCK_END = "<!-- registro-t7-reentreno-luminancia:fin -->"
#: The code that implements T7's training changes; it must postdate the registration.
ANCHOR = "src/dermssl/luminance_sampling.py"
MANIFEST_SHA256 = "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"
PSEUDO_R0_SHA256 = "8c9e47978070baa7bfa9f0174a80eb4e4236d45737801abf690804ae1eb0fc87"
CACHE_MANIFEST_SHA256 = "a83f927ab798b6f7a0614c3923fda77d082f887d10fdbe73416e34e9574ba71a"
REGISTERED_K = 2
DARK = "dark"
DARKENED = ("medium", "light")
UNAVAILABLE = "unavailable"


def check_registration(repo: Path) -> str:
    return pg.check_registration(repo, commit=REGISTRATION_COMMIT, doc=REGISTRATION_DOC, script=ANCHOR,
                                 start=BLOCK_START, end=BLOCK_END)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terciles_by_row(rows: Iterable[Mapping], manifest: Path, expected_sha256: str) -> np.ndarray:
    """``tone_tercile`` for every cache row (by ``index``); ids absent from the manifest are unavailable."""
    digest = _sha256(Path(manifest))
    if digest != expected_sha256:
        raise ValueError(f"el manifiesto {manifest} tiene sha256 {digest[:12]}, no el registrado")
    with open(manifest, newline="") as handle:
        by_id = {r["image_id"]: r["tone_tercile"] for r in csv.DictReader(handle)}
    rows = list(rows)
    labels = np.full(max(int(r["index"]) for r in rows) + 1, UNAVAILABLE, dtype=object)
    for row in rows:
        labels[int(row["index"])] = by_id.get(row["image_id"], UNAVAILABLE)
    return labels.astype(str)


def darken_eligible(labels: np.ndarray) -> np.ndarray:
    """Rows the darkening augmentation may touch: the medium and light terciles only."""
    return np.isin(np.asarray(labels), DARKENED)


def oversampled_epoch(train: np.ndarray, is_dark: np.ndarray, k: int, n_samples: int, seed: int,
                      epoch: int) -> np.ndarray:
    """The epoch's sample order: train plus ``k - 1`` copies of the dark rows, permuted, cut to ``n_samples``."""
    train, is_dark = np.asarray(train), np.asarray(is_dark, dtype=bool)
    if k < 1:
        raise ValueError(f"k tiene que ser un entero >= 1, no {k}")
    if is_dark.shape != train.shape:
        raise ValueError(f"is_dark tiene longitud {is_dark.size}, train {train.size}")
    pool = np.concatenate([train, *([train[is_dark]] * (k - 1))])
    if n_samples > pool.size:
        raise ValueError(f"se piden {n_samples} muestras y el multiconjunto tiene {pool.size}")
    order = np.random.default_rng([seed, epoch]).permutation(pool.size)
    return pool[order][:n_samples]
