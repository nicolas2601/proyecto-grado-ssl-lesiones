"""Dataset resolution with three interchangeable backends.

Why this module exists
----------------------
The v4 notebook hardcoded ``/kaggle/input``. That path only exists inside a
Kaggle session, so the exact same notebook could not run on a university
cluster. This module removes that coupling: the notebook asks for datasets by
name and gets back local paths, no matter where it runs.

Backends
--------
``kaggle-input``
    Datasets already mounted by Kaggle under ``/kaggle/input``. Offline, instant.
``kagglehub``
    Datasets downloaded on demand with the ``kagglehub`` package. Works on any
    machine with credentials (``KAGGLE_USERNAME`` + ``KAGGLE_KEY``, or
    ``~/.kaggle/kaggle.json``). Inside Kaggle it transparently attaches the
    input instead of downloading.
``local``
    Copies already staged on disk (a cluster scratch directory, an NFS mount).

``auto`` resolves in that order of preference and reports which one it picked.

All discovery helpers take an explicit list of search roots so they can be unit
tested against temporary directories without any cloud access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

KAGGLE_INPUT = Path("/kaggle/input")

# Kaggle dataset handles used by the project. Datasets themselves are NOT
# changing; only the way we reach them does.
DATASET_HANDLES = {
    "ham10000": "kmader/skin-cancer-mnist-ham10000",
    "isic2019": "nischaydnk/isic-2019-jpg-224x224-resized",
    "isic2020": "nischaydnk/isic-2020-jpg-256x256-resized",
}

BACKENDS = ("kaggle-input", "kagglehub", "local")


class DataResolutionError(RuntimeError):
    """Raised when a required dataset cannot be located in any backend."""


@dataclass
class ResolvedData:
    """Everything downstream cells need, with provenance attached."""

    backend: str
    roots: list[Path] = field(default_factory=list)
    ham_metadata_csv: Path | None = None
    ham_images: dict[str, Path] = field(default_factory=dict)
    pool_images: dict[str, Path] = field(default_factory=dict)
    pool_indexed: bool = True
    candidate_label_csvs: list[Path] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "backend": self.backend,
            "roots": [str(r) for r in self.roots],
            "ham_metadata_csv": str(self.ham_metadata_csv) if self.ham_metadata_csv else None,
            "ham_images": len(self.ham_images),
            # Distinguish "indexed and found none" from "deliberately not indexed".
            "pool_images": len(self.pool_images) if self.pool_indexed else "no indexado",
            "candidate_label_csvs": [str(p) for p in self.candidate_label_csvs],
        }


def detect_backend(
    kaggle_input: Path = KAGGLE_INPUT,
    local_root: Path | None = None,
) -> str:
    """Pick a backend without touching the network.

    Preference: mounted Kaggle inputs > staged local copies > kagglehub download.
    """
    if kaggle_input.is_dir() and any(kaggle_input.iterdir()):
        return "kaggle-input"
    if local_root is not None and Path(local_root).is_dir():
        if any(Path(local_root).rglob("HAM10000_metadata.csv")):
            return "local"
    return "kagglehub"


def index_images(roots: Iterable[Path], pattern: str = "**/*.jpg") -> dict[str, Path]:
    """Map image stem -> path for every match under ``roots``.

    Later roots win on collisions, which is why callers pass the authoritative
    root last. Stems are unique per dataset in ISIC/HAM, so collisions mean the
    same image is mounted twice.
    """
    found: dict[str, Path] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in root.glob(pattern):
            if path.is_file():
                found[path.stem] = path
    return found


def find_files(roots: Iterable[Path], filename: str) -> list[Path]:
    """Locate every file called ``filename`` under any of ``roots``."""
    hits: list[Path] = []
    for root in roots:
        root = Path(root)
        if root.is_dir():
            hits.extend(sorted(root.rglob(filename)))
    return hits


def find_label_csvs(roots: Iterable[Path], keyword: str = "groundtruth") -> list[Path]:
    """Find candidate ground-truth CSVs (external validation labels).

    Matching is case-insensitive on the file name so ``ISIC_2019_Training_
    GroundTruth.csv`` and ``groundtruth.csv`` both match.
    """
    hits: list[Path] = []
    key = keyword.lower()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.csv")):
            if key in path.name.lower():
                hits.append(path)
    return hits


def _kagglehub_download(handle: str, output_dir: Path | None) -> Path:
    """Download one dataset through kagglehub. Imported lazily on purpose.

    Lazy import keeps this module usable (and testable) on machines that never
    installed kagglehub.
    """
    try:
        import kagglehub  # noqa: PLC0415 - lazy on purpose
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise DataResolutionError(
            "backend 'kagglehub' requires the kagglehub package: pip install kagglehub"
        ) from exc

    kwargs = {}
    if output_dir is not None:
        kwargs["output_dir"] = str(output_dir)
    return Path(kagglehub.dataset_download(handle, **kwargs))


def credentials_present() -> bool:
    """True when kagglehub has some way to authenticate.

    Checked before downloading so the failure message is about credentials
    instead of an opaque HTTP 401 several minutes in.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").is_file()


def dataset_roots(
    backend: str,
    datasets: Sequence[str],
    kaggle_input: Path = KAGGLE_INPUT,
    local_root: Path | None = None,
    cache_dir: Path | None = None,
) -> list[Path]:
    """Return the filesystem roots to search, per backend."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")

    if backend == "kaggle-input":
        if not Path(kaggle_input).is_dir():
            raise DataResolutionError(f"{kaggle_input} does not exist; are you inside Kaggle?")
        return [Path(kaggle_input)]

    if backend == "local":
        if local_root is None:
            raise DataResolutionError("backend 'local' needs local_root=<path>")
        root = Path(local_root)
        if not root.is_dir():
            raise DataResolutionError(f"local_root {root} does not exist")
        return [root]

    if not credentials_present():
        raise DataResolutionError(
            "backend 'kagglehub' found no credentials. Set KAGGLE_USERNAME + KAGGLE_KEY "
            "or place ~/.kaggle/kaggle.json (chmod 600). Never commit that file."
        )
    roots = []
    for name in datasets:
        handle = DATASET_HANDLES.get(name, name)
        target = None if cache_dir is None else Path(cache_dir) / name
        roots.append(_kagglehub_download(handle, target))
    return roots


def resolve_datasets(
    backend: str = "auto",
    datasets: Sequence[str] = ("ham10000", "isic2019", "isic2020"),
    kaggle_input: Path = KAGGLE_INPUT,
    local_root: Path | None = None,
    cache_dir: Path | None = None,
    index_pool: bool = True,
) -> ResolvedData:
    """Locate HAM10000 plus the unlabeled pool, whatever the environment.

    ``index_pool=False`` skips building the path index for the unlabelled pool.
    That index costs real time: on Kaggle, walking the 48 442 ISIC images took
    157 of the 192 seconds of a smoke run - and the frozen-backbone evaluation
    never touches the pool. Only the continual pre-training stage needs it, so it
    is opt-in per stage rather than always paid.

    Returns a :class:`ResolvedData` describing what was found. Raises
    :class:`DataResolutionError` with an actionable message when HAM10000 (the
    only mandatory dataset) is missing.
    """
    if backend == "auto":
        backend = detect_backend(kaggle_input=kaggle_input, local_root=local_root)

    roots = dataset_roots(
        backend,
        datasets,
        kaggle_input=kaggle_input,
        local_root=local_root,
        cache_dir=cache_dir,
    )

    metadata_hits = find_files(roots, "HAM10000_metadata.csv")
    if not metadata_hits:
        raise DataResolutionError(
            "HAM10000_metadata.csv not found. On Kaggle: Add Input -> "
            "'Skin Cancer MNIST: HAM10000' (kmader). Elsewhere: use "
            "backend='kagglehub' or stage the dataset and pass local_root."
        )
    ham_metadata_csv = metadata_hits[0]
    ham_dir = ham_metadata_csv.parent

    ham_images = index_images([ham_dir])
    pool_images: dict[str, Path] = {}
    if index_pool:
        # Pool = every other jpg reachable from the roots, minus the HAM images.
        all_images = index_images(roots)
        pool_images = {k: v for k, v in all_images.items() if k not in ham_images}

    return ResolvedData(
        backend=backend,
        roots=[Path(r) for r in roots],
        ham_metadata_csv=ham_metadata_csv,
        ham_images=ham_images,
        pool_images=pool_images,
        pool_indexed=index_pool,
        candidate_label_csvs=find_label_csvs(roots),
    )
