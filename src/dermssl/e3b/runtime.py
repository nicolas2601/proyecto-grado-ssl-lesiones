"""What E3b needs from the outside world (real data on the pod, synthetic items in the tests)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SPLITS = ("train", "val")


@dataclass
class Runtime:
    ids: dict[str, list[str]]
    labels_of: Callable[[str], np.ndarray]
    groups_of: Callable[[str], np.ndarray]
    classes: tuple[str, ...]
    mel: int
    load_item: Callable[[str], tuple[np.ndarray, np.ndarray]]
    hair_fn: Callable[[np.ndarray], np.ndarray] | None
    build_backbone: Callable[[str], object]
    fingerprint: dict = field(default_factory=dict)

    @property
    def n_classes(self) -> int:
        return len(self.classes)


@dataclass(frozen=True)
class Settings:
    out_dir: Path
    models: tuple[str, ...]
    batch_size: int
    workers: int
    n_resamples: int
    e3_dir: Path | None
    limit: int | None
    selection_sha256: str
    revision: str

    @property
    def smoke(self) -> bool:
        return self.limit is not None
