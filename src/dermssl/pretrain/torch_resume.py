"""What a resume needs beyond weights and optimizer to replay the same batches.

A single-GPU run draws its batch order from a sampler seeded by
``(seed, epoch)`` instead of the global RNG, so the order of any epoch can be
rebuilt, and a resume skips the batches already consumed. The global random
state (augmentations draw from it) travels in the checkpoint. Checkpoints
land by rename, so a kill during the write leaves the previous one intact.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler


class ResumableRandomSampler(Sampler[int]):
    """Shuffled indices per epoch from ``(seed, epoch)``; can start mid-epoch once."""

    def __init__(self, n: int, seed: int) -> None:
        self.n, self.seed, self.epoch, self._start = n, seed, 0, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def start_at(self, offset: int) -> None:
        self._start = offset

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        order = torch.randperm(self.n, generator=generator).tolist()
        start, self._start = self._start, 0
        return iter(order[start:])

    def __len__(self) -> int:
        return self.n


def capture_rng() -> dict:
    state = {
        "torch": torch.get_rng_state(),
        # Plain types: checkpoints are read with weights_only=True.
        "numpy": _numpy_state_plain(np.random.get_state()),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _numpy_state_plain(state) -> tuple:
    kind, keys, pos, has_gauss, cached = state
    return kind, [int(k) for k in keys], int(pos), int(has_gauss), float(cached)


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    kind, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_save(obj: dict, path: Path) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
