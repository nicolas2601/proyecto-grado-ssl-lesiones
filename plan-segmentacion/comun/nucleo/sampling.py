"""Deterministic, rank-aware batching over memmap row indices (no DataLoader).

The cluster pod has a 64 MB ``/dev/shm``, so DataLoader workers are avoided:
batches are plain index arrays read straight from the memmap in the main process.
"""

from __future__ import annotations

import numpy as np


def steps_per_epoch(count: int, batch_size: int, world_size: int) -> int:
    """Number of optimizer steps per epoch when incomplete global batches are dropped."""
    return count // (batch_size * world_size)


def shard_batches(
    indices: np.ndarray,
    batch_size: int,
    rank: int,
    world_size: int,
    epoch: int,
    seed: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
) -> list[np.ndarray]:
    """Return this rank's batches of row indices for one epoch.

    Training (``drop_last=True``): every rank gets exactly the same number of
    full batches, which DDP requires. Evaluation (``drop_last=False``): ranks
    take interleaved slices and keep every index exactly once; the caller must
    reduce sums only after the loop, never per step.
    """
    indices = np.asarray(indices)
    if shuffle:
        order = np.random.default_rng([seed, epoch]).permutation(len(indices))
        indices = indices[order]

    if not drop_last:
        shard = indices[rank::world_size]
        return [
            np.sort(shard[start : start + batch_size])
            for start in range(0, len(shard), batch_size)
        ]

    steps = steps_per_epoch(len(indices), batch_size, world_size)
    if steps == 0:
        raise ValueError(
            f"{len(indices)} images cannot fill one global batch of {batch_size * world_size}"
        )
    grid = indices[: steps * batch_size * world_size].reshape(steps, world_size, batch_size)
    return [np.sort(grid[step, rank]) for step in range(steps)]
