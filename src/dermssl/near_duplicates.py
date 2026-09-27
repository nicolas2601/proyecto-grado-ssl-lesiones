"""Perceptual-hash search for near-duplicates across two image sets.

``lesion_id`` only exists in HAM10000, so a lesion photographed again in ISIC
2019/2020 under another id and other bytes escapes the id and content checks
of the pretraining exclusion. A 64-bit pHash with a small Hamming radius is
the cheapest CPU-only net for that gap.
"""

from __future__ import annotations

import numpy as np

HASH_SIZE = 8


def bits_to_uint64(bits: np.ndarray) -> np.uint64:
    """Row-major 8x8 boolean grid to one integer, first bit most significant."""
    flat = np.asarray(bits, dtype=bool).reshape(-1)
    if flat.size != HASH_SIZE * HASH_SIZE:
        raise ValueError(f"se esperaban {HASH_SIZE * HASH_SIZE} bits, llegaron {flat.size}")
    value = 0
    for bit in flat:
        value = (value << 1) | int(bit)
    return np.uint64(value)


def phash(path) -> np.uint64:
    """64-bit pHash, the imagehash recipe (32x32 grey, DCT, 8x8 low band, median)."""
    import imagehash  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as image:
        return bits_to_uint64(imagehash.phash(image, hash_size=HASH_SIZE).hash)


def _popcount(values: np.ndarray) -> np.ndarray:
    return np.unpackbits(values.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def near_pairs(
    queries: np.ndarray, pool: np.ndarray, *, max_distance: int
) -> list[tuple[int, int, int]]:
    """``(query_index, pool_index, hamming)`` for every pair within the radius."""
    queries = np.asarray(queries, dtype=np.uint64)
    pool = np.asarray(pool, dtype=np.uint64)
    pairs = []
    for qi, q in enumerate(queries):
        distances = _popcount(np.bitwise_xor(pool, q))
        for pi in np.flatnonzero(distances <= max_distance):
            pairs.append((qi, int(pi), int(distances[pi])))
    return pairs
