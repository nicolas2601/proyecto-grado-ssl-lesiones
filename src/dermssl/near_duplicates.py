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
    """64-bit pHash, the imagehash recipe without the dependency.

    Grey 32x32 (Lanczos), 2-D DCT-II, the 8x8 low-frequency block, one bit
    per coefficient above its median; identical to ``imagehash.phash``.
    """
    from PIL import Image  # noqa: PLC0415
    from scipy.fftpack import dct  # noqa: PLC0415

    side = HASH_SIZE * 4
    with Image.open(path) as image:
        grey = image.convert("L").resize((side, side), Image.Resampling.LANCZOS)
    pixels = np.asarray(grey, dtype=np.float64)
    low = dct(dct(pixels, axis=0), axis=1)[:HASH_SIZE, :HASH_SIZE]
    return bits_to_uint64(low > np.median(low))


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


def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cosine_topk(
    queries: np.ndarray, pool: np.ndarray, *, k: int, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """``(scores, index)`` of the ``k`` most similar pool rows per query, best first."""
    q, p = unit_rows(queries), unit_rows(pool)
    scores, index = [], []
    for start in range(0, len(q), chunk):
        sim = q[start:start + chunk] @ p.T
        top = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
        top_scores = np.take_along_axis(sim, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        index.append(np.take_along_axis(top, order, axis=1))
        scores.append(np.take_along_axis(top_scores, order, axis=1))
    return np.concatenate(scores), np.concatenate(index)


def summarize_top1(top1: np.ndarray, *, threshold: float) -> dict:
    top1 = np.asarray(top1, dtype=np.float64)
    return {
        "percentiles": {str(q): float(np.percentile(top1, q)) for q in (50, 90, 99, 99.9)},
        "threshold": threshold,
        "n_at_or_above": int((top1 >= threshold).sum()),
        "n_queries": int(len(top1)),
    }
