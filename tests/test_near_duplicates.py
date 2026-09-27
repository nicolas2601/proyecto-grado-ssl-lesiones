"""Near-duplicate search between an evaluation set and a pretraining pool."""

import numpy as np

from dermssl import near_duplicates as nd


def test_pairs_within_threshold_only():
    q = np.array([0b0000, 0xFF], dtype=np.uint64)
    p = np.array([0b0111, 0b1111111, 0xFF00], dtype=np.uint64)
    pairs = nd.near_pairs(q, p, max_distance=3)
    # q0 vs p0: 3 bits; q1 vs p1: 1 bit; q1 vs p2: 16 bits.
    assert pairs == [(0, 0, 3), (1, 1, 1)]


def test_identical_hash_is_distance_zero():
    h = np.array([0xDEADBEEFCAFEF00D], dtype=np.uint64)
    assert nd.near_pairs(h, h.copy(), max_distance=0) == [(0, 0, 0)]


def test_hash_bits_roundtrip():
    bits = np.zeros((8, 8), dtype=bool)
    bits[0, 0] = bits[7, 7] = True
    value = nd.bits_to_uint64(bits)
    assert int(value) == (1 << 63) | 1


def test_phash_resized_copy_is_near(tmp_path):
    import pytest

    pytest.importorskip("imagehash")
    from PIL import Image

    rng = np.random.default_rng(0)
    base = Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).resize((256, 256))
    base.save(tmp_path / "a.png")
    base.resize((200, 200)).save(tmp_path / "b.png")
    a, b = nd.phash(tmp_path / "a.png"), nd.phash(tmp_path / "b.png")
    assert nd.near_pairs(np.array([a]), np.array([b]), max_distance=6)
