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


def test_phash_matches_imagehash(tmp_path):
    import pytest

    imagehash = pytest.importorskip("imagehash")
    from PIL import Image

    rng = np.random.default_rng(1)
    Image.fromarray((rng.random((90, 120, 3)) * 255).astype("uint8")).save(tmp_path / "c.png")
    with Image.open(tmp_path / "c.png") as image:
        expected = nd.bits_to_uint64(imagehash.phash(image).hash)
    assert nd.phash(tmp_path / "c.png") == expected


def test_cosine_topk_finds_the_rescaled_copy_first():
    rng = np.random.default_rng(2)
    pool = rng.normal(size=(50, 16)).astype(np.float32)
    queries = np.stack([pool[7] * 3.0, pool[21] + 1e-3])
    scores, index = nd.cosine_topk(queries, pool, k=3, chunk=1)
    assert index[:, 0].tolist() == [7, 21]
    assert scores[0, 0] == np.float32(1.0) or abs(scores[0, 0] - 1.0) < 1e-5
    assert (np.diff(scores, axis=1) <= 1e-7).all()


def test_top1_summary_counts_the_threshold_inclusively():
    summary = nd.summarize_top1(np.array([0.5, 0.95, 0.99, 0.2]), threshold=0.95)
    assert summary["n_at_or_above"] == 2
    assert set(summary["percentiles"]) == {"50", "90", "99", "99.9"}


def test_pool_max_cosine_flags_every_copy_not_just_the_top1():
    rng = np.random.default_rng(3)
    queries = rng.normal(size=(2, 8)).astype(np.float32)
    pool = rng.normal(size=(5, 8)).astype(np.float32)
    pool[1] = queries[0] * 2  # two copies of the same query
    pool[3] = queries[0] + 1e-4
    best = nd.pool_max_cosine(queries, pool, chunk=2)
    assert best.shape == (5,)
    assert (np.flatnonzero(best > 0.999) == [1, 3]).all()
