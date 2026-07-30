"""The cache must be a speedup, never a source of wrong numbers."""

import json

import numpy as np
import pytest

from dermssl.feature_cache import FeatureCache, cache_key, fingerprint_ids


def test_fingerprint_is_order_sensitive():
    """Features are stored aligned to the id order, so order is part of identity."""
    assert fingerprint_ids(["a", "b"]) != fingerprint_ids(["b", "a"])
    assert fingerprint_ids(["a", "b"]) == fingerprint_ids(["a", "b"])


def test_fingerprint_separates_ids_unambiguously():
    """Without a separator, ['ab','c'] and ['a','bc'] would collide."""
    assert fingerprint_ids(["ab", "c"]) != fingerprint_ids(["a", "bc"])


def test_cache_key_changes_with_every_relevant_input():
    base = dict(backbone="dinov2_vitb14", split="train", resolution=224, ids=["a", "b"])
    key = cache_key(**base)
    assert key != cache_key(**{**base, "backbone": "resnet50"})
    assert key != cache_key(**{**base, "split": "val"})
    assert key != cache_key(**{**base, "resolution": 252})
    assert key != cache_key(**{**base, "ids": ["a", "c"]})
    assert key != cache_key(**base, transform_tag="tta")


def test_cache_key_is_filesystem_safe():
    key = cache_key("facebook/dinov2-base", "train", 224, ["a"])
    assert "/" not in key


def test_save_then_load_roundtrip(tmp_path):
    cache = FeatureCache(tmp_path)
    x = np.random.RandomState(0).randn(6, 4).astype(np.float32)
    y = np.array([0, 1, 2, 0, 1, 2])
    cache.save("k1", x, y, meta={"backbone": "test"})

    loaded = cache.load("k1")
    assert loaded is not None
    assert np.allclose(loaded.features, x)
    assert np.array_equal(loaded.labels, y)
    assert loaded.meta["backbone"] == "test"
    assert loaded.meta["n_samples"] == 6
    assert loaded.meta["dim"] == 4


def test_missing_key_loads_as_none(tmp_path):
    assert FeatureCache(tmp_path).load("never-written") is None
    assert FeatureCache(tmp_path).has("never-written") is False


def test_length_mismatch_is_rejected_at_save(tmp_path):
    cache = FeatureCache(tmp_path)
    with pytest.raises(ValueError, match="length mismatch"):
        cache.save("bad", np.zeros((3, 2)), np.array([0, 1]))


def test_corrupt_metadata_loads_as_none_instead_of_raising(tmp_path):
    cache = FeatureCache(tmp_path)
    cache.save("k", np.zeros((2, 2), dtype=np.float32), np.array([0, 1]))
    _, _, meta_path = cache.paths_for("k")
    meta_path.write_text("{not json", encoding="utf-8")
    assert cache.load("k") is None


def test_wrong_schema_version_is_ignored(tmp_path):
    cache = FeatureCache(tmp_path)
    cache.save("k", np.zeros((2, 2), dtype=np.float32), np.array([0, 1]))
    _, _, meta_path = cache.paths_for("k")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["schema"] = 999
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert cache.load("k") is None


def test_get_or_extract_computes_once_then_reuses(tmp_path):
    cache = FeatureCache(tmp_path)
    calls = {"n": 0}

    def extract():
        calls["n"] += 1
        return np.ones((4, 3), dtype=np.float32), np.array([0, 1, 0, 1])

    first = cache.get_or_extract("key", extract)
    second = cache.get_or_extract("key", extract)

    assert calls["n"] == 1, "second call must hit the cache, not the GPU"
    assert np.allclose(first.features, second.features)
    assert np.array_equal(first.labels, second.labels)


def test_get_or_extract_recomputes_for_a_different_key(tmp_path):
    cache = FeatureCache(tmp_path)
    calls = {"n": 0}

    def extract():
        calls["n"] += 1
        return np.zeros((2, 2), dtype=np.float32), np.array([0, 1])

    cache.get_or_extract("a", extract)
    cache.get_or_extract("b", extract)
    assert calls["n"] == 2


def test_stale_split_cannot_be_reused(tmp_path):
    """Changing the split changes the key, so old features are never returned."""
    cache = FeatureCache(tmp_path)
    ids_v1 = ["img1", "img2", "img3"]
    ids_v2 = ["img1", "img2", "img4"]
    key_v1 = cache_key("dinov2_vits14", "train", 224, ids_v1)
    cache.save(key_v1, np.zeros((3, 2), dtype=np.float32), np.array([0, 1, 2]))

    key_v2 = cache_key("dinov2_vits14", "train", 224, ids_v2)
    assert cache.load(key_v2) is None


def test_manifest_lists_saved_entries(tmp_path):
    cache = FeatureCache(tmp_path)
    cache.save("k1", np.zeros((2, 2), dtype=np.float32), np.array([0, 1]))
    cache.save("k2", np.zeros((3, 2), dtype=np.float32), np.array([0, 1, 0]))
    keys = {entry["key"] for entry in cache.manifest()}
    assert keys == {"k1", "k2"}
