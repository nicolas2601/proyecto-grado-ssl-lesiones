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


class TestAnInterruptedWriteCannotProduceAMixedEntry:
    """The failure a truncated file does NOT cover.

    Measured, not assumed: a truncated ``.npy`` raises ``ValueError`` on load,
    and ``load`` already catches that, so a write interrupted mid-features
    self-heals. The hole is elsewhere. Three separate writes mean a crash
    BETWEEN them can leave:

        features -> the new array (complete)
        labels   -> the previous entry's array (complete)
        meta     -> the previous entry's meta (complete)

    Every existing check passes: the files are all there, both arrays load,
    the lengths agree, the schema matches. The cache then serves one run's
    features paired with another run's labels, and every downstream number is
    computed against the wrong answer key.

    So the entry carries the digests of what was written, and load refuses a
    set that does not match them.
    """

    def _cache(self, tmp_path):
        from dermssl.feature_cache import FeatureCache

        return FeatureCache(tmp_path / "cache")

    def test_a_saved_entry_records_the_digest_of_both_arrays(self, tmp_path):
        import numpy as np

        cache = self._cache(tmp_path)
        entry = cache.save("k", np.ones((4, 3), dtype=np.float32), np.arange(4))

        assert len(entry.meta["features_sha256"]) == 64
        assert len(entry.meta["labels_sha256"]) == 64

    def test_labels_left_over_from_a_previous_write_are_refused(self, tmp_path):
        """The exact mixed state: new features, old labels, old meta."""
        import numpy as np

        cache = self._cache(tmp_path)
        cache.save("k", np.zeros((4, 3), dtype=np.float32), np.array([0, 0, 0, 0]))
        features_path, _, _ = cache.paths_for("k")

        # A second write that got as far as the features and then died.
        np.save(features_path, np.ones((4, 3), dtype=np.float32))

        assert cache.load("k") is None, "sirvio features nuevas con labels viejas"

    def test_features_left_over_from_a_previous_write_are_refused(self, tmp_path):
        import numpy as np

        cache = self._cache(tmp_path)
        cache.save("k", np.zeros((4, 3), dtype=np.float32), np.arange(4))
        _, labels_path, _ = cache.paths_for("k")

        np.save(labels_path, np.arange(4) + 100)

        assert cache.load("k") is None

    def test_an_intact_entry_still_loads(self, tmp_path):
        import numpy as np

        cache = self._cache(tmp_path)
        features = np.arange(12, dtype=np.float32).reshape(4, 3)
        cache.save("k", features, np.arange(4))

        loaded = cache.load("k")

        assert loaded is not None
        assert np.array_equal(loaded.features, features)

    def test_an_entry_written_before_digests_existed_still_loads(self, tmp_path):
        """Backward compatibility with the bank cache already on the pod.

        Those entries have no digests. Refusing them would silently re-extract
        the whole v6 bank and, worse, make a cache written by the published
        run look corrupt.
        """
        import json

        import numpy as np

        cache = self._cache(tmp_path)
        cache.save("k", np.zeros((4, 3), dtype=np.float32), np.arange(4))
        _, _, meta_path = cache.paths_for("k")
        meta = json.loads(meta_path.read_text())
        del meta["features_sha256"]
        del meta["labels_sha256"]
        meta_path.write_text(json.dumps(meta))

        assert cache.load("k") is not None

    def test_nothing_is_visible_until_the_whole_entry_is_written(self, tmp_path):
        """Each file lands by rename, so a reader never sees a partial one."""
        import numpy as np
        from pathlib import Path

        cache = self._cache(tmp_path)
        seen: list[str] = []
        original = Path.replace

        def spy(self, target):
            seen.append(Path(target).name)
            return original(self, target)

        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "replace", spy)
            cache.save("k", np.zeros((4, 3), dtype=np.float32), np.arange(4))

        assert len(seen) == 3, f"no todo se escribio por rename: {seen}"
        # The meta lands last: it is what makes the entry visible to `has`.
        assert seen[-1].endswith(".meta.json")
        assert not list((tmp_path / "cache").glob("*.tmp*"))


class TestC1RefusesEntriesWithoutADigest:
    """The legacy door is open for the v6 bank only.

    Entries written before digests existed still load, so the published bank
    cache on the pod keeps working. But that door would also let ANY stale
    cache from some earlier run walk into the 24 C1 cells -- and a cache that
    predates the centre-crop fix holds features extracted one row off.

    So C1 asks for digests. A missing one RAISES rather than returning None:
    None means "extract it then", which is safe but silent, and nobody would
    ever learn that something wrote an unverifiable entry into the C1 cache
    directory. That is a fact worth stopping for.
    """

    def _cache_with_legacy_entry(self, tmp_path):
        import json

        import numpy as np

        from dermssl.feature_cache import FeatureCache

        cache = FeatureCache(tmp_path / "cache")
        cache.save("k", np.zeros((4, 3), dtype=np.float32), np.arange(4))
        _, _, meta_path = cache.paths_for("k")
        meta = json.loads(meta_path.read_text())
        del meta["features_sha256"]
        del meta["labels_sha256"]
        meta_path.write_text(json.dumps(meta))
        return cache

    def test_the_default_still_accepts_a_legacy_entry(self, tmp_path):
        cache = self._cache_with_legacy_entry(tmp_path)

        assert cache.load("k") is not None

    def test_requiring_a_digest_raises_instead_of_returning_none(self, tmp_path):
        from dermssl.feature_cache import LegacyEntryRefused

        cache = self._cache_with_legacy_entry(tmp_path)

        with pytest.raises(LegacyEntryRefused, match="digest"):
            cache.load("k", require_digest=True)

    def test_a_digested_entry_passes_the_strict_read(self, tmp_path):
        import numpy as np

        from dermssl.feature_cache import FeatureCache

        cache = FeatureCache(tmp_path / "cache")
        features = np.arange(12, dtype=np.float32).reshape(4, 3)
        cache.save("k", features, np.arange(4))

        loaded = cache.load("k", require_digest=True)

        assert loaded is not None
        assert np.array_equal(loaded.features, features)

    def test_an_absent_entry_is_still_a_plain_miss(self, tmp_path):
        """Nothing to refuse: there is no entry, so 'extract it' is right."""
        from dermssl.feature_cache import FeatureCache

        cache = FeatureCache(tmp_path / "cache")

        assert cache.load("nope", require_digest=True) is None

    def test_get_or_extract_propagates_the_requirement(self, tmp_path):
        """Otherwise C1 would reach the legacy entry through the other door."""
        import numpy as np

        from dermssl.feature_cache import LegacyEntryRefused

        cache = self._cache_with_legacy_entry(tmp_path)

        def extract():
            raise AssertionError("no debio re-extraer: tenia que fallar antes")

        with pytest.raises(LegacyEntryRefused):
            cache.get_or_extract("k", extract, require_digest=True)
