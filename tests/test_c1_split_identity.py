"""Identity of the split C1 runs on.

The bank's own guarantee that two runs used the same images is the cache key,
which embeds ``fingerprint_ids`` -- a sha256 of the ORDERED id list, truncated
to 16 hex. That is enough to make a stale cache miss, but too short and too
implicit to quote as provenance in a manifest.

So C1 records three things, each under its real name:

* ``feature_cache_ids_fingerprint`` -- the bank's own value, truncated, so a
  reader can line C1 up against the v6 cache filenames.
* ``c1_train_ids_sha256`` / ``c1_val_ids_sha256`` -- full digests, declared as
  NEW to this experiment. There is no "split_hash of the bank": that name
  exists in the repo but hashes the PRE-TRAINING pool, and borrowing it would
  put a number in the manifest that looks like traceability and is not.
"""

from __future__ import annotations

import pytest

from dermssl import c1_split_identity as si


class TestDigests:
    def test_the_full_digest_is_not_truncated(self):
        assert len(si.ids_sha256(["a", "b"])) == 64

    def test_order_changes_the_digest(self):
        """Features are stored aligned to the id order, so a reordered split
        is a different split even with identical membership."""
        assert si.ids_sha256(["a", "b"]) != si.ids_sha256(["b", "a"])

    def test_it_matches_the_bank_fingerprint_on_its_first_16_chars(self):
        """Same construction, different length: the bank truncates.

        Pinned so the two can never silently drift apart -- if the bank ever
        changes how it builds the fingerprint, this fails instead of leaving
        C1 quoting a number that no longer lines up with the cache names.
        """
        from dermssl.feature_cache import fingerprint_ids

        ids = ["ISIC_0024306", "ISIC_0024307", "ISIC_0024308"]

        assert si.ids_sha256(ids)[:16] == fingerprint_ids(ids)

    def test_an_empty_id_list_is_refused(self):
        """A digest of nothing is a valid-looking constant that would match
        any other empty split."""
        with pytest.raises(ValueError, match="vacia"):
            si.ids_sha256([])


class TestSplitIdentity:
    def test_it_records_all_three_values_under_their_real_names(self):
        identity = si.split_identity(train_ids=["a", "b"], val_ids=["c"])

        assert set(identity) == {
            "c1_train_ids_sha256",
            "c1_val_ids_sha256",
            "feature_cache_ids_fingerprint",
            "n_train",
            "n_val",
            "fingerprint_note",
        }

    def test_the_fingerprint_is_the_one_the_cache_uses_for_val(self):
        """The cache key for the val split is what names the feature files."""
        from dermssl.feature_cache import fingerprint_ids

        identity = si.split_identity(train_ids=["a"], val_ids=["c", "d"])

        assert identity["feature_cache_ids_fingerprint"] == fingerprint_ids(["c", "d"])

    def test_the_note_says_it_is_truncated(self):
        """A 16-char hex next to two 64-char ones invites the reader to assume
        they are the same kind of thing."""
        identity = si.split_identity(train_ids=["a"], val_ids=["c"])

        assert "16" in identity["fingerprint_note"]

    def test_there_is_no_key_called_split_hash(self):
        """That name belongs to the pre-training pool, not to this split."""
        identity = si.split_identity(train_ids=["a"], val_ids=["c"])

        assert not any("split_hash" in key for key in identity)

    def test_an_id_on_both_sides_is_refused(self):
        with pytest.raises(ValueError, match="ambos lados"):
            si.split_identity(train_ids=["a", "b"], val_ids=["b"])


class TestComparingAgainstAnExistingIndex:
    def test_an_identical_index_agrees(self):
        stored = {"train_ids": ["a", "b"], "val_ids": ["c"]}

        agrees, reasons = si.compare_to_index(stored, train_ids=["a", "b"], val_ids=["c"])

        assert agrees is True
        assert reasons == []

    def test_a_different_order_is_a_difference(self):
        """Same members, different order: the features would be aligned to the
        other order and every row would be mislabelled."""
        stored = {"train_ids": ["a", "b"], "val_ids": ["c", "d"]}

        agrees, reasons = si.compare_to_index(stored, train_ids=["a", "b"], val_ids=["d", "c"])

        assert agrees is False
        assert any("orden" in reason for reason in reasons)

    def test_a_missing_id_is_reported_with_how_many(self):
        stored = {"train_ids": ["a", "b"], "val_ids": ["c", "d"]}

        agrees, reasons = si.compare_to_index(stored, train_ids=["a", "b"], val_ids=["c"])

        assert agrees is False
        assert any("val" in reason for reason in reasons)

    def test_both_sides_are_checked_not_just_val(self):
        """The gate only ever reads val_ids, so a train difference would go
        unnoticed -- and the probe is FIT on train."""
        stored = {"train_ids": ["a", "b"], "val_ids": ["c"]}

        agrees, reasons = si.compare_to_index(stored, train_ids=["a", "z"], val_ids=["c"])

        assert agrees is False
        assert any("train" in reason for reason in reasons)

    def test_an_index_without_train_ids_is_reported_not_assumed_equal(self):
        """The file on the pod may predate this script. Treating an absent
        side as 'matches' is the vacuous pass this whole check exists to
        prevent."""
        stored = {"val_ids": ["c"]}

        agrees, reasons = si.compare_to_index(stored, train_ids=["a"], val_ids=["c"])

        assert agrees is False
        assert any("train_ids" in reason for reason in reasons)


class TestLesionLeakage:
    def test_a_lesion_on_both_sides_is_reported(self):
        """StratifiedGroupKFold should prevent it; verified anyway, because a
        lesion in both splits leaks the answer into the probe's training."""
        lesions = {"a": "L1", "b": "L2", "c": "L1"}

        shared = si.shared_lesions(train_ids=["a", "b"], val_ids=["c"], lesion_by_id=lesions)

        assert shared == ["L1"]

    def test_a_clean_split_shares_nothing(self):
        lesions = {"a": "L1", "b": "L2", "c": "L3"}

        assert si.shared_lesions(train_ids=["a", "b"], val_ids=["c"], lesion_by_id=lesions) == []

    def test_an_unknown_id_is_refused_rather_than_skipped(self):
        """Skipping ids with no lesion would make the check pass by knowing
        less."""
        with pytest.raises(KeyError):
            si.shared_lesions(train_ids=["a"], val_ids=["zzz"], lesion_by_id={"a": "L1"})
