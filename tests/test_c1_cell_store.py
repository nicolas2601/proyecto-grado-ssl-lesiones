"""Per-cell storage for the 24-cell grid: atomic, resumable, sha-verified.

The pod died four times during v6. A grid that loses everything on the fourth
crash is a grid that never finishes, so each cell lands on disk the moment it
is done and a relaunch skips it.

The dangerous half is the skip. Skipping because a FILE EXISTS trusts the
name; a cell extracted before the centre-crop fix has the same name and
features that are one row off. So a cell is skipped only when its manifest
says it came from the same code revision, the same masks, the same split and
the same condition -- and when its features still hash to what the manifest
recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import c1_cell_store as cs
from dermssl.feature_cache import _digest

PROVENANCE = {
    "code_revision": "abc123",
    "masks_sha256": "f" * 64,
    "c1_train_ids_sha256": "a" * 64,
    "c1_val_ids_sha256": "b" * 64,
    "probe": "logreg",
    "backbone_spec": "dinov2_vitb14:public",
    "labels_sha256": "c" * 64,
}


def _proba(n_rows: int = 6) -> np.ndarray:
    return np.full((n_rows, 3), 1 / 3)


def _features(n_rows: int = 6, n_dims: int = 4) -> np.ndarray:
    return np.arange(n_rows * n_dims, dtype=np.float32).reshape(n_rows, n_dims)


def _train(n_rows: int = 9, n_dims: int = 4) -> np.ndarray:
    return np.linspace(0, 1, n_rows * n_dims, dtype=np.float32).reshape(n_rows, n_dims)


class TestSavingACell:
    def test_a_saved_cell_can_be_read_back(self, tmp_path):
        store = cs.CellStore(tmp_path)
        features = _features()

        store.save("dinov2_public_vitb14", "mask_pool", features, proba=_proba(), train_features=_train(), provenance=PROVENANCE)
        loaded = store.load("dinov2_public_vitb14", "mask_pool", provenance=PROVENANCE)

        assert loaded is not None
        assert np.array_equal(loaded.features, features)
        assert loaded.manifest["condition"] == "mask_pool"

    def test_nothing_partial_is_ever_visible(self, tmp_path):
        """A relaunch must not find a half-written cell and trust it."""
        from pathlib import Path

        store = cs.CellStore(tmp_path)
        seen: list[str] = []
        original = Path.replace

        def spy(self, target):
            seen.append(Path(target).name)
            return original(self, target)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "replace", spy)
            store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance=PROVENANCE)

        assert seen, "no se escribio por rename"
        assert seen[-1].endswith(".manifest.json"), (
            "el manifiesto no se escribio ultimo: una corrida podria verlo antes que "
            "los features que describe"
        )
        assert not list(tmp_path.glob("*.tmp*"))

    def test_the_manifest_records_what_it_was_built_from(self, tmp_path):
        store = cs.CellStore(tmp_path)

        store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance=PROVENANCE)
        loaded = store.load("b", "none", provenance=PROVENANCE)

        for key, value in PROVENANCE.items():
            assert loaded.manifest[key] == value
        assert len(loaded.manifest["features_sha256"]) == 64


class TestTheSkipIsEarned:
    def _saved(self, tmp_path, **overrides):
        store = cs.CellStore(tmp_path)
        store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance={**PROVENANCE, **overrides})
        return store

    def test_a_cell_from_another_code_revision_is_not_reused(self, tmp_path):
        """The centre-crop fix changed what extraction produces. A cell from
        before it has the same filename and features one row off."""
        store = self._saved(tmp_path, code_revision="old_commit")

        assert store.load("b", "none", provenance=PROVENANCE) is None

    def test_a_cell_built_on_other_masks_is_not_reused(self, tmp_path):
        store = self._saved(tmp_path, masks_sha256="0" * 64)

        assert store.load("b", "none", provenance=PROVENANCE) is None

    def test_a_cell_built_on_another_split_is_not_reused(self, tmp_path):
        store = self._saved(tmp_path, c1_val_ids_sha256="0" * 64)

        assert store.load("b", "none", provenance=PROVENANCE) is None

    def test_tampered_features_are_refused(self, tmp_path):
        """Existence is not integrity: the file may have been truncated, or
        replaced by a different run that crashed mid-write."""
        store = self._saved(tmp_path)
        path = store.features_path("b", "none")
        np.save(path, _features() + 1)

        assert store.load("b", "none", provenance=PROVENANCE) is None

    def test_an_absent_cell_is_a_plain_miss(self, tmp_path):
        store = cs.CellStore(tmp_path)

        assert store.load("nope", "none", provenance=PROVENANCE) is None

    def test_a_cell_of_another_condition_is_not_returned(self, tmp_path):
        """Each condition is a different experiment; the key must separate
        them or (c) would be served the features of (0)."""
        store = cs.CellStore(tmp_path)
        store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance=PROVENANCE)

        assert store.load("b", "mask_pool", provenance=PROVENANCE) is None


class TestPlanningTheGrid:
    def test_it_reports_which_cells_are_done_and_which_remain(self, tmp_path):
        store = cs.CellStore(tmp_path)
        store.save("b1", "none", _features(), proba=_proba(), train_features=_train(), provenance=PROVENANCE)

        plan = store.plan(
            backbones=["b1", "b2"],
            conditions=["none", "mask_pool"],
            provenance=PROVENANCE,
        )

        assert plan["done"] == [("b1", "none")]
        assert ("b2", "mask_pool") in plan["pending"]
        assert len(plan["pending"]) == 3

    def test_the_order_puts_the_professors_question_first(self, tmp_path):
        """The 8 cells of dinov2_public vs ssl_proposed ViT-B answer the
        question that was actually asked; they run before the other 16."""
        store = cs.CellStore(tmp_path)

        plan = store.plan(
            backbones=["ssl_plain_vits14", "ssl_proposed_vitb14", "dinov2_public_vitb14"],
            conditions=["none", "mask_pool"],
            provenance=PROVENANCE,
            priority_backbones=["dinov2_public_vitb14", "ssl_proposed_vitb14"],
        )

        first_four = plan["pending"][:4]
        assert {backbone for backbone, _ in first_four} == {
            "dinov2_public_vitb14",
            "ssl_proposed_vitb14",
        }

    def test_an_unknown_condition_is_refused_before_any_gpu_time(self, tmp_path):
        store = cs.CellStore(tmp_path)

        with pytest.raises(ValueError, match="condicion"):
            store.plan(backbones=["b"], conditions=["inventada"], provenance=PROVENANCE)

    def test_missing_provenance_is_refused(self, tmp_path):
        """A cell saved without its provenance can never be safely reused, and
        the run would look resumable while not being."""
        store = cs.CellStore(tmp_path)

        with pytest.raises(ValueError, match="procedencia"):
            store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance={"code_revision": "abc"})


class TestTheManifestMustAgreeWithItsOwnFilename:
    """Defence in depth, pinned separately.

    The key already encodes backbone and condition, so asking for another
    condition simply misses. That makes the manifest's own backbone/condition
    fields redundant -- until a file is renamed, copied between directories,
    or restored from a backup of a different cell. Then the name says one
    thing and the contents another, and without this check the mismatched
    features load as if they belonged.

    Each layer gets its own test, or the day the outer one is relaxed the
    inner one turns out never to have been exercised.
    """

    def _saved_with_manifest(self, tmp_path, **manifest_overrides):
        import json

        store = cs.CellStore(tmp_path)
        store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance=PROVENANCE)
        path = store.manifest_path("b", "none")
        manifest = json.loads(path.read_text())
        manifest.update(manifest_overrides)
        path.write_text(json.dumps(manifest))
        return store

    def test_a_manifest_claiming_another_condition_is_refused(self, tmp_path):
        store = self._saved_with_manifest(tmp_path, condition="mask_pool")

        assert store.load("b", "none", provenance=PROVENANCE) is None

    def test_a_manifest_claiming_another_backbone_is_refused(self, tmp_path):
        store = self._saved_with_manifest(tmp_path, backbone="otro_backbone")

        assert store.load("b", "none", provenance=PROVENANCE) is None


class TestTheProbaIsVerifiedToo:
    """Review P1-3: the proba is what enters the table. Verifying only the
    features let an old proba ride along with a new manifest."""

    def test_a_proba_from_another_run_is_refused(self, tmp_path):
        store = cs.CellStore(tmp_path)
        store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance=PROVENANCE)
        np.save(store.proba_path("b", "none"), _proba() * 0 + 0.5)

        assert store.load("b", "none", provenance=PROVENANCE) is None

    def test_a_missing_proba_is_a_miss_not_a_half_cell(self, tmp_path):
        store = cs.CellStore(tmp_path)
        store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance=PROVENANCE)
        store.proba_path("b", "none").unlink()

        assert store.load("b", "none", provenance=PROVENANCE) is None

    def test_another_probe_is_not_reused(self, tmp_path):
        store = cs.CellStore(tmp_path)
        store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance={**PROVENANCE, "probe": "knn"})

        assert store.load("b", "none", provenance=PROVENANCE) is None

    def test_the_proba_lands_before_the_manifest(self, tmp_path):
        from pathlib import Path

        store = cs.CellStore(tmp_path)
        seen: list[str] = []
        original = Path.replace

        def spy(self, target):
            seen.append(Path(target).name)
            return original(self, target)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "replace", spy)
            store.save("b", "none", _features(), proba=_proba(), train_features=_train(), provenance=PROVENANCE)

        assert seen.index(store.proba_path("b", "none").name) < seen.index(
            store.manifest_path("b", "none").name
        )


class TestTrainFeaturesAreHashed:
    """Only val features are stored; the train ones the probe was fitted on are
    hashed, so a published cell can be re-audited (e.g. the BLAS-thread check)
    against a CPU-side or scratch copy instead of trusting a GPU re-extraction."""

    def test_the_manifest_records_the_sha256_of_the_train_features(self, tmp_path):
        store = cs.CellStore(tmp_path)
        train = _train()

        manifest = store.save("b", "none", _features(), proba=_proba(), train_features=train,
                              provenance=PROVENANCE)

        assert manifest["train_features_sha256"] == _digest(train)
        assert store.load("b", "none", provenance=PROVENANCE).manifest["train_features_sha256"] == _digest(train)

    def test_the_train_hash_uses_the_val_convention(self, tmp_path):
        """Same function and dtype cast as ``features_sha256``: float64 input hashes as its float32 copy."""
        store = cs.CellStore(tmp_path)
        train = _train().astype(np.float64)

        manifest = store.save("b", "none", _features(), proba=_proba(), train_features=train,
                              provenance=PROVENANCE)

        assert manifest["train_features_sha256"] == _digest(train.astype(np.float32))

    def test_a_cell_cannot_be_saved_without_its_train_features(self, tmp_path):
        store = cs.CellStore(tmp_path)

        with pytest.raises(TypeError, match="train_features"):
            store.save("b", "none", _features(), proba=_proba(), provenance=PROVENANCE)
