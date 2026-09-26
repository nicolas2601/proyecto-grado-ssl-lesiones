"""Tests for the C1 step-0 mask artifact.

The 24 C1 cells all consume one `.npy` of pseudo-masks. If that file is not
pinned by hash, nobody can prove afterwards which masks produced which numbers,
and the experiment stops being reproducible even though every individual run
looks fine. These tests exist to make the provenance mandatory rather than
customary.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dermssl import c1_mask_export as mx


class TestEncodeMask:
    def test_bool_becomes_zero_and_one(self):
        mask = np.array([[True, False], [False, True]])
        encoded = mx.encode_mask(mask)
        assert encoded.dtype == np.uint8
        assert set(np.unique(encoded).tolist()) == {0, 1}

    def test_lesion_maps_to_the_lesion_label(self):
        encoded = mx.encode_mask(np.array([[True, False]]))
        assert encoded[0, 0] == mx.LESION_LABEL
        assert encoded[0, 1] == 0

    def test_rejects_non_boolean_input(self):
        """A float probability map must not be silently truncated to a mask."""
        with pytest.raises(TypeError, match="bool"):
            mx.encode_mask(np.array([[0.7, 0.2]], dtype=np.float32))


class TestManifest:
    def _manifest(self, **overrides):
        payload = {
            "n_images": 3,
            "image_ids": ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"],
            "masks_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
            "checkpoint_path": "salidas/final/student_r3/best.pt",
            "code_revision": "4ef40fd",
            "mask_shape": [450, 600],
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
        }
        payload.update(overrides)
        return mx.build_masks_manifest(**payload)

    def test_carries_both_hashes(self):
        manifest = self._manifest()
        assert manifest["masks_sha256"] == "a" * 64
        assert manifest["checkpoint_sha256"] == "b" * 64

    def test_records_the_label_convention_explicitly(self):
        """Whoever reads this later must not have to guess what 1 means."""
        manifest = self._manifest()
        assert manifest["label_convention"]["lesion"] == mx.LESION_LABEL
        assert manifest["label_convention"]["background"] == 0

    def test_documents_that_student_r3_emits_no_uncertain_band(self):
        manifest = self._manifest()
        assert manifest["label_convention"]["uncertain"] is None

    def test_carries_code_revision(self):
        assert self._manifest()["code_revision"] == "4ef40fd"

    def test_rejects_id_count_that_disagrees_with_n_images(self):
        with pytest.raises(ValueError, match="image_ids"):
            self._manifest(n_images=5)

    def test_rejects_duplicate_image_ids(self):
        """Duplicates would make the mask index ambiguous for the whole grid."""
        with pytest.raises(ValueError, match="duplicate"):
            self._manifest(image_ids=["ISIC_0000001", "ISIC_0000001", "ISIC_0000003"])

    def test_rejects_a_hash_that_is_not_sha256(self):
        with pytest.raises(ValueError, match="sha256"):
            self._manifest(masks_sha256="deadbeef")

    def test_is_json_serialisable(self):
        json.dumps(self._manifest())


class TestArtifactVerification:
    def test_matching_hash_passes(self, tmp_path):
        path = tmp_path / "masks.npy"
        np.save(path, np.zeros((2, 4, 4), dtype=np.uint8))
        mx.verify_artifact(path, mx.sha256_file(path))

    def test_altered_file_is_rejected(self, tmp_path):
        """The point of the hash: a silently regenerated file must not pass."""
        path = tmp_path / "masks.npy"
        np.save(path, np.zeros((2, 4, 4), dtype=np.uint8))
        recorded = mx.sha256_file(path)
        np.save(path, np.ones((2, 4, 4), dtype=np.uint8))
        with pytest.raises(ValueError, match="sha256"):
            mx.verify_artifact(path, recorded)


class TestCoverageSummary:
    def test_reports_empty_masks(self):
        masks = np.zeros((3, 8, 8), dtype=np.uint8)
        masks[0, 2:5, 2:5] = mx.LESION_LABEL
        summary = mx.summarise_coverage(masks)
        assert summary["n_empty"] == 2

    def test_reports_mean_foreground_fraction(self):
        masks = np.zeros((2, 10, 10), dtype=np.uint8)
        masks[0] = mx.LESION_LABEL  # fully lesion
        summary = mx.summarise_coverage(masks)
        assert summary["mean_foreground_fraction"] == pytest.approx(0.5)

    def test_empty_masks_are_flagged_not_hidden(self):
        """An all-empty batch means the segmenter failed, not that lesions vanished."""
        summary = mx.summarise_coverage(np.zeros((4, 8, 8), dtype=np.uint8))
        assert summary["n_empty"] == 4
        assert summary["mean_foreground_fraction"] == 0.0
