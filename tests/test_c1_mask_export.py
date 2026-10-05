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
            "inference": {
                "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
                "tta": False,
                "resolution": 224,
                "threshold": 0.5,
                "upsample": "bilinear_prob_then_threshold",
            },
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

    def test_records_the_inference_that_produced_the_masks(self):
        """The checkpoint alone does not identify a mask."""
        inference = self._manifest()["inference"]
        assert inference["path"].endswith("eval_student.predict_masks")
        assert inference["tta"] is False
        assert inference["resolution"] == 224
        assert inference["threshold"] == 0.5
        assert inference["upsample"] == "bilinear_prob_then_threshold"


class TestInferenceBlockHasNoDefaults:
    """Same failure mode as ``segmentation_run``: a producer describing itself.

    A default of ``tta: False`` would let a TTA run record ``tta: false`` simply
    because the caller forgot the argument, and the masks would be documented as
    something they are not.
    """

    def _inference(self, **over):
        payload = {
            "path": "plan-segmentacion/comun/evaluacion/eval_student.predict_masks",
            "tta": False,
            "resolution": 224,
            "threshold": 0.5,
            "upsample": "bilinear_prob_then_threshold",
        }
        payload.update(over)
        return payload

    @pytest.mark.parametrize("dropped", mx.INFERENCE_KEYS)
    def test_every_field_is_required(self, dropped):
        payload = self._inference()
        del payload[dropped]
        with pytest.raises(ValueError, match=dropped):
            mx._validate_inference(payload)

    def test_unknown_keys_are_refused(self):
        """A typo would otherwise be silently dropped and the field left unrecorded."""
        with pytest.raises(ValueError, match="unexpected|tta_enabled"):
            mx._validate_inference(self._inference(tta_enabled=True))

    @pytest.mark.parametrize("truthy", ["false", 1, "no"])
    def test_a_truthy_tta_is_not_a_bool(self, truthy):
        """`bool("false")` is True; coercion here would invert the record."""
        with pytest.raises(TypeError, match="tta"):
            mx._validate_inference(self._inference(tta=truthy))

    def test_resolution_must_be_an_int(self):
        with pytest.raises(TypeError, match="resolution"):
            mx._validate_inference(self._inference(resolution="224"))

    def test_a_bool_is_not_a_resolution(self):
        """`isinstance(True, int)` is True, so bool has to be excluded by hand."""
        with pytest.raises(TypeError, match="resolution"):
            mx._validate_inference(self._inference(resolution=True))

    def test_threshold_must_be_a_float(self):
        with pytest.raises(TypeError, match="threshold"):
            mx._validate_inference(self._inference(threshold="0.5"))

    def test_the_valid_block_is_returned_with_exactly_the_known_keys(self):
        assert mx._validate_inference(self._inference()) == self._inference()


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
