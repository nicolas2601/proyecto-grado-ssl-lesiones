"""Step 0 must run the PUBLISHED inference, not an equivalent-looking one.

Two independent implementations of "run the U-Net over an image" already
diverged in this repo without raising anything: one lineage stores its weights
under ``model`` and the other under ``model_state_dict``, and the published
path normalises with ImageNet statistics while ``dermssl.torch_unet`` does not.
Both produce plausible masks. Neither produces an error.

So the contract tested here is not "the masks look right" -- it is "step 0 does
not have its own inference". ``predict_all`` is expected to *call* the published
functions, which is something a test can assert without a GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def script():
    """Load scripts/c1_predict_masks.py, which is a script rather than a package."""
    spec = importlib.util.spec_from_file_location(
        "c1_predict_masks", REPO / "scripts" / "c1_predict_masks.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def images(tmp_path):
    """Three 10x12 PNGs -- deliberately non-square and not the network's grid."""
    from PIL import Image

    paths = []
    for index in range(3):
        path = tmp_path / f"ISIC_000000{index}.png"
        Image.fromarray(
            np.full((12, 10, 3), 10 * index + 5, dtype=np.uint8)
        ).save(path)
        paths.append(path)
    return paths


@pytest.fixture
def stub_student(monkeypatch):
    """Replace the published inference with a recorder, keeping its signature.

    Patching the module attributes rather than the script's own names is
    deliberate: ``predict_all`` imports inside the function, so if it ever
    stopped going through ``eval_student`` the stub would not be reached and
    the call count assertion would fail.
    """
    import eval_student

    calls = {"resolutions": [], "batch_sizes": [], "tta": []}

    def fake_load_student(checkpoint_path, device):
        return object(), 8

    def fake_predict_masks(model, images, device, batch_size, tta=False):
        calls["resolutions"].append(images.shape[1:3])
        calls["batch_sizes"].append(batch_size)
        calls["tta"].append(tta)
        # A ramp, so an upsample is visible and a threshold is not the identity.
        column = np.linspace(0.0, 1.0, 8, dtype=np.float32)
        return np.broadcast_to(column, (len(images), 8, 8)).copy()

    monkeypatch.setattr(eval_student, "load_student", fake_load_student)
    monkeypatch.setattr(eval_student, "predict_masks", fake_predict_masks)
    return calls


class TestStepZeroDelegatesToThePublishedPath:
    def test_it_calls_eval_student_predict_masks(self, script, tmp_path, images, stub_student):
        script.predict_all(images, tmp_path / "ckpt.pt", "cpu", tmp_path / "m.npy", 2)
        assert stub_student["batch_sizes"], "predict_masks was never called"

    def test_it_never_enables_tta(self, script, tmp_path, images, stub_student):
        """The published Dice is the no-TTA number; C1 must not quietly beat it."""
        script.predict_all(images, tmp_path / "ckpt.pt", "cpu", tmp_path / "m.npy", 2)
        assert stub_student["tta"] == [False, False]
        assert script.USE_TTA is False

    def test_it_feeds_the_resolution_the_checkpoint_declares(
        self, script, tmp_path, images, stub_student
    ):
        """load_student returns 8 here; a hardcoded 224 would show up as (224, 224)."""
        _, _, resolution = script.predict_all(
            images, tmp_path / "ckpt.pt", "cpu", tmp_path / "m.npy", 2
        )
        assert resolution == 8
        assert stub_student["resolutions"] == [(8, 8), (8, 8)]

    def test_it_honours_the_batch_size(self, script, tmp_path, images, stub_student):
        script.predict_all(images, tmp_path / "ckpt.pt", "cpu", tmp_path / "m.npy", 2)
        assert stub_student["batch_sizes"] == [2, 1]

    def test_ids_stay_in_path_order(self, script, tmp_path, images, stub_student):
        _, image_ids, _ = script.predict_all(
            images, tmp_path / "ckpt.pt", "cpu", tmp_path / "m.npy", 2
        )
        assert image_ids == [path.stem for path in images]


class TestMasksLandOnTheOriginalGrid:
    def test_shape_is_the_image_shape_not_the_network_grid(
        self, script, tmp_path, images, stub_student
    ):
        masks, _, _ = script.predict_all(
            images, tmp_path / "ckpt.pt", "cpu", tmp_path / "m.npy", 2
        )
        assert masks.shape == (3, 12, 10)

    def test_ragged_input_is_refused(self, script, tmp_path, images, stub_student):
        """A positional index over a ragged stack silently pairs the wrong mask."""
        from PIL import Image

        odd = tmp_path / "ISIC_0000009.png"
        Image.fromarray(np.zeros((7, 7, 3), dtype=np.uint8)).save(odd)
        with pytest.raises(SystemExit, match="ragged|ABORT"):
            script.predict_all(
                [*images, odd], tmp_path / "ckpt.pt", "cpu", tmp_path / "m.npy", 4
            )


class TestUpsampleHappensBeforeTheThreshold:
    """Resizing labels interpolates between 0 and 1, which is a second threshold."""

    def test_probabilities_survive_the_resize(self, script):
        probabilities = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        resized = script.upsample_probabilities(probabilities, 8, 8)
        assert resized.shape == (8, 8)
        interior = resized[:, 1:-1]
        assert ((interior > 0.0) & (interior < 1.0)).any(), (
            "a bilinear resize of a 0/1 ramp must produce intermediate values; "
            "if everything is 0 or 1 the map was binarised first"
        )

    def test_same_shape_is_returned_untouched(self, script):
        probabilities = np.full((4, 5), 0.37, dtype=np.float32)
        np.testing.assert_array_equal(
            script.upsample_probabilities(probabilities, 4, 5), probabilities
        )

    def test_the_two_orders_disagree(self, script):
        """The reason the order is specified rather than left to taste.

        A barely-confident pixel (0,55) is the case that separates them: resized
        as a probability it stays barely confident and shrinks to a single pixel;
        thresholded first it becomes a full 1,0 and the resize spreads that
        certainty over five. Five times the lesion area from the same tensor.
        """
        probabilities = np.zeros((3, 3), dtype=np.float32)
        probabilities[1, 1] = 0.55

        prob_then_threshold = script.upsample_probabilities(probabilities, 9, 9) > 0.5
        threshold_then_upsample = (
            script.upsample_probabilities((probabilities > 0.5).astype(np.float32), 9, 9) > 0.5
        )

        assert prob_then_threshold.sum() == 1
        assert threshold_then_upsample.sum() == 5
        assert not np.array_equal(prob_then_threshold, threshold_then_upsample)

    def test_the_threshold_comes_from_the_published_module(self, script):
        from evaluate_segmentation import DEFAULT_THRESHOLD

        assert script._published_threshold() == DEFAULT_THRESHOLD
        assert isinstance(script._published_threshold(), float)


class TestInferenceIsRecordedInTheManifest:
    def test_the_constants_match_what_the_manifest_expects(self, script):
        from dermssl import c1_mask_export as mx

        assert set(mx.INFERENCE_KEYS) == {
            "path",
            "tta",
            "resolution",
            "threshold",
            "upsample",
        }
        assert script.INFERENCE_PATH.endswith("eval_student.predict_masks")
        assert script.UPSAMPLE == "bilinear_prob_then_threshold"


class TestProbabilitiesAreCheckedBeforeTheyBecomeAMask:
    """Round 7 (P1). ``predict_masks`` returns ``torch.sigmoid`` output, so
    [0, 1] holds today. The contract is at the callee anyway, for the same
    reason ``image_contract`` is: the next person to change the inference path
    will not re-derive this invariant, and every way of breaking it is silent.

    NaN is the sharpest case. PIL propagates it through the resize, and
    ``nan > 0.5`` is False, so a NaN probability map becomes an EMPTY mask --
    which then trips the bbox/CLS fallback and looks like a hard image rather
    than a broken forward pass.
    """

    def test_a_nan_probability_is_refused(self, script):
        probabilities = np.full((4, 4), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN|finit"):
            script.upsample_probabilities(probabilities, 8, 8)

    def test_a_single_nan_among_valid_values_is_refused(self, script):
        probabilities = np.full((4, 4), 0.7, dtype=np.float32)
        probabilities[2, 2] = np.nan
        with pytest.raises(ValueError, match="NaN|finit"):
            script.upsample_probabilities(probabilities, 8, 8)

    def test_values_above_one_are_refused(self, script):
        """Logits instead of probabilities: every pixel clears 0,5 at once."""
        probabilities = np.full((2, 2), 2.0, dtype=np.float32)
        with pytest.raises(ValueError, match=r"\[0, 1\]|rango|range"):
            script.upsample_probabilities(probabilities, 4, 4)

    def test_negative_values_are_refused(self, script):
        probabilities = np.full((2, 2), -0.5, dtype=np.float32)
        with pytest.raises(ValueError, match=r"\[0, 1\]|rango|range"):
            script.upsample_probabilities(probabilities, 4, 4)

    def test_an_empty_map_is_refused(self, script):
        """An empty array satisfies every range check vacuously."""
        with pytest.raises(ValueError, match="vac|empty|sin pixel"):
            script.upsample_probabilities(np.zeros((0, 4), dtype=np.float32), 4, 4)

    def test_the_check_also_runs_when_no_resize_is_needed(self, script):
        """The early return for a matching shape must not skip the contract."""
        probabilities = np.full((4, 4), np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="NaN|finit"):
            script.upsample_probabilities(probabilities, 4, 4)

    def test_the_valid_range_still_passes(self, script):
        probabilities = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
        resized = script.upsample_probabilities(probabilities, 4, 4)
        assert resized.shape == (4, 4)


class TestCropGates:
    """Thresholds fixed in docs/27 §10.3 BEFORE the data, so they are a
    prediction rather than a rationalisation of whatever came out."""

    def _summary(self, **over):
        payload = {
            "n_images": 10_015,
            "n_with_lesion": 10_000,
            "n_empty_from_segmenter": 15,
            "n_emptied_by_crop": 50,
            "n_below_half": 200,
            "fraction_empty_from_segmenter": 15 / 10_015,
            "fraction_emptied_by_crop": 50 / 10_015,
            "fraction_below_half": 200 / 10_000,
        }
        payload.update(over)
        return payload

    def test_the_predicted_case_passes(self, script):
        ok, reasons = script.evaluate_crop_gates(self._summary())
        assert ok
        assert reasons == []

    def test_too_many_emptied_by_crop_fails(self, script):
        ok, reasons = script.evaluate_crop_gates(
            self._summary(fraction_emptied_by_crop=0.025)
        )
        assert not ok
        assert any("emptied_by_crop" in r or "recorte" in r for r in reasons)

    def test_exactly_at_the_threshold_passes(self, script):
        """The gate is 'more than 2%', so 2,0% is not a failure."""
        ok, _ = script.evaluate_crop_gates(self._summary(fraction_emptied_by_crop=0.02))
        assert ok

    def test_too_many_below_half_fails(self, script):
        ok, reasons = script.evaluate_crop_gates(self._summary(fraction_below_half=0.06))
        assert not ok
        assert any("below_half" in r or "mitad" in r for r in reasons)

    def test_both_can_fail_at_once(self, script):
        ok, reasons = script.evaluate_crop_gates(
            self._summary(fraction_emptied_by_crop=0.9, fraction_below_half=0.9)
        )
        assert not ok
        assert len(reasons) == 2, "both failures must be reported, not just the first"

    def test_empty_from_segmenter_is_reported_not_gated(self, script):
        """It is the segmenter's problem, not the crop's; it must not FAIL here."""
        ok, _ = script.evaluate_crop_gates(
            self._summary(fraction_empty_from_segmenter=0.9)
        )
        assert ok

    def test_a_missing_below_half_rate_is_refused(self, script):
        """None means no mask had a lesion at all -- that is not a pass."""
        ok, reasons = script.evaluate_crop_gates(self._summary(fraction_below_half=None))
        assert not ok
        assert any("lesion" in r for r in reasons)

    def test_the_thresholds_are_the_documented_ones(self, script):
        assert script.MAX_FRACTION_EMPTIED_BY_CROP == 0.02
        assert script.MAX_FRACTION_BELOW_HALF == 0.05


class TestReferenceGate:
    """The artifact has to BE the published segmentation, not merely resemble it.

    Same discipline as the ISIC check: paired per-image comparison against the
    published per_image.csv, never a comparison of means. The reference masks
    are read ONLY to verify the artifact and never enter C1.
    """

    def _paired(self, **over):
        payload = {
            "n_matched": 10_015,
            "n_published": 10_015,
            "max_abs_diff": 8e-6,
            "mean_abs_diff": 1.2e-6,
        }
        payload.update(over)
        return payload

    def test_a_faithful_artifact_passes(self, script):
        ok, reasons = script.evaluate_reference_gate(self._paired())
        assert ok
        assert reasons == []

    def test_drift_above_the_threshold_fails(self, script):
        ok, reasons = script.evaluate_reference_gate(self._paired(max_abs_diff=2e-3))
        assert not ok
        assert any("max_abs_diff" in r or "publicad" in r for r in reasons)

    def test_the_threshold_itself_passes(self, script):
        """The gate is 'above 1e-3', so exactly 1e-3 is not a failure."""
        ok, _ = script.evaluate_reference_gate(self._paired(max_abs_diff=1e-3))
        assert ok

    def test_an_unmatched_image_fails(self, script):
        """A partial join would average over whatever happened to match."""
        ok, reasons = script.evaluate_reference_gate(self._paired(n_matched=10_000))
        assert not ok
        assert any("empareja" in r or "matched" in r for r in reasons)

    def test_nan_drift_cannot_pass(self, script):
        """`nan > 1e-3` is False, which is how a gate becomes a no-op."""
        ok, reasons = script.evaluate_reference_gate(
            self._paired(max_abs_diff=float("nan"))
        )
        assert not ok
        assert any("NaN" in r or "finit" in r for r in reasons)

    def test_zero_matches_is_not_a_pass(self, script):
        ok, _ = script.evaluate_reference_gate(
            self._paired(n_matched=0, n_published=0)
        )
        assert not ok

    def test_the_threshold_is_the_documented_one(self, script):
        assert script.MAX_ABS_DIFF_VS_PUBLISHED == 1e-3
