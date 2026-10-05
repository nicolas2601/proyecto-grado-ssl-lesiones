"""Regression tests for the C1 review findings.

Every test here corresponds to a hole an independent reviewer found in code
whose own 37 tests were green. That is the useful lesson: the tests proved the
behaviour I thought about, not the behaviour I forgot. Each one fails against
the reviewed version.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _registered_checkpoint(monkeypatch):
    """Production fails closed until the pod measurement lands; these suites
    exercise what happens after that, so they register a known checkpoint."""
    from dermssl import c1_mask_provenance as _mp

    monkeypatch.setattr(_mp, "FINAL_STUDENT_R3_SHA256", "f" * 64)

from dermssl import c1_extraction as ex
from dermssl import c1_mask_provenance as mp
from dermssl import c1_provenance as pv
from dermssl import mask_conditions as mc


def _manifest(**overrides):
    payload = {
        "producer": "student_r3",
        "producer_is_label_free": True,
        "masks_sha256": "a" * 64,
        "checkpoint_sha256": "f" * 64,
        "image_ids": ["ISIC_0000001", "ISIC_0000002"],
        "mask_shape": [64, 64],
        "segmentation_run": "final",
        "segmentation_commit": "d81a492",
        "inference": dict(mp.PUBLISHED_INFERENCE),
    }
    payload.update(overrides)
    return payload


def _tracked(mask: np.ndarray, image_id: str = "ISIC_0000001", **overrides):
    return mp.TrackedMask(
        array=mask,
        source=mp.MaskSource.from_manifest(_manifest(**overrides)),
        image_id=image_id,
    )


def _square_mask(size: int = 64, box=(16, 48, 16, 48)) -> np.ndarray:
    y0, y1, x0, x1 = box
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


# ── #1 the image must be resized bilinear, like the bank ──────────────────────


class TestImageResizeMatchesBank:
    def test_image_resize_is_bilinear_not_nearest(self):
        """A smooth ramp must stay smooth. Nearest duplicates rows instead.

        The bank uses Resize(255, BILINEAR) + CenterCrop(224). Extracting
        condition (0) with nearest would produce different features from the
        cached ones and invalidate every C1 comparison, silently.
        """
        ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
        image = np.stack([ramp] * 3, axis=-1)
        out = ex._resize_square(image, 32)
        row = out[16, :, 0].astype(int)
        steps = np.unique(np.diff(row))
        assert len(steps) > 1 or steps[0] != 0, "nearest-style duplication detected"

    def test_mask_resize_stays_nearest_and_binary(self):
        resized = ex._resize_square(_square_mask(), 32)
        assert resized.dtype == np.bool_
        assert set(np.unique(resized).tolist()) <= {False, True}

    def test_bilinear_interpolates_between_neighbours(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        image[0, 0] = 0
        image[0, 1] = 100
        out = ex._resize_square(image, 4)
        assert 0 < int(out[0, 1, 0]) < 100, "no interpolation happened"


# ── #2 mask provenance is mandatory ───────────────────────────────────────────


class TestMaskProvenance:
    def test_student_r3_manifest_is_accepted(self):
        assert mp.MaskSource.from_manifest(_manifest()).producer == "student_r3"

    def test_reference_masks_are_refused_structurally(self):
        """The SSL rule as code: a supervised source cannot even be constructed."""
        with pytest.raises(ValueError, match="producer"):
            mp.MaskSource.from_manifest(_manifest(producer="ground_truth"))

    def test_source_that_is_not_label_free_is_refused(self):
        with pytest.raises(ValueError, match="label"):
            mp.MaskSource.from_manifest(_manifest(producer_is_label_free=False))

    def test_extraction_refuses_a_bare_array(self):
        with pytest.raises(TypeError, match="TrackedMask"):
            ex.build_eval_input(np.zeros((64, 64, 3), np.uint8), _square_mask(), "none")

    def test_image_id_absent_from_the_manifest_is_refused(self):
        with pytest.raises(ValueError, match="image_id"):
            _tracked(_square_mask(), image_id="ISIC_9999999")


# ── #7 mask dtype ─────────────────────────────────────────────────────────────


class TestMaskDtype:
    def test_uint8_mask_is_refused_before_it_reaches_the_tilde(self):
        """``~mask`` on uint8 gives 254/255, which indexes instead of inverting."""
        raw = np.zeros((64, 64), dtype=np.uint8)
        raw[16:48, 16:48] = 1
        with pytest.raises(TypeError, match="bool"):
            _tracked(raw)

    def test_lesion_mask_is_the_single_conversion_boundary(self):
        raw = np.zeros((64, 64), dtype=np.uint8)
        raw[16:48, 16:48] = mc.LESION_LABEL
        raw[0:4, 0:4] = mc.UNCERTAIN_LABEL
        tracked = _tracked(mc.lesion_mask(raw))
        assert tracked.array.dtype == np.bool_
        assert not tracked.array[0, 0]


# ── #4 unknown conditions must fail loudly ────────────────────────────────────


class TestUnknownCondition:
    def test_typo_does_not_fall_through_to_baseline(self):
        with pytest.raises(ValueError, match="condition"):
            ex.build_eval_input(
                np.zeros((64, 64, 3), np.uint8), _tracked(_square_mask()), "mask_poo"
            )

    @pytest.mark.parametrize("condition", mc.CONDITIONS)
    def test_every_registered_condition_is_accepted(self, condition):
        out = ex.build_eval_input(
            np.zeros((64, 64, 3), np.uint8), _tracked(_square_mask()), condition
        )
        assert out.image.shape[:2] == (ex.RESOLUTION, ex.RESOLUTION)


# ── #3 train/val isolation ────────────────────────────────────────────────────


class TestSplitIsolation:
    def _features(self, split: str, ids: list[str], condition: str = "mask_pool"):
        rng = np.random.default_rng(0)
        return pv.ConditionedFeatures(
            condition=condition,
            backbone="dinov2_public_vitb14",
            split=split,
            features=rng.normal(size=(len(ids), 8)).astype(np.float32),
            labels=np.arange(len(ids)) % 4,
            image_ids=tuple(ids),
            masks_sha256="a" * 64,
        )

    def test_swapped_splits_are_rejected(self):
        """opencode passed train as 'val' and the old guard stayed silent."""
        train = self._features("val", ["a", "b", "c", "d"])
        val = self._features("train", ["e", "f"])
        with pytest.raises(ValueError, match="split"):
            pv.fit_and_score(train, val, n_classes=4)

    def test_overlapping_image_ids_are_rejected(self):
        train = self._features("train", ["a", "b", "c", "d"])
        val = self._features("val", ["d", "e"])
        with pytest.raises(ValueError, match="overlap|leak"):
            pv.fit_and_score(train, val, n_classes=4)

    def test_different_mask_artifacts_are_rejected(self):
        train = self._features("train", ["a", "b", "c", "d"])
        val = pv.ConditionedFeatures(
            condition="mask_pool",
            backbone="dinov2_public_vitb14",
            split="val",
            features=np.zeros((2, 8), dtype=np.float32),
            labels=np.zeros(2, dtype=int),
            image_ids=("e", "f"),
            masks_sha256="b" * 64,
        )
        with pytest.raises(ValueError, match="masks_sha256"):
            pv.fit_and_score(train, val, n_classes=4)

    def test_invalid_split_name_rejected(self):
        with pytest.raises(ValueError, match="split"):
            self._features("training", ["a"])


# ── #5 fallback aggregation ───────────────────────────────────────────────────


class TestFallbackTally:
    def test_counts_and_rates_the_cls_fallbacks(self):
        tally = pv.FallbackTally()
        for used in (True, False, False, False):
            tally.record(used)
        assert tally.n_cls_fallback == 1
        assert tally.n_total == 4
        assert tally.rate == pytest.approx(0.25)

    def test_summary_is_json_ready(self):
        tally = pv.FallbackTally()
        tally.record(False)
        assert tally.summary() == {
            "n_cls_fallback": 0,
            "n_total": 1,
            "cls_fallback_rate": 0.0,
            "warning": False,
        }

    def test_rate_is_zero_when_nothing_was_seen(self):
        assert pv.FallbackTally().rate == 0.0


# ── bank reproduction gate ────────────────────────────────────────────────────


class TestBankReproduction:
    """Condition (0) must reproduce the cached bank features, or C1 is invalid.

    This is the gate that attacks the nearest-vs-bilinear bug at the root
    instead of by symptom: if the extraction path drifts in any way, (0) stops
    matching the cache and the whole 6x4 table is comparing against a baseline
    that is not the published one.
    """

    def test_identical_features_pass(self):
        rng = np.random.default_rng(0)
        feats = rng.normal(size=(50, 768)).astype(np.float32)
        report = pv.compare_to_bank(feats, feats.copy(), tolerance=1e-4)
        assert report["reproduces"]
        assert report["max_abs_diff"] == 0.0

    def test_float_noise_within_tolerance_passes(self):
        rng = np.random.default_rng(1)
        feats = rng.normal(size=(20, 768)).astype(np.float32)
        noisy = feats + rng.normal(scale=1e-6, size=feats.shape).astype(np.float32)
        assert pv.compare_to_bank(feats, noisy, tolerance=1e-4)["reproduces"]

    def test_a_real_pipeline_drift_fails(self):
        """Nearest-vs-bilinear moves features far more than float noise."""
        rng = np.random.default_rng(2)
        feats = rng.normal(size=(20, 768)).astype(np.float32)
        drifted = feats.copy()
        drifted[3, 10] += 0.05
        report = pv.compare_to_bank(feats, drifted, tolerance=1e-4)
        assert not report["reproduces"]
        assert report["n_rows_over_tolerance"] == 1
        assert report["worst_row"] == 3

    def test_shape_mismatch_is_an_error_not_a_failed_comparison(self):
        with pytest.raises(ValueError, match="shape"):
            pv.compare_to_bank(np.zeros((10, 768)), np.zeros((10, 384)), tolerance=1e-4)

    def test_report_is_json_ready(self):
        import json

        feats = np.zeros((4, 8), dtype=np.float32)
        json.dumps(pv.compare_to_bank(feats, feats, tolerance=1e-4))


class TestConditionZeroIgnoresTheMask:
    """(0) is the no-mask baseline: its output must not depend on the mask.

    If it ever did, the baseline column of the table would silently vary with
    segmentation quality and every comparison against it would be meaningless.
    """

    def test_different_masks_give_identical_output(self):
        image = np.tile(
            np.linspace(0, 255, 64, dtype=np.uint8), (64, 1)
        )[:, :, None].repeat(3, axis=2)
        full = _tracked(np.ones((64, 64), dtype=bool))
        empty = _tracked(np.zeros((64, 64), dtype=bool))
        a = ex.build_eval_input(image, full, "none")
        b = ex.build_eval_input(image, empty, "none")
        assert np.array_equal(a.image, b.image)


# ── resize semantics must match torchvision exactly ───────────────────────────


class TestResizeSemanticsMatchTorchvision:
    """`Resize(255)` with an INT resizes the SHORTER side, preserving aspect.

    My first implementation produced a 255x255 square. That distorts a 600x450
    dermatoscopic image and changes which region CenterCrop(224) keeps, so
    condition (0) could never have reproduced the bank. Caught before spending
    GPU, by reading what the bank actually does rather than assuming.
    """

    def test_shorter_side_becomes_the_target_aspect_preserved(self):
        image = np.zeros((450, 600, 3), dtype=np.uint8)  # h=450, w=600
        out = ex._resize_shorter_side(image, 255)
        assert out.shape[:2] == (255, 340), "expected h=255 (short side), w=340"

    def test_portrait_image_resizes_on_its_short_side_too(self):
        image = np.zeros((600, 450, 3), dtype=np.uint8)  # h=600, w=450
        out = ex._resize_shorter_side(image, 255)
        assert out.shape[:2] == (340, 255)

    def test_square_stays_square(self):
        out = ex._resize_shorter_side(np.zeros((500, 500, 3), dtype=np.uint8), 255)
        assert out.shape[:2] == (255, 255)

    def test_mask_resize_preserves_aspect_and_stays_binary(self):
        mask = np.zeros((450, 600), dtype=bool)
        mask[100:200, 100:200] = True
        out = ex._resize_shorter_side(mask, 255)
        assert out.shape == (255, 340)
        assert out.dtype == np.bool_

    def test_baseline_geometry_ends_at_224_square(self):
        image = np.zeros((450, 600, 3), dtype=np.uint8)
        assert ex._baseline_geometry_image(image).shape == (224, 224, 3)

    def test_image_and_mask_baseline_geometry_agree_on_a_non_square_image(self):
        image = np.zeros((450, 600, 3), dtype=np.uint8)
        mask = np.zeros((450, 600), dtype=bool)
        mask[200:300, 250:350] = True
        assert (
            ex._baseline_geometry_image(image).shape[:2]
            == ex._baseline_geometry_mask(mask).shape[:2]
        )


class TestGeometryMatchesTorchvisionExactly:
    """Direct equality against the real transform. Skips without torchvision.

    The unit tests above pin the SEMANTICS (short side, aspect, binary masks);
    this pins the PIXELS. It is the local half of the bank-reproduction gate:
    if this passes and the extraction point is right, condition (0) reproduces
    the cache. Runs on the pod, where torchvision exists.
    """

    def _reference(self, image):
        torchvision = pytest.importorskip("torchvision", reason="needs torchvision")
        from PIL import Image

        transforms = torchvision.transforms
        pipeline = transforms.Compose(
            [
                transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
            ]
        )
        return np.asarray(pipeline(Image.fromarray(image)))

    @pytest.mark.parametrize("shape", [(450, 600), (600, 450), (500, 500)])
    def test_baseline_geometry_is_pixel_identical(self, shape):
        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (*shape, 3), dtype=np.uint8)
        mine = ex._baseline_geometry_image(image)
        assert np.array_equal(mine, self._reference(image))


class TestComparingAgainstThePublishedCache:
    """The SECONDARY check: does the recomputed bank match the stored cache?

    Kept apart from the primary gate on purpose. The cache was written with
    ``torch.amp.autocast("cuda")`` and the verification may not run under the
    same precision, so a residue of ~1e-2 is expected and is NOT evidence that
    C1's preprocessing is wrong. Measured on the pod 2026-09-26: recomputing
    the bank's own pipeline and comparing against its cache gave
    max_abs_diff 1.64e-02 while the cosine stayed at 1.000 -- the same vector
    at a different precision. Gating C1 on 1e-4 against that cache made the
    gate unfailable for reasons that had nothing to do with C1.

    So this reports, and does not decide alone: a magnitude bound loose enough
    for mixed precision, plus a per-row cosine tight enough that a genuinely
    different image cannot hide under it.
    """

    def _features(self, n_rows: int = 8, n_dims: int = 768) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.normal(size=(n_rows, n_dims)).astype(np.float32) * 5.0

    def test_identical_features_agree(self):
        features = self._features()

        report = pv.compare_to_published_cache(features, features.copy())

        assert report["agrees"] is True
        assert report["max_abs_diff"] == 0.0
        assert report["min_cosine"] == pytest.approx(1.0)
        assert report["reasons"] == []

    def test_mixed_precision_residue_agrees(self):
        """float16 round-trip: what autocast actually does to the cache."""
        features = self._features()
        half = features.astype(np.float16).astype(np.float32)

        report = pv.compare_to_published_cache(features, half)

        assert report["agrees"] is True, report["reasons"]
        assert report["max_abs_diff"] > 0.0, "el escenario no produjo diferencia"
        assert report["min_cosine"] > 0.999

    def test_a_magnitude_drift_beyond_the_bound_disagrees(self):
        features = self._features()
        drifted = features.copy()
        drifted[3, 7] += 0.5

        report = pv.compare_to_published_cache(features, drifted)

        assert report["agrees"] is False
        assert report["worst_row"] == 3
        assert any("max_abs_diff" in reason for reason in report["reasons"])

    def test_a_rotated_row_disagrees_even_when_every_value_is_close(self):
        """The case a magnitude bound alone cannot see.

        Two vectors can differ by less than the absolute bound at every
        coordinate and still point somewhere else, which is exactly what a
        wrong image looks like when the features are small.
        """
        features = np.full((4, 768), 0.001, dtype=np.float32)
        rotated = features.copy()
        rotated[2] = -rotated[2]

        report = pv.compare_to_published_cache(features, rotated)

        assert report["max_abs_diff"] < pv.PUBLISHED_CACHE_MAX_ABS_DIFF
        assert report["agrees"] is False, "una fila dada vuelta paso el chequeo"
        assert report["n_rows_below_cosine"] == 1
        assert any("coseno" in reason for reason in report["reasons"])

    def test_nan_disagrees_rather_than_slipping_through(self):
        features = self._features()
        corrupted = features.copy()
        corrupted[1, 2] = np.nan

        report = pv.compare_to_published_cache(features, corrupted)

        assert report["agrees"] is False, "NaN paso el chequeo (`nan <= tol` es False)"
        assert any("nan" in reason.lower() for reason in report["reasons"])

    def test_a_nan_row_also_counts_against_the_magnitude_bound(self):
        """Both defences are pinned separately, on purpose.

        The non-finite check already catches NaN, so writing the bound as
        ``diff > tol`` instead of ``~(diff <= tol)`` passes every other test
        here. Then the day someone relaxes the non-finite check, the NaN walks
        straight through a bound that never rejected it. Each layer gets its
        own test.
        """
        features = self._features()
        corrupted = features.copy()
        corrupted[1, 2] = np.nan

        report = pv.compare_to_published_cache(features, corrupted)

        assert report["n_rows_over_max_abs_diff"] >= 1, (
            "una fila NaN no se conto contra el margen: `nan > tol` es False"
        )

    def test_a_zero_row_is_refused_instead_of_reporting_a_nan_cosine(self):
        """A zero vector has no direction; ``0/0`` would give NaN and NaN
        compares False against the bound, so the row would pass unnoticed."""
        features = self._features()
        zeroed = features.copy()
        zeroed[5] = 0.0

        report = pv.compare_to_published_cache(features, zeroed)

        assert report["agrees"] is False
        assert any("norma cero" in reason for reason in report["reasons"])

    def test_shape_mismatch_is_an_error_not_a_disagreement(self):
        with pytest.raises(ValueError, match="shape"):
            pv.compare_to_published_cache(np.zeros((4, 768)), np.zeros((4, 384)))

    @pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0])
    def test_an_unusable_bound_is_refused(self, bad):
        features = self._features()

        with pytest.raises(ValueError):
            pv.compare_to_published_cache(features, features, max_abs_diff=bad)

    @pytest.mark.parametrize("bad", [float("nan"), -0.5, 1.5])
    def test_an_unusable_cosine_floor_is_refused(self, bad):
        features = self._features()

        with pytest.raises(ValueError):
            pv.compare_to_published_cache(features, features, min_cosine=bad)

    def test_the_report_is_json_serialisable_for_wandb(self):
        import json as _json

        features = self._features()

        _json.dumps(pv.compare_to_published_cache(features, features))


class TestThePrimaryGateTolerance:
    def test_the_default_is_tight_because_it_compares_within_one_process(self):
        """The primary gate no longer compares against a stored cache, so the
        mixed-precision residue does not apply to it and the bound can be
        what an identical computation actually produces."""
        assert pv.DEFAULT_PRIMARY_TOLERANCE == 1e-5
        assert pv.DEFAULT_PRIMARY_TOLERANCE < pv.PUBLISHED_CACHE_MAX_ABS_DIFF
