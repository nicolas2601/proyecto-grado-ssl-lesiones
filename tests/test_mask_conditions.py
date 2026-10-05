"""Tests for the C1 mask conditions.

The condition machinery is where the vgg19 bug can happen again: every one of
the four conditions changes WHERE the feature comes from, and two of them
change the image geometry as well. A probe fitted on one condition's train
features will happily consume another condition's val features and return a
plausible number that means nothing. Most of this file exists to make that
impossible rather than unlikely.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import c1_provenance as pv
from dermssl import mask_conditions as mc


def _lesion_square(size: int = 32, box: tuple[int, int, int, int] = (8, 24, 8, 24)) -> np.ndarray:
    y0, y1, x0, x1 = box
    raw = np.zeros((size, size), dtype=np.uint8)
    raw[y0:y1, x0:x1] = mc.LESION_LABEL
    return raw


class TestLesionMask:
    def test_label_one_is_lesion(self):
        raw = _lesion_square()
        assert mc.lesion_mask(raw)[10, 10]

    def test_uncertain_255_is_not_lesion(self):
        """The decision, made explicit: 255 means 'unsure', and unsure is not lesion.

        Counting it as lesion would let the uncertainty band leak into the crop
        and the pooling, inflating the foreground exactly where the segmenter
        was least confident.
        """
        raw = _lesion_square()
        raw[0:4, 0:4] = mc.UNCERTAIN_LABEL
        mask = mc.lesion_mask(raw)
        assert not mask[0:4, 0:4].any()

    def test_background_zero_is_not_lesion(self):
        assert not mc.lesion_mask(_lesion_square())[0, 0]


class TestBoundingBox:
    def test_tight_box_matches_lesion_extent(self):
        mask = mc.lesion_mask(_lesion_square(box=(8, 24, 6, 20)))
        assert mc.bbox_with_margin(mask, margin_frac=0.0) == (8, 24, 6, 20)

    def test_margin_expands_and_clamps_to_image(self):
        mask = mc.lesion_mask(_lesion_square(size=32, box=(0, 4, 0, 4)))
        y0, y1, x0, x1 = mc.bbox_with_margin(mask, margin_frac=0.5)
        assert (y0, x0) == (0, 0), "margin must not produce negative coordinates"
        assert y1 <= 32 and x1 <= 32, "margin must not exceed the image"

    def test_empty_mask_returns_none(self):
        assert mc.bbox_with_margin(np.zeros((16, 16), dtype=bool), margin_frac=0.1) is None


class TestNeutralBackground:
    def test_outside_mask_becomes_imagenet_mean(self):
        image = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask = mc.lesion_mask(_lesion_square())
        out = mc.apply_neutral_background(image, mask)
        expected = np.round(np.array(mc.IMAGENET_MEAN) * 255).astype(np.uint8)
        assert np.array_equal(out[0, 0], expected)

    def test_inside_mask_is_untouched(self):
        image = np.full((32, 32, 3), 200, dtype=np.uint8)
        mask = mc.lesion_mask(_lesion_square())
        out = mc.apply_neutral_background(image, mask)
        assert np.array_equal(out[10, 10], image[10, 10])

    def test_neutralised_pixels_land_near_zero_after_normalize(self):
        """The whole point of grey: after Normalize the background contributes ~0."""
        image = np.full((8, 8, 3), 200, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        out = mc.apply_neutral_background(image, mask).astype(np.float32) / 255.0
        normalised = (out - np.array(mc.IMAGENET_MEAN)) / np.array(mc.IMAGENET_STD)
        assert np.abs(normalised).max() < 0.01


class TestPatchGrid:
    def test_grid_shape_matches_request(self):
        mask = mc.lesion_mask(_lesion_square(size=224, box=(0, 112, 0, 112)))
        assert mc.mask_to_patch_grid(mask, grid=16).shape == (16, 16)

    def test_quadrant_mask_activates_that_quadrant_only(self):
        mask = mc.lesion_mask(_lesion_square(size=224, box=(0, 112, 0, 112)))
        grid = mc.mask_to_patch_grid(mask, grid=16)
        assert grid[:8, :8].all()
        assert not grid[8:, 8:].any()

    def test_downsampling_never_invents_foreground(self):
        grid = mc.mask_to_patch_grid(np.zeros((224, 224), dtype=bool), grid=16)
        assert not grid.any()


class TestMaskPool:
    def test_averages_only_patches_inside_the_mask(self):
        tokens = np.zeros((4, 3), dtype=np.float32)
        tokens[0] = 10.0
        tokens[3] = 2.0
        grid = np.array([[True, False], [False, True]])
        pooled, fallback = mc.mask_pool(tokens, grid, cls_token=np.full(3, -99.0, dtype=np.float32))
        assert not fallback
        assert np.allclose(pooled, 6.0)

    def test_falls_back_to_cls_when_no_patch_is_inside(self):
        """An empty grid must be visible, not silently averaged into a zero vector."""
        tokens = np.ones((4, 3), dtype=np.float32)
        cls = np.array([7.0, 7.0, 7.0], dtype=np.float32)
        pooled, fallback = mc.mask_pool(tokens, np.zeros((2, 2), dtype=bool), cls_token=cls)
        assert fallback
        assert np.allclose(pooled, cls)

    def test_rejects_token_count_that_does_not_match_the_grid(self):
        with pytest.raises(ValueError, match="grid"):
            mc.mask_pool(np.ones((5, 3)), np.ones((2, 2), dtype=bool), cls_token=np.ones(3))


class TestConditionPairingGuard:
    """The vgg19 lesson: a probe must refuse features it was not fitted on."""

    def _features(
        self,
        condition: str,
        backbone: str = "dinov2_public_vitb14",
        seed: int = 0,
        split: str = "train",
        prefix: str = "tr",
    ):
        rng = np.random.default_rng(seed)
        return pv.ConditionedFeatures(
            condition=condition,
            backbone=backbone,
            split=split,
            features=rng.normal(size=(24, 8)).astype(np.float32),
            labels=np.tile(np.arange(4), 6),
            image_ids=tuple(f"{prefix}_{i}" for i in range(24)),
            masks_sha256="a" * 64,
        )

    def test_matching_condition_and_backbone_is_accepted(self):
        # Only the happy path actually reaches the probe; the guards below raise
        # before sklearn is touched, so they still run everywhere.
        pytest.importorskip("sklearn", reason="logreg probe needs scikit-learn")
        train = self._features("mask_pool")
        val = pv.ConditionedFeatures(
            condition="mask_pool",
            backbone="dinov2_public_vitb14",
            split="val",
            features=np.random.default_rng(1).normal(size=(10, 8)).astype(np.float32),
            labels=np.tile(np.arange(2), 5),
            image_ids=tuple(f"va_{i}" for i in range(10)),
            masks_sha256="a" * 64,
        )
        proba = pv.fit_and_score(train, val, n_classes=4)
        assert proba.shape == (10, 4)

    def test_mixing_conditions_raises(self):
        train = self._features("mask_pool")
        val = self._features("neutral_bg", seed=1, split="val", prefix="va")
        with pytest.raises(ValueError, match="condition"):
            pv.fit_and_score(train, val, n_classes=4)

    def test_mixing_backbones_raises(self):
        train = self._features("mask_pool", backbone="dinov2_public_vitb14")
        val = self._features("mask_pool", backbone="ssl_proposed_vitb14", seed=1, split="val", prefix="va")
        with pytest.raises(ValueError, match="backbone"):
            pv.fit_and_score(train, val, n_classes=4)

    def test_mismatched_feature_dim_raises(self):
        train = self._features("mask_pool")
        val = pv.ConditionedFeatures(
            condition="mask_pool",
            backbone="dinov2_public_vitb14",
            split="val",
            features=np.zeros((10, 7), dtype=np.float32),
            labels=np.zeros(10, dtype=int),
            image_ids=tuple(f"va_{i}" for i in range(10)),
            masks_sha256="a" * 64,
        )
        with pytest.raises(ValueError, match="dim"):
            pv.fit_and_score(train, val, n_classes=4)


class TestConditionRegistry:
    def test_exactly_four_conditions(self):
        assert mc.CONDITIONS == ("none", "bbox_crop", "neutral_bg", "mask_pool")

    def test_only_mask_pool_uses_patch_tokens(self):
        uses = {c: mc.uses_patch_tokens(c) for c in mc.CONDITIONS}
        assert uses == {
            "none": False,
            "bbox_crop": False,
            "neutral_bg": False,
            "mask_pool": True,
        }

    def test_unknown_condition_rejected(self):
        with pytest.raises(ValueError, match="condition"):
            mc.uses_patch_tokens("segment_anything")
