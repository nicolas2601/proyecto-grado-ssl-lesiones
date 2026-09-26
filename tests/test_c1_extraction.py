"""Geometry tests for C1 extraction.

These run without torch on purpose: the part that can silently corrupt the
experiment is the geometry, not the forward pass. If the mask and the image
stop travelling through the same transform, condition (c) pools the wrong
tokens and still returns a confident number.
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
from dermssl import mask_conditions as mc


def _image(size: int = 450, value: int = 200) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _lesion(size: int = 450, box: tuple[int, int, int, int] = (100, 200, 100, 200)) -> np.ndarray:
    y0, y1, x0, x1 = box
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _tracked(mask: np.ndarray, image_id: str = "IMG_0001"):
    """Bind a mask to a step-0 manifest; extraction refuses bare arrays."""
    source = mp.MaskSource.from_manifest(
        {
            "producer": "student_r3",
            "producer_is_label_free": True,
            "masks_sha256": "a" * 64,
        "checkpoint_sha256": "f" * 64,
            "image_ids": [image_id],
            "mask_shape": list(mask.shape),
            "segmentation_run": "final",
            "segmentation_commit": "d81a492",
        }
    )
    return mp.TrackedMask(array=mask, source=source, image_id=image_id)


class TestBaselineGeometry:
    def test_resize_edge_is_255_not_256(self):
        assert int(ex.RESOLUTION * ex.RESIZE_FACTOR) == 255

    def test_grid_is_16_by_16(self):
        assert ex.GRID == 16
        assert ex.GRID * ex.GRID == 256

    @pytest.mark.parametrize("condition", ["none", "neutral_bg", "mask_pool"])
    def test_output_is_224_square(self, condition):
        prepared = ex.build_eval_input(_image(), _tracked(_lesion()), condition)
        assert prepared.image.shape == (224, 224, 3)
        assert prepared.mask.shape == (224, 224)


class TestImageAndMaskTravelTogether:
    """The load-bearing invariant of condition (c)."""

    @pytest.mark.parametrize("condition", ["none", "bbox_crop", "neutral_bg", "mask_pool"])
    def test_mask_and_image_end_up_the_same_size(self, condition):
        prepared = ex.build_eval_input(_image(), _tracked(_lesion()), condition)
        assert prepared.image.shape[:2] == prepared.mask.shape[:2]

    def test_offcentre_lesion_keeps_its_side_after_baseline_geometry(self):
        """A lesion in the top-left must not migrate to another quadrant."""
        prepared = ex.build_eval_input(_image(), _tracked(_lesion(box=(0, 180, 0, 180))), "mask_pool")
        grid = mc.mask_to_patch_grid(prepared.mask, ex.GRID)
        assert grid[:4, :4].any(), "lesion lost from the quadrant it started in"
        assert not grid[12:, 12:].any(), "lesion leaked into the opposite quadrant"

    def test_mismatched_shapes_rejected(self):
        with pytest.raises(ValueError, match="disagree"):
            ex.build_eval_input(_image(64), _tracked(_lesion(32)), "none")


class TestBboxCrop:
    def test_crop_fills_the_frame_with_lesion(self):
        """After cropping to the box, the lesion should dominate the 224 frame."""
        prepared = ex.build_eval_input(_image(), _tracked(_lesion(box=(100, 200, 100, 200))), "bbox_crop")
        assert prepared.mask.mean() > 0.5
        assert not prepared.used_bbox_fallback

    def test_empty_mask_falls_back_and_is_flagged(self):
        prepared = ex.build_eval_input(_image(), _tracked(np.zeros((450, 450), dtype=bool)), "bbox_crop"
        )
        assert prepared.used_bbox_fallback
        assert prepared.image.shape == (224, 224, 3)

    def test_crop_is_not_centre_cropped_twice(self):
        """A lesion touching the edge must survive; double-cropping would cut it."""
        prepared = ex.build_eval_input(_image(), _tracked(_lesion(box=(0, 60, 0, 60))), "bbox_crop")
        assert prepared.mask.any(), "edge lesion was cropped away"


class TestNeutralBackground:
    def test_background_is_grey_away_from_the_lesion_edge(self):
        """Grey is applied before the resize, so the edge blends -- by design.

        Bilinear resampling mixes the lesion and the grey across the boundary,
        exactly as it would for any image in the bank. Asserting that EVERY
        background pixel is pure grey would only hold under nearest, which is
        the resampling this condition must not use.
        """
        prepared = ex.build_eval_input(_image(value=200), _tracked(_lesion()), "neutral_bg")
        grey = int(round(mc.IMAGENET_MEAN[0] * 255))
        corner = prepared.image[:8, :8, 0].astype(int)
        assert np.abs(corner - grey).max() <= 1

    def test_lesion_interior_keeps_its_own_pixels(self):
        prepared = ex.build_eval_input(_image(value=200), _tracked(_lesion()), "neutral_bg")
        assert prepared.image[prepared.mask][:, 0].max() >= 199

    def test_none_condition_leaves_pixels_alone(self):
        prepared = ex.build_eval_input(_image(value=200), _tracked(_lesion()), "none")
        assert prepared.image.min() == 200


class TestEveryC1ModuleCanBeImportedFirst:
    """No module may depend on another having been imported before it.

    The C1 package was split into five modules that import each other, and for
    a while the suite only passed because of the order the test files happened
    to import them in: importing ``c1_provenance`` first raised, importing
    ``mask_conditions`` first did not. Seventy-six tests were green on luck.

    A fresh interpreter per module is the only way to see it -- inside one
    process the first import wins and every later one is a cache hit.
    """

    MODULES = (
        "c1_extraction",
        "c1_mask_export",
        "c1_mask_provenance",
        "c1_provenance",
        "image_contract",
        "mask_conditions",
    )

    @pytest.mark.parametrize("module", MODULES)
    def test_it_imports_in_a_clean_interpreter(self, module):
        import subprocess
        import sys
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "src"
        result = subprocess.run(
            [sys.executable, "-c", f"import dermssl.{module}"],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, (
            f"dermssl.{module} cannot be imported first:\n{result.stderr}"
        )
