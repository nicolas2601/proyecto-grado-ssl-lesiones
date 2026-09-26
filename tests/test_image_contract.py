"""Tests for the unit-range image contract.

Split into its own module so the fix it guards ships with its own tests. The
defect it prevents produces no exception: a U-Net trained on to_tensor output
([0, 1]) fed a raw uint8 array in [0, 255] saturates and returns masks that
look like masks. Two of three callers divided by 255; one did not, and nothing
noticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import image_contract as ic


class TestUnitFloatImageContract:
    def test_accepts_float_in_unit_range(self):
        ic.assert_unit_float_image(np.zeros((8, 8, 3), dtype=np.float32))

    def test_rejects_uint8(self):
        """The exact shape of the C1 bug: uint8 0..255 into a [0,1] network."""
        with pytest.raises(ValueError, match="dtype"):
            ic.assert_unit_float_image(np.zeros((8, 8, 3), dtype=np.uint8))

    def test_rejects_float_above_one(self):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[0, 0, 0] = 255.0
        with pytest.raises(ValueError, match="range|0, 1"):
            ic.assert_unit_float_image(image)

    def test_rejects_negative_values(self):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[0, 0, 0] = -0.5
        with pytest.raises(ValueError, match="range|0, 1"):
            ic.assert_unit_float_image(image)

    def test_error_names_the_fix(self):
        """A message that says 'divide by 255' saves the next person the trace."""
        with pytest.raises(ValueError, match="255"):
            ic.assert_unit_float_image(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_all_zero_float_image_is_valid(self):
        ic.assert_unit_float_image(np.zeros((4, 4, 3), dtype=np.float64))


class TestImageContractShape:
    """A [0,1] float check is not enough: shape and finiteness matter too."""

    def test_rejects_grayscale_hw(self):
        with pytest.raises(ValueError, match="shape|ndim|channel"):
            ic.assert_unit_float_image(np.zeros((8, 8), dtype=np.float32))

    def test_rejects_rgba_four_channels(self):
        with pytest.raises(ValueError, match="channel|shape"):
            ic.assert_unit_float_image(np.zeros((8, 8, 4), dtype=np.float32))

    def test_rejects_nan(self):
        """NaN passes a min/max range check because every comparison is False."""
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="finite|NaN"):
            ic.assert_unit_float_image(image)

    def test_rejects_inf(self):
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="finite|inf"):
            ic.assert_unit_float_image(image)

    def test_black_image_is_valid(self):
        """All-zero is a legitimate, if odd, input -- the placeholder path uses it."""
        ic.assert_unit_float_image(np.zeros((8, 8, 3), dtype=np.float32))

    def test_white_image_is_valid(self):
        ic.assert_unit_float_image(np.ones((8, 8, 3), dtype=np.float32))


class TestEmptyImageRejected:
    @pytest.mark.parametrize("shape", [(0, 8, 3), (8, 0, 3), (0, 0, 3)])
    def test_zero_sized_image_is_refused(self, shape):
        with pytest.raises(ValueError, match="empty|vac|size"):
            ic.assert_unit_float_image(np.zeros(shape, dtype=np.float32))
