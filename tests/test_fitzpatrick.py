"""FST estimation: sRGB->LAB conversion, ITA/bucket mapping, and the
mask-guided estimator. Pure numpy in production; scikit-image is a test
oracle only (D2)."""

import numpy as np
import pytest

from dermssl import fitzpatrick as fp


def random_srgb(rng, shape=(16, 16, 3)):
    return rng.uniform(0.0, 1.0, size=shape)


def test_srgb_to_lab_matches_skimage_reference():
    skimage_color = pytest.importorskip("skimage.color")
    rng = np.random.default_rng(0)
    img = random_srgb(rng)

    lab = fp.srgb_to_lab(img)
    expected = skimage_color.rgb2lab(img)

    assert lab.shape == expected.shape
    np.testing.assert_allclose(lab, expected, atol=1e-4)


def test_srgb_to_lab_matches_skimage_on_boundary_colors():
    """Black/white/near-black exercise the piecewise gamma and Lab-f()
    thresholds (different branch than the mid-range random swatch above)."""
    skimage_color = pytest.importorskip("skimage.color")
    swatches = np.array(
        [
            [0.0, 0.0, 0.0],  # black: exercises the linear gamma branch
            [1.0, 1.0, 1.0],  # white: near the top of both piecewise fns
            [0.01, 0.01, 0.01],  # just below the sRGB gamma cutover (0.04045)
            [1.0, 0.0, 0.0],  # saturated red
        ]
    ).reshape(1, 4, 3)

    lab = fp.srgb_to_lab(swatches)
    expected = skimage_color.rgb2lab(swatches)

    np.testing.assert_allclose(lab, expected, atol=1e-4)
    # Black must land at the Lab origin - a concrete, non-trivial value.
    np.testing.assert_allclose(lab[0, 0], [0.0, 0.0, 0.0], atol=1e-6)
