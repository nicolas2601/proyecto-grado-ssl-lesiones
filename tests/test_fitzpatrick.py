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


def test_ita_degrees_matches_the_groh_formula():
    """ITA = arctan((L*-50)/b*) * 180/pi, computed elementwise."""
    L = np.array([70.0, 30.0])
    b = np.array([20.0, -10.0])

    ita = fp.ita_degrees(L, b)

    expected = np.degrees(np.arctan2(L - 50.0, b))
    np.testing.assert_allclose(ita, expected)


@pytest.mark.parametrize(
    "ita, expected_bucket",
    [
        (60.0, 1),  # well above the lightest threshold
        (55.0001, 1),  # just above 55
        (55.0, 2),  # exactly on the 55 boundary falls to the next bucket
        (50.0, 2),
        (41.0, 3),  # exactly on the 41 boundary
        (35.0, 3),
        (28.0, 4),  # exactly on the 28 boundary
        (24.0, 4),
        (19.0, 5),  # exactly on the 19 boundary
        (15.0, 5),
        (10.0, 6),  # exactly on the 10 boundary
        (-30.0, 6),  # well below the darkest threshold
    ],
)
def test_fst_bucket_at_groh_thresholds_and_boundaries(ita, expected_bucket):
    assert fp.fst_bucket(ita) == expected_bucket
