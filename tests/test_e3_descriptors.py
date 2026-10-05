"""Shape and colour descriptors of the mask (E3 F4) on synthetic lesions."""

import numpy as np
import pytest

from dermssl import e3_descriptors as d

SKIN = np.array([200, 150, 130], dtype=np.uint8)
DARK = np.array([60, 30, 20], dtype=np.uint8)


def scene(mask, lesion=DARK, skin=SKIN):
    image = np.empty(mask.shape + (3,), dtype=np.uint8)
    image[:] = skin
    image[mask] = lesion
    return image


def rect(y0=40, y1=60, x0=50, x1=80, shape=(100, 120)):
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def disc(r=20, shape=(100, 120)):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (yy - 50) ** 2 + (xx - 60) ** 2 <= r * r


def comb():
    m = np.zeros((100, 120), dtype=bool)
    m[40:60, 40:80] = True
    m[40:48, 45:50] = False
    m[40:48, 55:60] = False
    m[40:48, 65:70] = False
    return m


def named(mask, image=None):
    vector = d.descriptors(scene(mask) if image is None else image, mask)
    return dict(zip(d.DESCRIPTOR_NAMES, vector))


def test_there_are_seven_named_descriptors():
    assert d.DESCRIPTOR_NAMES == ("asymmetry", "border_irregularity", "lab_std_l", "lab_std_a",
                                  "lab_std_b", "relative_diameter", "contrast_de")
    assert d.descriptors(scene(rect()), rect()).shape == (7,)


def test_a_rectangle_is_symmetric_about_its_own_centroid():
    assert named(rect())["asymmetry"] == pytest.approx(0.0, abs=1e-9)


def test_a_half_disc_is_asymmetric_and_a_disc_much_less():
    half = disc() & (np.arange(120)[None, :] < 60)
    assert named(half)["asymmetry"] > named(disc())["asymmetry"] + 0.05


def test_a_notched_border_is_more_irregular_than_a_disc_or_a_rectangle():
    assert named(comb())["border_irregularity"] > named(disc())["border_irregularity"]
    assert named(comb())["border_irregularity"] > named(rect())["border_irregularity"]


def test_a_uniform_lesion_has_no_colour_variegation_and_a_two_tone_one_has():
    mask = rect()
    uniform = named(mask)
    assert uniform["lab_std_l"] == pytest.approx(0.0, abs=1e-9)
    image = scene(mask)
    image[40:50, 50:80] = [120, 60, 40]
    assert named(mask, image)["lab_std_l"] > 5.0


def test_relative_diameter_is_the_equivalent_diameter_over_the_short_side():
    assert named(rect())["relative_diameter"] == pytest.approx(np.sqrt(4 * 600 / np.pi) / 100)


def test_contrast_is_zero_without_a_colour_difference_and_grows_with_it():
    mask = rect()
    flat = named(mask, scene(mask, lesion=SKIN))["contrast_de"]
    assert flat == pytest.approx(0.0, abs=1e-6)
    assert named(mask)["contrast_de"] > 20.0


def test_contrast_reads_only_the_pixels_of_the_skin_mask_when_given():
    mask = rect()
    image = scene(mask)
    image[:, :20] = [0, 255, 0]  # far from the lesion: must not enter the default ring
    default = d.descriptors(image, mask)[-1]
    assert default == pytest.approx(d.descriptors(scene(mask), mask)[-1])


def test_an_empty_mask_gives_nan_everywhere():
    empty = np.zeros((20, 20), dtype=bool)
    assert np.isnan(d.descriptors(scene(empty), empty)).all()


def test_standardize_uses_the_fit_statistics_and_fills_nan_with_the_fit_mean():
    fit = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
    other = np.array([[3.0, 10.0], [np.nan, np.nan]])
    z_fit, z_other, stats = d.standardize(fit, other)
    assert z_fit[:, 0].mean() == pytest.approx(0.0)
    assert z_fit[:, 0].std() == pytest.approx(1.0)
    assert (z_fit[:, 1] == 0).all()  # constant column: zero, not a division by zero
    assert z_other[0, 0] == pytest.approx(0.0) and (z_other[1] == 0).all()
    assert stats["n_filled"] == [1, 1]
