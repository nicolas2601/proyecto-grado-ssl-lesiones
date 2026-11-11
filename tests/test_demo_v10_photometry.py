from __future__ import annotations

import numpy as np
import pytest

from dermssl import fitzpatrick as fz
from demo.backend.photometry import ita_category, measure_photometry

SKIN = (224, 172, 140)
DARK_LESION = (60, 30, 20)
BLUE_CAST = (150, 150, 170)


def _lab_of(rgb):
    pixel = np.array([[rgb]], dtype=np.float64) / 255.0
    return fz.srgb_to_lab(pixel)[0, 0]


def _scene(size=160, radius=20, skin=SKIN, lesion_rgb=DARK_LESION):
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = skin
    yy, xx = np.mgrid[:size, :size]
    lesion = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= radius**2
    image[lesion] = lesion_rgb
    return image, lesion


@pytest.mark.parametrize(
    ("ita", "category"),
    [
        (56.0, "very_light"),
        (55.0, "light"),
        (45.0, "light"),
        (41.0, "intermediate"),
        (40.9, "intermediate"),
        (28.0, "tan"),
        (27.9, "tan"),
        (20.0, "tan"),
        (10.0, "brown"),
        (10.1, "tan"),
        (-10.0, "brown"),
        (-30.0, "dark"),
        (-29.9, "brown"),
        (-80.0, "dark"),
    ],
)
def test_ita_category_uses_six_class_scheme_with_half_open_bounds(ita, category):
    assert ita_category(ita) == category


def test_ita_category_is_none_without_an_angle():
    assert ita_category(None) is None


def test_skin_ring_yields_ita_matching_atan2_of_median_lab():
    image, lesion = _scene()
    lab = _lab_of(SKIN)

    result = measure_photometry(image, lesion)

    expected = float(np.degrees(np.arctan2(lab[0] - 50.0, lab[2])))
    assert result["status"] == "ok"
    assert result["reason"] is None
    assert result["region"] == "perilesional_ring"
    assert result["ita_deg"] == pytest.approx(expected, abs=0.01)
    assert result["median_L"] == pytest.approx(lab[0], abs=0.01)
    assert result["median_b"] == pytest.approx(lab[2], abs=0.01)
    assert result["ita_category"] == ita_category(result["ita_deg"])
    assert result["category_scheme"].startswith("Chardon 1991")
    assert 0.0 < result["valid_fraction"] <= 1.0
    assert result["n_px"] > 0


def test_bluish_skin_is_indeterminate_but_keeps_medians():
    image, lesion = _scene(skin=BLUE_CAST)

    result = measure_photometry(image, lesion)

    assert result["status"] == "indeterminate"
    assert result["reason"] == "ita_indeterminate"
    assert result["ita_deg"] is None
    assert result["ita_category"] is None
    assert result["median_b"] < 5.0
    assert result["median_L"] is not None


def test_missing_mask_is_unavailable_not_invented():
    image, _ = _scene()

    result = measure_photometry(image, None)

    assert result["status"] == "unavailable"
    assert result["reason"] == "no_mask"
    assert result["ita_deg"] is None
    assert result["median_L"] is None


def test_empty_mask_is_degenerate():
    image, _ = _scene()

    result = measure_photometry(image, np.zeros(image.shape[:2], dtype=bool))

    assert result["status"] == "unavailable"
    assert result["reason"] == "degenerate_mask"


def test_small_ring_abstains_with_ring_too_small():
    image, lesion = _scene(size=24, radius=5)

    result = measure_photometry(image, lesion)

    assert result["status"] == "unavailable"
    assert result["reason"] == "ring_too_small"


def test_mask_on_wrong_grid_raises_instead_of_guessing():
    image, _ = _scene()

    with pytest.raises(ValueError):
        measure_photometry(image, np.zeros((10, 10), dtype=bool))
