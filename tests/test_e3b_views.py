"""The two images F4 feeds a baseline equal what E3's full renderer builds (same geometry, same fallbacks)."""

import numpy as np
import pytest

from dermssl import c1_extraction as ex
from dermssl import e3_render as render
from dermssl.e3b import views


def resize_crop(image, resolution):
    return ex._center_crop(ex._resize_shorter_side(image, int(resolution * 1.14)), resolution)


def case(kind):
    rng = np.random.default_rng(3)
    image = rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)
    mask = np.zeros((90, 120), dtype=bool)
    if kind == "centre":
        mask[25:65, 35:85] = True
    elif kind == "outside_crop":  # lesion in the margin the centre crop cuts away
        mask[:4, :6] = True
    return image, mask


@pytest.mark.parametrize("kind", ["centre", "empty", "outside_crop"])
def test_pair_equals_the_e3_renderer_for_full_and_crop_view(kind):
    image, mask = case(kind)
    expected, fallbacks = render.render(image, mask, resize_crop=resize_crop)
    full, crop, fell_back = views.render_pair(image, mask, resize_crop)
    assert np.array_equal(full, expected["full"]) and np.array_equal(crop, expected["crop_view"])
    assert fell_back == ("crop_view" in fallbacks)
    assert fell_back == (kind != "centre")


def test_the_fallback_is_the_full_view_itself():
    image, mask = case("empty")
    full, crop, fell_back = views.render_pair(image, mask, resize_crop)
    assert fell_back and crop is full


def test_the_fallback_column_is_the_one_e3_stores():
    assert render.CONDITIONS[views.FALLBACK_COLUMN] == "crop_view"
