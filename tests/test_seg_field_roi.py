"""Pure tests for the dermoscope field of view (valid ROI) detector."""

import numpy as np
import pytest

import seg_paths  # noqa: F401
import field_roi


def _vignetted(size=64, radius=26, inside=200, outside=10):
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    circle = (yy - center) ** 2 + (xx - center) ** 2 <= radius**2
    image = np.full((size, size, 3), outside, dtype=np.uint8)
    image[circle] = inside
    return image


def test_bright_corners_mean_no_vignette_and_keep_the_full_frame():
    valid, stats = field_roi.field_validity_mask(np.full((32, 32, 3), 180, dtype=np.uint8))
    assert valid.dtype == np.bool_
    assert stats["method"] == "full_frame"
    assert stats["field_fraction"] == 1.0
    assert valid.all()


def test_grey_vignette_ring_with_dark_corners_still_fits_the_circle():
    image = _vignetted(outside=40)  # grey ring passes the luminance threshold
    valid, stats = field_roi.field_validity_mask(image)
    assert stats["method"] == "circular"
    assert not valid[0, 0]


def test_one_bright_corner_is_not_enough_to_call_it_a_full_frame():
    image = _vignetted()
    image[:6, :6] = 200
    valid, stats = field_roi.field_validity_mask(image)
    assert stats["method"] == "circular"


def test_vignetted_image_drops_the_corners_and_keeps_the_centre():
    valid, stats = field_roi.field_validity_mask(_vignetted())
    assert stats["method"] == "circular"
    assert valid[32, 32]
    assert not valid[0, 0] and not valid[63, 63]
    assert 0.3 < stats["valid_fraction"] < 0.8


def test_all_black_image_falls_back_to_the_full_frame():
    valid, stats = field_roi.field_validity_mask(np.zeros((16, 16, 3), dtype=np.uint8))
    assert valid.all()
    assert stats["method"] == "fallback_full"


def test_a_single_bright_pixel_falls_back_instead_of_erasing_the_image():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[8, 8] = 255
    valid, stats = field_roi.field_validity_mask(image)
    assert valid.all() and stats["method"] == "fallback_full"


def test_margin_fraction_shrinks_the_valid_circle():
    image = _vignetted()
    wide, _ = field_roi.field_validity_mask(image, margin_fraction=0.0)
    narrow, _ = field_roi.field_validity_mask(image, margin_fraction=0.3)
    assert narrow.sum() < wide.sum()
    assert np.all(wide[narrow])


def test_a_grey_ring_is_below_the_adaptive_threshold_and_leaves_the_field():
    valid, stats = field_roi.field_validity_mask(_vignetted(outside=60), margin_fraction=0.10)
    assert stats["field_fraction"] < 0.6  # the ring no longer counts as field
    assert not valid[0, 0]
    assert stats["valid_fraction"] < 0.7


def test_valid_mask_has_no_holes_inside_the_circle():
    image = _vignetted()
    image[30:34, 30:34] = 0  # a very dark lesion core inside the bright field
    valid, _ = field_roi.field_validity_mask(image)
    assert valid[30:34, 30:34].all()


def test_black_threshold_out_of_range_raises():
    with pytest.raises(ValueError):
        field_roi.field_validity_mask(np.zeros((4, 4, 3), np.uint8), black_threshold=1.5)


def test_margin_fraction_out_of_range_raises():
    with pytest.raises(ValueError):
        field_roi.field_validity_mask(np.zeros((4, 4, 3), np.uint8), margin_fraction=1.0)


def test_compute_field_cache_fills_the_output_and_reports_stats():
    images = np.stack(
        [_vignetted(), np.full((64, 64, 3), 200, np.uint8), np.zeros((64, 64, 3), np.uint8)]
    )
    out = np.zeros((3, 64, 64), dtype=np.uint8)
    stats = field_roi.compute_field_cache(images, out, chunk=2)
    assert stats["images"] == 3
    assert stats["fallback_count"] == 1
    assert stats["circular_count"] == 1
    assert stats["full_frame_count"] == 1
    assert set(np.unique(out)) == {0, 1}
    assert out[1].all() and out[2].all()
    assert not out[0, 0, 0]
    assert 0.0 < stats["mean_valid_fraction"] <= 1.0
    assert stats["black_threshold"] == field_roi.DEFAULT_BLACK_THRESHOLD
    assert stats["margin_fraction"] == field_roi.DEFAULT_MARGIN_FRACTION


def _uniform(size=64, rgb=(78, 52, 44)):
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def _vignetted_rgb(size=64, radius=26, inside=(78, 52, 44), outside=(0, 0, 0)):
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    circle = (yy - center) ** 2 + (xx - center) ** 2 <= radius**2
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = outside
    image[circle] = inside
    return image


def test_a_dark_skin_full_frame_is_not_mistaken_for_a_vignette():
    """Fitzpatrick VI skin is darker than the old fixed corner threshold."""
    valid, stats = field_roi.field_validity_mask(_uniform())
    assert stats["method"] == "full_frame"
    assert valid.all()


def test_a_dark_skin_field_inside_a_black_ring_still_fits_the_circle():
    valid, stats = field_roi.field_validity_mask(_vignetted_rgb())
    assert stats["method"] == "circular"
    assert valid[32, 32]
    assert not valid[0, 0]


def test_the_field_threshold_adapts_to_the_brightness_of_the_centre():
    """A grey ring at 26/255 clears the fixed floor but not half the centre."""
    image = _vignetted_rgb(inside=(200, 200, 200), outside=(26, 26, 26))
    valid, stats = field_roi.field_validity_mask(image)
    assert stats["threshold"] > field_roi.DEFAULT_BLACK_THRESHOLD
    assert stats["method"] == "circular"
    assert stats["radius"] < 32.0
    assert not valid[0, 0] and valid[32, 32]


def test_the_threshold_never_exceeds_its_cap_or_drops_below_the_floor():
    bright = field_roi.field_validity_mask(_vignetted_rgb(inside=(255, 255, 255)))[1]
    dim = field_roi.field_validity_mask(_vignetted_rgb(inside=(4, 4, 4)))[1]
    assert bright["threshold"] <= field_roi.MAX_ADAPTIVE_THRESHOLD
    assert dim["threshold"] == field_roi.DEFAULT_BLACK_THRESHOLD


def test_the_threshold_floor_can_be_raised_by_the_caller():
    stats = field_roi.field_validity_mask(_vignetted_rgb(inside=(4, 4, 4)), black_threshold=0.5)[1]
    assert stats["threshold"] == 0.5


def test_a_single_channel_image_is_rejected():
    with pytest.raises(ValueError, match="H, W, 3"):
        field_roi.field_validity_mask(np.zeros((8, 8), dtype=np.uint8))


def test_an_image_with_an_alpha_channel_is_rejected():
    with pytest.raises(ValueError, match="H, W, 3"):
        field_roi.field_validity_mask(np.zeros((8, 8, 4), dtype=np.uint8))


def test_compute_field_cache_rejects_a_cache_without_three_channels():
    out = np.zeros((1, 8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="H, W, 3"):
        field_roi.compute_field_cache(np.zeros((1, 8, 8), dtype=np.uint8), out)


def _soft_falloff(base, size=64, corner_level=0.45):
    """Full frame with optical vignetting: corners keep ``corner_level`` of the centre."""
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2) / (center * np.sqrt(2))
    factor = 1.0 - (1.0 - corner_level) * radius
    image = np.asarray(base, dtype=np.float64)[None, None, :] * factor[..., None]
    return np.clip(image, 0, 255).astype(np.uint8)


def test_light_skin_with_soft_optical_vignetting_is_a_full_frame():
    valid, stats = field_roi.field_validity_mask(_soft_falloff((230, 170, 150)))
    assert stats["method"] == "full_frame"
    assert valid.all()


def test_dark_skin_with_moderate_optical_vignetting_is_a_full_frame():
    valid, stats = field_roi.field_validity_mask(_soft_falloff((78, 52, 44), corner_level=0.6))
    assert stats["method"] == "full_frame"


def test_dark_skin_with_corners_as_dark_as_a_ring_is_read_as_a_ring():
    # Known limit: at half the centre a Fitzpatrick VI corner is indistinguishable
    # from a brownish dermoscope ring, so the conservative circle wins.
    valid, stats = field_roi.field_validity_mask(_soft_falloff((78, 52, 44), corner_level=0.5))
    assert stats["method"] == "circular"


def test_inscribed_dermoscope_field_with_black_corners_stays_circular():
    image = _vignetted(size=64, radius=31, inside=200, outside=8)
    valid, stats = field_roi.field_validity_mask(image)
    assert stats["method"] == "circular"
    assert not valid[0, 0]


def test_brownish_dark_ring_is_not_read_as_skin():
    yy, xx = np.mgrid[:64, :64]
    circle = (yy - 31.5) ** 2 + (xx - 31.5) ** 2 <= 31**2
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[...] = (40, 25, 20)
    image[circle] = (210, 160, 140)
    valid, stats = field_roi.field_validity_mask(image)
    assert stats["method"] == "circular"


def test_a_single_black_corner_is_a_partial_vignette_and_keeps_the_full_frame():
    image = _soft_falloff((230, 170, 150), corner_level=0.7)
    image[-6:, -6:] = 5  # one corner cut by a dermoscope circle larger than the frame
    valid, stats = field_roi.field_validity_mask(image)
    assert stats["method"] == "full_frame"


def test_two_black_corners_are_a_vignette():
    image = _soft_falloff((230, 170, 150), corner_level=0.7)
    image[-6:, -6:] = 5
    image[:6, :6] = 5
    valid, stats = field_roi.field_validity_mask(image)
    assert stats["method"] == "circular"


def test_pale_skin_with_dim_corners_is_a_full_frame():
    # HAM10000 style: corners near a third of the centre, chroma about 0.06.
    valid, stats = field_roi.field_validity_mask(_soft_falloff((210, 180, 165), corner_level=0.35))
    assert stats["method"] == "full_frame"
