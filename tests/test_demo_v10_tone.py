from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from dermssl import fitzpatrick as fz
from demo.backend.tone_sim import (
    ToneSimulationError,
    check_mask_area,
    decode_mask_png,
    encode_webp_lossless,
    fit_size,
    simulate_tone,
    validate_delta,
)

SKIN = (200, 150, 120)
LESION = (70, 35, 25)


def _scene(size=96, radius=14):
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = SKIN
    yy, xx = np.mgrid[:size, :size]
    mask = (yy - size // 2) ** 2 + (xx - size // 2) ** 2 <= radius**2
    image[mask] = LESION
    return image, mask


def _lab(rgb):
    return fz.srgb_to_lab(rgb.astype(np.float64) / 255.0)


@pytest.mark.parametrize("value", [-16, 16, 0, 0.5, -16.0])
def test_delta_inside_range_is_accepted(value):
    assert validate_delta(value) == float(value)


@pytest.mark.parametrize("value", [16.01, -17, float("nan"), float("inf"), True, "8", None])
def test_delta_outside_range_or_not_a_number_is_rejected_without_clamping(value):
    with pytest.raises(ToneSimulationError) as caught:
        validate_delta(value)
    assert caught.value.code == "INVALID_DELTA"


def test_zero_delta_returns_byte_identical_pixels():
    image, mask = _scene()

    result = simulate_tone(image, mask, 0.0)

    assert result.rgb.tobytes() == image.tobytes()
    assert result.clipped_fraction == 0.0
    assert result.lesion_preserved is True


@pytest.mark.parametrize("delta", [-8.0, 8.0])
def test_lesion_pixels_are_restored_bit_exact(delta):
    image, mask = _scene()

    result = simulate_tone(image, mask, delta)

    assert np.array_equal(result.rgb[mask], image[mask])
    assert result.lesion_preserved is True


def test_positive_delta_raises_skin_lightness_and_keeps_chroma():
    image, mask = _scene()

    result = simulate_tone(image, mask, 8.0)

    before, after = _lab(image)[~mask], _lab(result.rgb)[~mask]
    assert np.median(after[:, 0] - before[:, 0]) == pytest.approx(8.0, abs=1.0)
    assert np.median(np.abs(after[:, 1] - before[:, 1])) < 1.5
    assert np.median(np.abs(after[:, 2] - before[:, 2])) < 1.5


def test_lightness_is_clipped_at_100_and_reported():
    bright = (250, 246, 240)
    image = np.full((48, 48, 3), bright, dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=bool)
    mask[20:28, 20:28] = True
    image[mask] = LESION

    result = simulate_tone(image, mask, 16.0)

    assert result.clipped_fraction > 0.9
    assert _lab(result.rgb)[~mask][:, 0].max() <= 100.5


def test_output_keeps_shape_and_uint8_dtype():
    image, mask = _scene()

    result = simulate_tone(image, mask, -4.0)

    assert result.rgb.shape == image.shape
    assert result.rgb.dtype == np.uint8


def test_fit_size_caps_longest_side_at_768_and_keeps_aspect():
    assert fit_size(1024, 768) == (768, 576)
    assert fit_size(500, 300) == (500, 300)
    assert fit_size(100, 2000) == (38, 768)


def test_mask_png_is_decoded_to_image_size_with_nearest_neighbour():
    small = np.zeros((4, 4), dtype=np.uint8)
    small[1:3, 1:3] = 255
    buffer = io.BytesIO()
    Image.fromarray(small, mode="L").save(buffer, format="PNG")

    mask = decode_mask_png(buffer.getvalue(), width=8, height=8)

    assert mask.shape == (8, 8)
    assert mask.dtype == bool
    assert mask.sum() == 16


def test_undecodable_mask_is_invalid_mask():
    with pytest.raises(ToneSimulationError) as caught:
        decode_mask_png(b"not a png", width=8, height=8)
    assert caught.value.code == "INVALID_MASK"


@pytest.mark.parametrize("area", [0.0, 0.005, 0.95])
def test_degenerate_mask_area_is_rejected(area):
    mask = np.zeros((100, 100), dtype=bool)
    mask.flat[: int(area * 10000)] = True

    with pytest.raises(ToneSimulationError) as caught:
        check_mask_area(mask)
    assert caught.value.code == "INVALID_MASK"


def test_plausible_mask_area_passes():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:20, :20] = True

    check_mask_area(mask)


def test_webp_output_is_lossless():
    image, _ = _scene()

    payload = encode_webp_lossless(image)

    decoded = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    assert payload[:4] == b"RIFF"
    assert np.array_equal(decoded, image)
    assert base64.b64encode(payload).decode("ascii").startswith("UklGR")


def test_clipped_fraction_counts_gamut_clipped_realised_shift():
    saturated = (0, 0, 255)
    image = np.full((32, 32, 3), saturated, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:4, 0:4] = True
    image[mask] = LESION

    result = simulate_tone(image, mask, 16.0)

    assert result.clipped_fraction > 0.9


def test_neutral_skin_is_not_reported_as_clipped():
    image, mask = _scene()

    result = simulate_tone(image, mask, 8.0)

    assert result.clipped_fraction < 0.01


def _shift_field(image, result):
    before = _lab(image)[..., 0]
    after = _lab(result.rgb)[..., 0]
    return after - before


def test_luminance_shift_has_no_step_at_the_lesion_boundary():
    image, mask = _scene(size=120, radius=20)

    result = simulate_tone(image, mask, 8.0)

    shift = _shift_field(image, result)
    dx = np.abs(np.diff(shift, axis=1))
    dy = np.abs(np.diff(shift, axis=0))
    assert max(dx.max(), dy.max()) < 2.5, "hard edge: a step of the full delta at the mask boundary"


def test_far_skin_still_receives_the_full_delta():
    image, mask = _scene(size=160, radius=20)

    result = simulate_tone(image, mask, 8.0)

    shift = _shift_field(image, result)
    far = np.zeros_like(mask)
    far[:5, :] = True
    far[-5:, :] = True
    assert np.median(shift[far]) == pytest.approx(8.0, abs=1.0)


def test_protected_margin_is_untouched_by_the_full_step_and_lesion_stays_exact():
    image, mask = _scene(size=120, radius=20)

    result = simulate_tone(image, mask, -8.0)

    assert np.array_equal(result.rgb[mask], image[mask])
    assert result.lesion_preserved is True
