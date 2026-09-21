"""Stage 2 CIRCLe transformer: deterministic LAB color shift, lesion-preserving,
bounded FST-shift sampling, and blend policy - all pure numpy (spec.md,
circle-adaptation domain).

This is Stage 2's MACHINERY, built ahead of the real Stage 1 Kaggle run (PR3's
manual step, still pending) that will produce a real ``fst_manifest.csv``. Every
test here checks the transform's CONTRACT (determinism, lesion preservation,
valid output range, bounded shift) - none of it measures or asserts a real
fairness improvement, because no real Stage 1 data exists yet.
"""

import numpy as np
import pytest

from dermssl import circle as c
from dermssl import mask_manifest as mm
from dermssl.fitzpatrick import (
    FST_THRESHOLDS,
    MAX_MASK_AREA_FRACTION,
    MIN_MASK_AREA_FRACTION,
    srgb_to_lab,
)
from dermssl.gates import FAIL, PASS, SKIP

N_BUCKETS = len(FST_THRESHOLDS) + 1  # 6


def _swatch(height=16, width=16, seed=0):
    """A small, reproducible sRGB image in [0, 1] with a plausible skin-tone
    palette (not pure noise - noise can produce achromatic pixels that make
    LAB round-trips numerically unstable near a*=b*=0)."""
    rng = np.random.RandomState(seed)
    base = np.array([0.75, 0.55, 0.45])  # mid skin tone, arbitrary
    img = base + rng.uniform(-0.05, 0.05, size=(height, width, 3))
    return np.clip(img, 0.05, 0.95)


def _center_square_mask(height=16, width=16, size=6):
    """A lesion mask that is a plausible single-lesion segmentation: an interior
    square, area well within [MIN_MASK_AREA_FRACTION, MAX_MASK_AREA_FRACTION]."""
    mask = np.zeros((height, width), dtype=bool)
    y0, x0 = (height - size) // 2, (width - size) // 2
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


# --------------------------------------------------------------------------
# lab_to_srgb: inverse of fitzpatrick.srgb_to_lab
# --------------------------------------------------------------------------


def test_lab_to_srgb_round_trips_srgb_to_lab():
    img = _swatch()
    lab = srgb_to_lab(img)
    back = c.lab_to_srgb(lab)
    assert np.allclose(back, img, atol=1e-3)


def test_lab_to_srgb_output_is_finite_and_in_unit_range():
    img = _swatch(seed=1)
    lab = srgb_to_lab(img)
    back = c.lab_to_srgb(lab)
    assert np.isfinite(back).all()
    assert back.min() >= 0.0
    assert back.max() <= 1.0


# --------------------------------------------------------------------------
# clamp_fst_shift / target_fst_bucket: bounded shift (spec.md req 4)
# --------------------------------------------------------------------------


def test_clamp_fst_shift_leaves_in_bound_shifts_untouched():
    assert c.clamp_fst_shift(2) == 2
    assert c.clamp_fst_shift(-2) == -2
    assert c.clamp_fst_shift(0) == 0


def test_clamp_fst_shift_clamps_to_the_max_allowed():
    assert c.clamp_fst_shift(3) == c.MAX_FST_SHIFT
    assert c.clamp_fst_shift(-3) == -c.MAX_FST_SHIFT
    assert c.clamp_fst_shift(100) == c.MAX_FST_SHIFT


def test_target_fst_bucket_within_bound_shift():
    """spec.md scenario: source bucket 2, requested shift 2 -> target bucket 4."""
    assert c.target_fst_bucket(source_bucket=2, requested_shift=2) == 4


def test_target_fst_bucket_clamps_shift_that_exceeds_the_bound():
    """spec.md scenario: source bucket 2, requested shift 3 -> MUST clamp to the
    max allowed (2), MUST NOT silently apply the larger shift (which would be
    bucket 5)."""
    clamped = c.target_fst_bucket(source_bucket=2, requested_shift=3)
    unclamped_would_be = 2 + 3
    assert clamped == c.target_fst_bucket(source_bucket=2, requested_shift=2) == 4
    assert clamped != unclamped_would_be


def test_target_fst_bucket_never_leaves_the_valid_range():
    assert c.target_fst_bucket(source_bucket=N_BUCKETS, requested_shift=2) == N_BUCKETS
    assert c.target_fst_bucket(source_bucket=1, requested_shift=-2) == 1


# --------------------------------------------------------------------------
# shift_skin_tone: the transformer itself (spec.md req 2)
# --------------------------------------------------------------------------


def test_shift_skin_tone_is_deterministic():
    img, mask = _swatch(), _center_square_mask()
    out1 = c.shift_skin_tone(img, mask, shift_buckets=1)
    out2 = c.shift_skin_tone(img, mask, shift_buckets=1)
    np.testing.assert_array_equal(out1, out2)


def test_shift_skin_tone_leaves_lesion_pixels_bit_exact():
    img, mask = _swatch(seed=2), _center_square_mask()
    out = c.shift_skin_tone(img, mask, shift_buckets=2)
    np.testing.assert_array_equal(out[mask], img[mask])


def test_shift_skin_tone_actually_changes_skin_pixels():
    """A no-op transform that happens to also leave skin unchanged would trivially
    pass the lesion-preservation test above without doing anything - this pins
    down that a nonzero shift really moves the skin pixels."""
    img, mask = _swatch(seed=3), _center_square_mask()
    out = c.shift_skin_tone(img, mask, shift_buckets=2)
    assert not np.allclose(out[~mask], img[~mask])


def test_shift_skin_tone_zero_shift_is_near_identity_on_skin():
    img, mask = _swatch(seed=4), _center_square_mask()
    out = c.shift_skin_tone(img, mask, shift_buckets=0)
    assert np.allclose(out[~mask], img[~mask], atol=1e-3)


def test_shift_skin_tone_raises_on_missing_mask():
    img = _swatch()
    with pytest.raises(ValueError):
        c.shift_skin_tone(img, None, shift_buckets=1)


@pytest.mark.parametrize(
    "area_fraction",
    [0.0, MIN_MASK_AREA_FRACTION / 2, MAX_MASK_AREA_FRACTION + 0.05, 1.0],
    ids=["empty", "too_small", "too_large", "whole_frame"],
)
def test_shift_skin_tone_raises_on_degenerate_mask(area_fraction):
    """spec.md: 'MUST NOT silently apply an unprotected color shift over the
    lesion' - a degenerate mask (too small/too large to be a real single-lesion
    segmentation) must raise, never silently transform the whole frame."""
    img = _swatch()
    mask = np.zeros((16, 16), dtype=bool)
    n_true = int(round(area_fraction * mask.size))
    flat = mask.reshape(-1)
    flat[:n_true] = True
    with pytest.raises(ValueError):
        c.shift_skin_tone(img, mask, shift_buckets=1)


def test_shift_skin_tone_output_is_finite_and_in_valid_rgb_range():
    img, mask = _swatch(seed=5), _center_square_mask()
    out = c.shift_skin_tone(img, mask, shift_buckets=2)
    assert np.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_shift_skin_tone_coerces_non_bool_mask():
    """Mirrors fitzpatrick.estimate_fst's own mask-dtype guard: an int/uint8
    0/1 mask must behave identically to a bool mask, not silently invert under
    bitwise negation."""
    img = _swatch(seed=6)
    bool_mask = _center_square_mask()
    int_mask = bool_mask.astype(np.uint8)
    out_bool = c.shift_skin_tone(img, bool_mask, shift_buckets=1)
    out_int = c.shift_skin_tone(img, int_mask, shift_buckets=1)
    np.testing.assert_array_equal(out_bool, out_int)


def test_traceable_mask_record_protects_lesion_pixels_bit_exact():
    img, mask = _swatch(seed=7), _center_square_mask()
    record = mm.build_mask_record(
        dataset="ham10000",
        image_id="sample",
        source_revision="v1",
        mask=mask,
        producer="otsu",
        version="1",
    )

    out = c.shift_skin_tone_from_mask_record(img, mask, record, shift_buckets=1)

    np.testing.assert_array_equal(out[mask], img[mask])


def test_traceable_mask_record_rejects_tampered_or_quarantined_masks():
    img, mask = _swatch(seed=8), _center_square_mask()
    record = mm.build_mask_record(
        dataset="ham10000",
        image_id="sample",
        source_revision="v1",
        mask=mask,
        producer="otsu",
        version="1",
    )
    tampered = mask.copy()
    tampered[0, 0] = True

    with pytest.raises(ValueError, match="procedencia"):
        c.shift_skin_tone_from_mask_record(img, tampered, record, shift_buckets=1)
    with pytest.raises(ValueError, match="elegible"):
        c.shift_skin_tone_from_mask_record(
            img,
            mask,
            {**record, "eligible": False},
            shift_buckets=1,
        )


# --------------------------------------------------------------------------
# synthetic_blend_mask: blend policy (spec.md req 4, "~50/50" default)
# --------------------------------------------------------------------------


def test_synthetic_blend_mask_has_the_requested_length():
    mask = c.synthetic_blend_mask(100, target_ratio=0.5, seed=0)
    assert mask.shape == (100,)
    assert mask.dtype == bool


def test_synthetic_blend_mask_approximates_the_target_ratio():
    mask = c.synthetic_blend_mask(1000, target_ratio=0.5, seed=0)
    assert 450 <= mask.sum() <= 550


def test_synthetic_blend_mask_is_deterministic_for_a_given_seed():
    m1 = c.synthetic_blend_mask(200, target_ratio=0.3, seed=7)
    m2 = c.synthetic_blend_mask(200, target_ratio=0.3, seed=7)
    np.testing.assert_array_equal(m1, m2)


def test_synthetic_blend_mask_zero_ratio_is_all_real():
    mask = c.synthetic_blend_mask(50, target_ratio=0.0, seed=0)
    assert mask.sum() == 0


def test_synthetic_blend_mask_full_ratio_is_all_synthetic():
    mask = c.synthetic_blend_mask(50, target_ratio=1.0, seed=0)
    assert mask.sum() == 50


def test_synthetic_blend_mask_rejects_an_out_of_range_ratio():
    with pytest.raises(ValueError):
        c.synthetic_blend_mask(10, target_ratio=1.5, seed=0)
    with pytest.raises(ValueError):
        c.synthetic_blend_mask(10, target_ratio=-0.1, seed=0)


# --------------------------------------------------------------------------
# should_run_stage2: stage-gate dependency on Stage 1 (spec.md req 1)
# --------------------------------------------------------------------------


def test_stage2_runs_only_when_stage1_fails():
    assert c.should_run_stage2(FAIL) is True


def test_stage2_is_skippable_when_stage1_passes_or_skips():
    assert c.should_run_stage2(PASS) is False
    assert c.should_run_stage2(SKIP) is False
