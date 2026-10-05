"""Color constancy as a stochastic augmentation, not a fixed preprocess."""

import numpy as np
import pytest

from dermssl.pretrain import color_constancy as cc


def tinted_image(rng=None):
    """Gray-ish image with a strong red cast, values in [0, 1]."""
    rng = rng or np.random.default_rng(0)
    img = rng.uniform(0.2, 0.6, size=(32, 32, 3))
    img[..., 0] = np.clip(img[..., 0] * 1.8, 0.0, 1.0)  # red cast
    return img


def channel_means(img):
    return img.reshape(-1, 3).mean(axis=0)


def test_gray_world_equalizes_channel_means():
    img = tinted_image()
    before = channel_means(img)
    assert before[0] > before[1] * 1.3  # the cast is real

    corrected = cc.gray_world(img)
    after = channel_means(corrected)
    assert np.allclose(after, after.mean(), rtol=0.05)


def test_shades_of_gray_with_p1_is_gray_world():
    img = tinted_image()
    assert np.allclose(cc.shades_of_gray(img, p=1), cc.gray_world(img), atol=1e-6)


def test_shades_of_gray_default_p6_reduces_the_cast():
    img = tinted_image()
    corrected = cc.shades_of_gray(img)
    before, after = channel_means(img), channel_means(corrected)
    assert (after[0] - after[1]) < (before[0] - before[1])


def test_output_stays_in_unit_range():
    img = tinted_image()
    for op in (cc.gray_world, cc.shades_of_gray):
        out = op(img)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_black_image_does_not_divide_by_zero():
    img = np.zeros((8, 8, 3))
    assert np.allclose(cc.gray_world(img), 0.0)
    assert np.allclose(cc.shades_of_gray(img), 0.0)


def test_sample_op_normalizes_probs_before_choice():
    """np.random choice demands probs summing to EXACTLY 1 (v5 gotcha #5)."""
    rng = np.random.default_rng(0)
    probs = (("identity", 0.65), ("gray_world", 0.15), ("shades_of_gray", 0.19))
    # sums to 0.99 - must not raise
    names = {cc.sample_op(rng, probs) for _ in range(200)}
    assert names == {"identity", "gray_world", "shades_of_gray"}


def test_sample_op_respects_zero_probability():
    rng = np.random.default_rng(0)
    probs = (("identity", 1.0), ("gray_world", 0.0), ("shades_of_gray", 0.0))
    assert all(cc.sample_op(rng, probs) == "identity" for _ in range(50))


def test_unknown_op_name_is_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="desconocida"):
        cc.sample_op(rng, (("sepia", 1.0),))


def test_apply_stochastic_draws_independently_per_call():
    """Two views of the same image must be able to receive different ops -
    that independence is exactly what makes it an invariance signal."""
    rng = np.random.default_rng(7)
    img = tinted_image()
    probs = (("identity", 0.5), ("gray_world", 0.5), ("shades_of_gray", 0.0))
    outs = [cc.apply_stochastic(img, rng, probs) for _ in range(30)]
    changed = [not np.allclose(o, img) for o in outs]
    assert any(changed) and not all(changed)  # both ops appeared


# --- docs/31 (T4): Shades of Gray as a deterministic preprocess before DINOv2 ---


def minkowski_norms(img, p):
    flat = img.reshape(-1, 3)
    return flat.max(axis=0) if np.isinf(p) else (flat**p).mean(axis=0) ** (1.0 / p)


def test_a_gray_image_is_left_unchanged():
    """Up to one ulp: the gain is mean(n)/n with three equal n, and the mean of
    three equal floats is not always bit-equal to them. Kept that way on purpose:
    the pretraining of A* and B* ran this exact arithmetic (docs/31 §2)."""
    rng = np.random.default_rng(3)
    gray = np.repeat(rng.uniform(0.1, 0.9, size=(16, 16, 1)), 3, axis=2)
    for p in (1, 6, np.inf):
        assert np.allclose(cc.illuminant_gains(gray, p), 1.0, rtol=0, atol=1e-15)
        assert np.allclose(cc.shades_of_gray(gray, p=p), gray, rtol=0, atol=1e-15)


def test_a_gray_uint8_image_moves_at_most_one_level():
    """The one-ulp gain times truncation can drop a level, exactly as in pretraining."""
    rng = np.random.default_rng(4)
    gray = np.repeat(rng.integers(0, 256, size=(16, 16, 1), dtype=np.uint8), 3, axis=2)
    diff = gray.astype(int) - cc.shades_of_gray_uint8(gray, p=6).astype(int)
    assert diff.min() >= 0 and diff.max() <= 1


def test_gains_equalise_the_channel_norms_and_keep_their_mean():
    """g_c = mean_k(n_k) / n_c: every corrected norm equals the old mean (docs/31 §2)."""
    img = tinted_image() * 0.5  # no clipping, so the identity is exact
    for p in (2, 6):
        norms = minkowski_norms(img, p)
        gains = cc.illuminant_gains(img, p)
        assert np.allclose(gains * norms, norms.mean())
        assert np.allclose(minkowski_norms(cc.shades_of_gray(img, p=p), p), norms.mean())


def test_p_infinity_is_max_rgb():
    img = tinted_image()
    expected = np.clip(img * (img.reshape(-1, 3).max(axis=0).mean() / img.reshape(-1, 3).max(axis=0)), 0, 1)
    assert np.allclose(cc.shades_of_gray(img, p=np.inf), expected)
    assert np.allclose(cc.max_rgb(img), expected)


def test_large_p_approaches_max_rgb():
    img = tinted_image()
    assert np.allclose(cc.illuminant_gains(img, 400), cc.illuminant_gains(img, np.inf), atol=2e-2)


def test_p_below_one_is_refused():
    with pytest.raises(ValueError, match="p"):
        cc.illuminant_gains(tinted_image(), 0.5)


def test_uint8_path_is_the_pretraining_composition():
    """float in [0, 1], the op, then truncation back to uint8: torch_data._finish_view."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(20, 24, 3), dtype=np.uint8)
    expected = (cc.shades_of_gray(img.astype(np.float64) / 255.0, p=6) * 255.0).astype(np.uint8)
    out = cc.shades_of_gray_uint8(img, p=6)
    assert out.dtype == np.uint8 and np.array_equal(out, expected)


def test_uint8_path_refuses_other_dtypes():
    with pytest.raises(ValueError, match="uint8"):
        cc.shades_of_gray_uint8(np.zeros((4, 4, 3)), p=6)
