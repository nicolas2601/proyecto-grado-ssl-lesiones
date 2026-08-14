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
    with pytest.raises(ValueError, match="unknown"):
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
