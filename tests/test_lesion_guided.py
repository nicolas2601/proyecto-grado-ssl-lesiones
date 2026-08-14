"""Lesion-guided crops must find the border of a synthetic lesion - and must
fall back to uniform sampling whenever the mask is degenerate."""

import numpy as np
import pytest

from dermssl.pretrain import lesion_guided as lg


def disk_image(size=128, radius=30, lesion=0.2, skin=0.8):
    """Dark disk (lesion) centered on bright background (skin)."""
    yy, xx = np.mgrid[:size, :size]
    dist = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    img = np.full((size, size), skin)
    img[dist <= radius] = lesion
    return img, dist


def test_otsu_separates_a_bimodal_image():
    img, _ = disk_image()
    thr = lg.otsu_threshold(img)
    assert 0.2 < thr < 0.8


def test_lesion_mask_marks_the_dark_region():
    img, dist = disk_image(radius=30)
    mask = lg.lesion_mask(img)
    assert mask[64, 64]  # center of the lesion
    assert not mask[5, 5]  # background corner
    # Area close to the true disk area.
    assert mask.sum() == pytest.approx(np.pi * 30**2, rel=0.1)


def test_border_ring_hugs_the_boundary():
    img, dist = disk_image(radius=30)
    ring = lg.border_ring(lg.lesion_mask(img), width=3)
    assert ring.any()
    distances = dist[ring]
    assert distances.min() >= 30 - 5 and distances.max() <= 30 + 5
    assert not ring[64, 64]  # lesion center is not border


def test_sample_centers_lands_on_the_requested_region():
    img, dist = disk_image(radius=30)
    rng = np.random.default_rng(0)
    centers = lg.sample_centers(rng, lg.border_ring(lg.lesion_mask(img), 3), n=50)
    assert centers.shape == (50, 2)
    d = np.sqrt(((centers - 64.0) ** 2).sum(axis=1))
    assert d.min() >= 25 and d.max() <= 35


def test_guided_centers_fall_back_to_uniform_when_mask_is_degenerate():
    rng = np.random.default_rng(0)
    flat = np.full((64, 64), 0.5)  # no lesion at all
    centers, note = lg.guided_crop_centers(flat, rng, n=10)
    assert centers.shape == (10, 2)
    assert note == "fallback_uniform"

    dark = np.full((64, 64), 0.1)  # "lesion" covers everything
    dark[0, 0] = 0.9
    centers, note = lg.guided_crop_centers(dark, rng, n=10)
    assert note == "fallback_uniform"


def test_guided_centers_report_the_ring_when_it_works():
    img, _ = disk_image()
    rng = np.random.default_rng(0)
    centers, note = lg.guided_crop_centers(img, rng, n=10)
    assert note == "border_ring"
    d = np.sqrt(((centers - 64.0) ** 2).sum(axis=1))
    assert (d > 20).all() and (d < 40).all()


def test_crop_slices_are_clamped_inside_the_image():
    ys, xs = lg.crop_slices(center=(2, 126), size=32, shape=(128, 128))
    assert ys.start == 0 and ys.stop == 32
    assert xs.stop == 128 and xs.start == 96


def test_no_labels_anywhere():
    """The guidance is computed from pixels alone - the function signatures
    must not even accept a label argument."""
    import inspect

    for fn in (lg.otsu_threshold, lg.lesion_mask, lg.guided_crop_centers):
        params = inspect.signature(fn).parameters
        assert not any("label" in p or "target" in p for p in params)
