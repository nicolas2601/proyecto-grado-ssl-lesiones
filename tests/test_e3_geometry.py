"""E3 geometry on synthetic masks with known boxes and areas (no images, no GPU)."""

import numpy as np
import pytest

from dermssl import e3_geometry as g


def square(h=100, w=120, y0=40, y1=60, x0=50, x1=80):
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_lesion_box_adds_ten_percent_of_each_side_and_clamps():
    # 20 x 30 box: dy = 2, dx = 3.
    assert g.lesion_box(square(), 0.10) == (38, 62, 47, 83)
    assert g.lesion_box(square(y0=0, y1=20), 0.10)[0] == 0


def test_lesion_box_of_an_empty_mask_is_none():
    assert g.lesion_box(np.zeros((10, 10), dtype=bool), 0.10) is None


def test_shifted_boxes_are_four_diagonals_of_the_same_size_inside_the_image():
    mask = square()
    base = g.lesion_box(mask, 0.10)
    boxes = g.shifted_boxes(mask, 0.10, 0.15)
    assert len(boxes) == g.N_SHIFTS == 4
    assert len(set(boxes)) == 4 and base not in boxes
    for y0, y1, x0, x1 in boxes:
        assert (y1 - y0, x1 - x0) == (base[1] - base[0], base[3] - base[2])
        assert 0 <= y0 < y1 <= 100 and 0 <= x0 < x1 <= 120


def test_shifted_boxes_slide_back_when_they_would_leave_the_image():
    mask = square(y0=0, y1=20, x0=0, x1=30)
    for y0, y1, x0, x1 in g.shifted_boxes(mask, 0.10, 0.15):
        assert y0 >= 0 and x0 >= 0 and y1 <= 100 and x1 <= 120


def test_shifted_boxes_of_an_empty_mask_is_none():
    assert g.shifted_boxes(np.zeros((8, 8), dtype=bool), 0.1, 0.1) is None


def test_dilate_grows_a_single_pixel_into_a_square():
    mask = np.zeros((9, 9), dtype=bool)
    mask[4, 4] = True
    out = g.dilate(mask, 2)
    assert out.sum() == 25 and out[2:7, 2:7].all()
    assert (g.dilate(mask, 0) == mask).all()


def test_erode_removes_one_pixel_of_border():
    assert g.erode(square()).sum() == 18 * 28


def test_ring_is_outside_the_mask_and_has_the_known_area():
    mask = square()
    ring = g.ring_mask(mask, 3)
    assert not (ring & mask).any()
    assert ring.sum() == (20 + 6) * (30 + 6) - 20 * 30


def test_ring_width_scales_with_the_lesion_and_has_a_floor():
    assert g.ring_width(square()) == round(g.RING_WIDTH_FRACTION * np.sqrt(4 * 600 / np.pi))
    tiny = np.zeros((50, 50), dtype=bool)
    tiny[10, 10] = True
    assert g.ring_width(tiny) == g.MIN_RING_WIDTH


def test_boundary_pixels_of_a_rectangle():
    assert g.boundary(square()).sum() == 2 * (20 + 30) - 4


def test_equivalent_diameter_of_a_known_area():
    assert g.equivalent_diameter(square()) == pytest.approx(np.sqrt(4 * 600 / np.pi))
    assert g.equivalent_diameter(np.zeros((5, 5), dtype=bool)) == 0.0
