"""E5 perilesional ring estimator: ring geometry, exclusions and abstention on synthetic images."""

import numpy as np
import pytest

from dermssl import e3_geometry as geo
from dermssl import e5_roi as roi
from dermssl import fitzpatrick as fz

H, W = 120, 160
SKIN = (0.70, 0.52, 0.40)  # ITA ~21: bucket 3 (III)
OTHER_SKIN = (0.93, 0.80, 0.70)  # ITA ~63: bucket 1 (I)


def _lesion(radius: int = 18) -> np.ndarray:
    yy, xx = np.mgrid[:H, :W]
    return (yy - H // 2) ** 2 + (xx - W // 2) ** 2 <= radius**2


def _image(color=SKIN) -> np.ndarray:
    img = np.empty((H, W, 3), dtype=np.float32)
    img[:] = color
    return img


def _ita(color) -> float:
    lab = fz.srgb_to_lab(np.array([[color]], dtype=np.float64))[0, 0]
    return float(fz.ita_degrees(lab[0], lab[2]))


def _paint(img: np.ndarray, where: np.ndarray, color) -> np.ndarray:
    out = img.copy()
    out[where] = color
    return out


def _ring_part(frac: float = 0.6) -> np.ndarray:
    """The first ``frac`` of the ring pixels in raster order (a big contiguous chunk of the ring)."""
    ring = geo.ring_mask(_lesion())
    idx = np.flatnonzero(ring)
    part = np.zeros(ring.size, dtype=bool)
    part[idx[: int(len(idx) * frac)]] = True
    return part.reshape(ring.shape)


def test_a_clean_ring_gives_the_ita_of_its_colour_and_counts_every_ring_pixel():
    est = roi.ring_estimate(_image(), _lesion())
    assert est.status == roi.OK and est.bucket == 3
    assert est.ita == pytest.approx(_ita(SKIN), abs=1e-3)
    assert est.n_px == est.ring_px == int(geo.ring_mask(_lesion()).sum())
    assert est.valid_fraction == 1.0


def test_only_ring_pixels_are_read_not_the_lesion_nor_the_far_skin():
    lesion, ring = _lesion(), geo.ring_mask(_lesion())
    far = ~geo.dilate(lesion, geo.ring_width(lesion) + 2)
    img = _paint(_paint(_image(), lesion, (0.25, 0.15, 0.10)), far, OTHER_SKIN)
    est = roi.ring_estimate(img, lesion)
    assert est.bucket == 3 and est.ita == pytest.approx(_ita(SKIN), abs=1e-3)
    assert est.n_px == int(ring.sum())


def test_specular_highlights_in_the_ring_are_excluded():
    chunk = _ring_part(0.6)
    img = _paint(_image(), chunk, (0.97, 0.97, 0.97))  # white, achromatic: would dominate the median
    est = roi.ring_estimate(img, _lesion())
    assert est.status == roi.OK and est.bucket == 3
    assert est.n_px == int(geo.ring_mask(_lesion()).sum()) - int(chunk.sum())
    assert est.valid_fraction == pytest.approx(0.4, abs=0.01)


def test_clipped_pixels_are_excluded_even_when_they_look_like_skin():
    chunk = _ring_part(0.6)
    img = _paint(_image(), chunk, (1.0, 0.80, 0.70))  # red channel at full scale, skin-like hue
    est = roi.ring_estimate(img, _lesion())
    assert est.bucket == 3 and est.ita == pytest.approx(_ita(SKIN), abs=1e-3)
    assert est.n_px == int(geo.ring_mask(_lesion()).sum()) - int(chunk.sum())


def test_hair_pixels_from_the_injected_detector_are_excluded():
    chunk = _ring_part(0.6)
    img = _paint(_image(), chunk, (0.12, 0.10, 0.09))  # dark: would push the median to a dark bucket
    est = roi.ring_estimate(img, _lesion(), hair_fn=lambda rgb: chunk)
    assert est.bucket == 3 and est.ita == pytest.approx(_ita(SKIN), abs=1e-3)
    assert roi.ring_estimate(img, _lesion()).bucket != 3  # without the detector the dark chunk wins


def test_hair_detector_receives_the_uint8_image():
    seen = {}

    def hair(rgb):
        seen["dtype"], seen["max"] = rgb.dtype, int(rgb.max())
        return np.zeros(rgb.shape[:2], dtype=bool)

    roi.ring_estimate(_image(), _lesion(), hair_fn=hair)
    assert seen["dtype"] == np.uint8 and seen["max"] == round(SKIN[0] * 255)


def test_no_mask_is_not_applicable_and_invents_nothing():
    est = roi.ring_estimate(_image(), None)
    assert est == roi.RoiEstimate(status=roi.NO_MASK)
    assert est.bucket is None and est.ita is None


@pytest.mark.parametrize("mask", [np.zeros((H, W), bool), np.ones((H, W), bool)])
def test_a_mask_that_is_not_one_lesion_abstains(mask):
    est = roi.ring_estimate(_image(), mask)
    assert est.status == roi.DEGENERATE_MASK and est.bucket is None


def test_a_ring_that_is_all_highlight_abstains_with_its_counts():
    img = _paint(_image(), geo.ring_mask(_lesion()), (0.97, 0.97, 0.97))
    est = roi.ring_estimate(img, _lesion())
    assert est.status == roi.RING_TOO_SMALL and est.bucket is None
    assert est.n_px == 0 and est.ring_px > 0 and est.valid_fraction == 0.0


def test_too_few_valid_pixels_abstains_even_when_nothing_is_excluded():
    tiny = np.zeros((H, W), bool)
    tiny[53:67, 73:87] = True  # 14x14 = 196 px (1.02 %): a plausible lesion with a ring under the floor
    est = roi.ring_estimate(_image(), tiny)
    assert est.status == roi.RING_TOO_SMALL and est.bucket is None
    assert est.n_px == est.ring_px < roi.MIN_RING_VALID_PX and est.valid_fraction == 1.0


def test_a_grey_ring_is_ita_indeterminate_with_no_bucket_but_keeps_L():
    est = roi.ring_estimate(_image((0.6, 0.6, 0.6)), _lesion())
    assert est.status == fz.ITA_INDETERMINATE and est.bucket is None
    assert est.L_median is not None and est.b_median < fz.MIN_B_FOR_ITA


def test_mask_and_image_on_different_grids_is_a_typed_error():
    with pytest.raises(roi.RoiShapeError):
        roi.ring_estimate(_image(), np.zeros((H + 1, W), bool))


def test_non_bool_masks_are_coerced_not_bitwise_inverted():
    assert roi.ring_estimate(_image(), _lesion().astype(np.uint8)).bucket == 3


def test_the_real_hair_detector_removes_a_thin_dark_line_crossing_the_ring():
    pytest.importorskip("cv2")
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plan-segmentacion" / "comun" / "nucleo"))
    import hair_removal

    img = _image()
    img[40:43, 5:155] = (0.08, 0.06, 0.05)  # a 3 px dark hair across the upper part of the ring
    ring = geo.ring_mask(_lesion())
    crossing = int((ring & (np.arange(H)[:, None] >= 40) & (np.arange(H)[:, None] < 43)).sum())
    assert crossing > 0
    est = roi.ring_estimate(img, _lesion(), hair_fn=lambda r: hair_removal.elongated_hair_mask(r).astype(bool))
    assert est.ring_px - est.n_px >= crossing  # the detector may take a pixel of margin, never less than the hair
    assert est.bucket == 3


def _frame(width: int = 12) -> np.ndarray:
    frame = np.ones((H, W), bool)
    frame[width:-width, width:-width] = False
    return frame


def test_the_border_estimator_reads_only_the_frame_of_the_image():
    img = _paint(_image(), ~_frame(), OTHER_SKIN)  # the centre is another tone: it must not count
    est = roi.border_estimate(img)
    assert est.status == roi.OK and est.bucket == 3 and est.ita == pytest.approx(_ita(SKIN), abs=1e-3)
    assert est.ring_px == est.n_px == int(_frame(round(roi.BORDER_BAND_FRACTION * min(H, W))).sum())


def test_the_border_estimator_excludes_highlights_and_hair_like_the_ring():
    chunk = _frame(round(roi.BORDER_BAND_FRACTION * min(H, W))) & (np.arange(H)[:, None] < H // 2)
    img = _paint(_image(), chunk, (0.97, 0.97, 0.97))
    assert roi.border_estimate(img).n_px == int(_frame(round(roi.BORDER_BAND_FRACTION * min(H, W))).sum()) - int(chunk.sum())
    dark = _paint(_image(), chunk, (0.12, 0.10, 0.09))
    assert roi.border_estimate(dark, hair_fn=lambda rgb: chunk).bucket == 3


def test_the_border_estimator_abstains_when_the_frame_is_all_highlight():
    est = roi.border_estimate(_paint(_image(), _frame(), (0.97, 0.97, 0.97)))
    assert est.status == roi.RING_TOO_SMALL and est.bucket is None and est.n_px == 0


def test_the_border_estimator_abstains_on_a_grey_frame_with_no_bucket():
    est = roi.border_estimate(_image((0.6, 0.6, 0.6)))
    assert est.status == fz.ITA_INDETERMINATE and est.bucket is None
