"""Rendering of the E3 conditions from an original-resolution image and its mask."""

import numpy as np
import pytest

from dermssl import c1_extraction as ex
from dermssl import e3_geometry as geo
from dermssl import e3_render as r

SKIN = np.array([200, 150, 130], dtype=np.uint8)
DARK = np.array([60, 30, 20], dtype=np.uint8)
GREY = np.round(np.array(r.IMAGENET_MEAN) * 255).astype(np.uint8)


def resize_crop(image, resolution):
    return ex._center_crop(ex._resize_shorter_side(image, int(resolution * 1.14)), resolution)


def scene(h=450, w=600, box=(180, 270, 240, 360)):
    mask = np.zeros((h, w), dtype=bool)
    mask[box[0]:box[1], box[2]:box[3]] = True
    image = np.empty((h, w, 3), dtype=np.uint8)
    image[:] = SKIN
    image[mask] = DARK
    return image, mask


def render(image, mask, **kw):
    return r.render(image, mask, resize_crop=resize_crop, **kw)


def test_every_condition_is_rendered_with_its_registered_size():
    images, fallbacks = render(*scene())
    assert tuple(images) == r.CONDITIONS and not fallbacks
    for name, array in images.items():
        side = 448 if name == "crop_orig448" else 224
        assert array.shape == (side, side, 3) and array.dtype == np.uint8


def test_the_full_condition_is_the_protocol_view_of_the_whole_image():
    image, mask = scene()
    images, _ = render(image, mask)
    assert (images["full"] == resize_crop(image, 224)).all()


def test_crops_from_the_original_have_a_skin_margin_around_a_dark_lesion():
    images, _ = render(*scene())
    for name in ("crop_view", "crop_orig224", "crop_orig448"):
        crop = images[name]
        mid = crop.shape[0] // 2
        assert (crop[mid, mid] == DARK).all() and (crop[2, 2] == SKIN).all(), name


def test_shifted_crops_still_contain_the_lesion_centre_region():
    images, _ = render(*scene())
    for k in range(geo.N_SHIFTS):
        crop = images[f"shift{k}"]
        assert (crop == DARK).all(axis=2).mean() > 0.4


def test_the_ring_condition_keeps_only_context_pixels_and_greys_the_rest():
    image, mask = scene()
    images, _ = render(image, mask)
    ring = images["ring"]
    assert (ring[112, 112] == GREY).all()  # lesion centre is masked out
    skin = (ring == SKIN).all(axis=2)
    assert skin.any() and not (ring == DARK).all(axis=2).any()


def test_flagged_pixels_are_excluded_from_the_ring():
    image, mask = scene()
    image[150:160, 240:360] = 255  # a specular band just above the lesion, inside the ring
    keep = r.ring_keep(image, mask)
    assert not keep[150:160, 240:360].any() and keep.any()
    assert keep.sum() < geo.ring_mask(mask).sum()


def test_a_hair_function_adds_its_pixels_to_the_flags():
    image, mask = scene()
    hair = np.zeros(mask.shape, dtype=bool)
    hair[100:110, :] = True
    flags = r.artifact_flags(image, hair_fn=lambda rgb: hair)
    assert flags[100:110].all() and not flags[:100].any()


def test_an_empty_mask_falls_back_to_the_full_view_everywhere_and_says_so():
    image, _ = scene()
    images, fallbacks = render(image, np.zeros(image.shape[:2], dtype=bool))
    assert fallbacks == frozenset(set(r.CONDITIONS) - {"full"})
    for name in fallbacks:
        side = 448 if name == "crop_orig448" else 224
        assert (images[name] == resize_crop(image, side)).all()


def test_a_lesion_outside_the_centre_crop_only_breaks_the_view_crop():
    image, mask = scene(box=(0, 60, 0, 60))
    _, fallbacks = render(image, mask)
    assert fallbacks == frozenset({"crop_view"})


def test_the_bank_geometry_of_c1_equals_torchvision_resize_crop():
    torchvision = pytest.importorskip("torchvision")  # noqa: F841
    from PIL import Image

    from dermssl import torch_v71 as tv

    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (450, 600, 3), dtype=np.uint8)
    assert (tv.resize_crop_uint8(Image.fromarray(image), 224) == resize_crop(image, 224)).all()


def test_descriptors_exclude_flagged_pixels_of_the_ring_from_the_skin_reference():
    image, mask = scene()
    clean = r.describe(image, mask)
    image[150:160, 240:360] = 255  # specular band inside the ring
    assert clean.shape == (7,) and np.isfinite(clean).all()
    assert r.describe(image, mask)[-1] == pytest.approx(clean[-1])
