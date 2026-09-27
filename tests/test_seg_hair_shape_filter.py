"""S2-prime: keep only elongated components of the black-hat mask.

S2 (a) flagged 58,453 of 58,457 cache images: the plain black-hat also
catches pigment network and globules inside the lesion, and TELEA smooths
them. The shape filter keeps a component only if the long side of its
minimum-area rectangle is >= 15 px (at 224) and its mean width (area /
length) is <= 5 px - the thresholds registered in docs/27 before running.
A first version asked for a 3x aspect ratio; crossing synthetic hairs form
one component whose joint rectangle is square, so it dropped all hair.
"""

import numpy as np
import pytest

import seg_paths  # noqa: F401

cv2 = pytest.importorskip("cv2")
import hair_removal  # noqa: E402

SIZE = 224
SKIN = (205, 160, 140)
LESION = (140, 90, 70)
DARK = (45, 30, 25)


def _skin():
    return np.full((SIZE, SIZE, 3), SKIN, dtype=np.uint8)


def _lesion_with_globules_and_network(image):
    cv2.ellipse(image, (112, 112), (60, 42), 15, 0, 360, LESION, -1)
    rng = np.random.default_rng(0)
    for _ in range(60):  # globules: dark dots of radius 1-2
        x, y = rng.integers(62, 162), rng.integers(80, 144)
        cv2.circle(image, (int(x), int(y)), int(rng.integers(1, 3)), DARK, -1)
    for _ in range(40):  # broken network: dashes of 3-6 px
        x, y = rng.integers(62, 162), rng.integers(80, 144)
        dx, dy = rng.integers(-5, 6), rng.integers(-5, 6)
        cv2.line(image, (int(x), int(y)), (int(x + dx), int(y + dy)), DARK, 1)
    return image


def _hairs(image):
    drawn = np.zeros(image.shape[:2], np.uint8)
    for start, end in (((0, 20), (223, 70)), ((30, 223), (200, 0)), ((10, 150), (223, 160))):
        cv2.line(image, start, end, DARK, 1)
        cv2.line(drawn, start, end, 1, 1)
    return image, drawn.astype(bool)


def test_lesion_without_hair_leaves_the_mask_almost_empty():
    image = _lesion_with_globules_and_network(_skin())
    mask = hair_removal.elongated_hair_mask(image)
    assert mask.mean() < 0.005


def test_thin_hair_on_skin_is_kept():
    image, hair = _hairs(_skin())
    mask = hair_removal.elongated_hair_mask(image).astype(bool)
    assert mask[hair].mean() >= 0.90


def test_hair_crossing_a_lesion_is_kept_but_its_globules_are_not():
    image = _lesion_with_globules_and_network(_skin())
    image, hair = _hairs(image)
    mask = hair_removal.elongated_hair_mask(image).astype(bool)
    assert mask[hair].mean() >= 0.90
    lesion = np.zeros(hair.shape, np.uint8)
    cv2.ellipse(lesion, (112, 112), (60, 42), 15, 0, 360, 1, -1)
    off_hair_in_lesion = lesion.astype(bool) & ~cv2.dilate(hair.astype(np.uint8), np.ones((5, 5))).astype(bool)
    assert mask[off_hair_in_lesion].mean() < 0.01


def test_the_registered_thresholds_are_pinned():
    assert hair_removal.MIN_HAIR_LENGTH == 15 and hair_removal.MAX_MEAN_WIDTH == 5.0


def test_shaped_removal_leaves_a_hairless_lesion_untouched():
    image = _lesion_with_globules_and_network(_skin())
    clean, fraction = hair_removal.remove_hair(image, shaped=True)
    assert fraction < 0.005
    assert np.abs(clean.astype(int) - image.astype(int)).mean() < 0.5


def test_cache_stage_records_the_shape_filter(tmp_path):
    import json

    from PIL import Image

    import prepare_cache

    root = tmp_path / "ham"
    root.mkdir()
    Image.fromarray(_lesion_with_globules_and_network(_skin())).save(root / "ISIC_0001.png")
    out = tmp_path / "shaped"
    prepare_cache.run(prepare_cache.parse_args(
        ["--output-dir", str(out), "--resolution", "224", "--root", f"ham10000={root}",
         "--hair-removal", "--hair-shape-filter"]
    ))
    info = json.loads((out / "cache_info.json").read_text())["hair_removal"]
    assert info["shape_filter"] == {"min_length": 15, "max_mean_width": 5.0}
    assert info["images_with_hair"] == 0 or info["mean_fraction"] < 0.005


def test_granular_skin_texture_without_hair_is_not_hair():
    """Like cache image 27657: speckle over skin and lesion, no hair at all."""
    rng = np.random.default_rng(5)
    image = _skin().astype(int) + rng.integers(-25, 26, size=(SIZE, SIZE, 1))
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.ellipse(image, (112, 112), (40, 30), 0, 0, 360, LESION, -1)
    assert hair_removal.elongated_hair_mask(image).mean() < 0.005


def test_heavily_inpainted_images_are_counted_apart():
    """Registered before running: fraction > 0.30 after the filter is mostly paint."""
    import prepare_cache

    info = prepare_cache._hair_info([0.0, 0.1, 0.31, 0.5], shaped=True)
    assert hair_removal.HEAVY_INPAINT_FRACTION == 0.30
    assert info["heavy_inpaint_count"] == 2
    assert info["heavy_inpaint_indices"] == [2, 3]
