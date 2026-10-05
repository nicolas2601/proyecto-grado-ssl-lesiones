"""Per-backbone feature extraction and the (backbone-independent) mask descriptors, with a fake extractor."""

import hashlib

import numpy as np

from dermssl import c1_extraction as ex
from dermssl import e3_render as render
from dermssl import e3_store as st
from dermssl.e3b import backbones as bb
from dermssl.e3b import extraction as xt


def load_item(image_id):
    rng = np.random.default_rng(int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16))
    image = rng.integers(0, 255, (90, 120, 3), dtype=np.uint8)
    mask = np.zeros((90, 120), dtype=bool)
    if not image_id.endswith("_0"):  # image 0 has an empty mask -> fallback
        mask[25:65, 35:85] = True
    return image, mask


def resize_crop(image, resolution):
    return ex._center_crop(ex._resize_shorter_side(image, int(resolution * 1.14)), resolution)


def fake_backbone(dim=4):
    def extract(images):
        flat = images.reshape(len(images), -1, 3).astype(np.float32)
        return np.concatenate([flat.mean(axis=1), flat.std(axis=1)], axis=1)[:, :dim].astype(np.float32)

    return bb.Backbone("resnet50", dim, None, resize_crop, extract, "w" * 64, "bilinear")


IDS = [f"s_{i}" for i in range(11)]


def test_features_have_one_row_per_image_in_order_and_the_fallbacks_are_counted(tmp_path):
    meta = xt.extract_model_split(fake_backbone(), "val", IDS, load_item, tmp_path, batch_size=4, workers=2)
    full = np.load(tmp_path / "resnet50" / "val__full.npy")
    crop = np.load(tmp_path / "resnet50" / "val__crop_view.npy")
    flags = np.load(tmp_path / "resnet50" / "val__fallbacks.npy")
    assert full.shape == crop.shape == (11, 4) and flags.shape == (11,)
    assert flags[0] and flags.sum() == 1 and meta["crop_fallbacks"] == 1
    assert np.array_equal(crop[0], full[0]) and not np.array_equal(crop[1], full[1])
    expected = fake_backbone().extract(np.stack([resize_crop(load_item(i)[0], 224) for i in IDS]))
    assert np.allclose(full, expected)
    assert meta["sha256"]["full"] == st.file_sha256(tmp_path / "resnet50" / "val__full.npy")


def test_the_batch_size_does_not_change_the_features(tmp_path):
    xt.extract_model_split(fake_backbone(), "a", IDS, load_item, tmp_path / "x", batch_size=2, workers=1)
    xt.extract_model_split(fake_backbone(), "a", IDS, load_item, tmp_path / "y", batch_size=11, workers=3)
    for part in ("full", "crop_view"):
        assert np.array_equal(np.load(tmp_path / "x" / "resnet50" / f"a__{part}.npy"),
                              np.load(tmp_path / "y" / "resnet50" / f"a__{part}.npy"))


def test_descriptors_equal_the_e3_describe_and_nan_marks_an_empty_mask(tmp_path):
    meta = xt.describe_split("val", IDS, load_item, None, tmp_path, workers=2)
    got = np.load(tmp_path / "val__descriptors.npy")
    want = np.stack([render.describe(*load_item(i), None) for i in IDS])
    assert got.shape == (11, 7) and np.array_equal(got, want, equal_nan=True)
    assert meta["nan_rows"] == 1 and meta["sha256"] == st.file_sha256(tmp_path / "val__descriptors.npy")
