"""The extract loop with a fake extractor: files, shapes, view counts, fallback accounting."""

import numpy as np
import pytest

from dermssl import c1_extraction as ex
from dermssl import e3_extract as xt
from dermssl import e3_render as r
from dermssl import e3_store as st

DIM = 4


def resize_crop(image, resolution):
    return ex._center_crop(ex._resize_shorter_side(image, int(resolution * 1.14)), resolution)


def item(image_id):
    rng = np.random.default_rng(abs(hash(image_id)) % 2**32)
    image = rng.integers(0, 200, (90, 120, 3), dtype=np.uint8)
    mask = np.zeros((90, 120), dtype=bool)
    if not image_id.startswith("empty"):
        mask[30:60, 40:80] = True
    return image, mask


class FakeExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, images, n_views):
        self.calls.append((images.shape, n_views))
        base = images.reshape(len(images), -1).astype(np.float32).mean(axis=1)
        return np.stack([np.tile((base + v)[:, None], (1, DIM)) for v in range(n_views)])


def run(tmp_path, ids, split="holdout", conditions=("full", "crop_view"), n_views=2, batch_size=2, fake=None):
    fake = fake or FakeExtractor()
    meta = xt.extract_cell(split, ids, conditions, load_item=item, extract_fn=fake, resize_crop=resize_crop,
                           out_dir=tmp_path, batch_size=batch_size, n_views=n_views)
    return meta, fake


def test_arrays_have_views_images_dim_and_the_ids_order(tmp_path):
    ids = ["a", "b", "c", "d", "e"]
    meta, fake = run(tmp_path, ids)
    reader = st.FeatureReader(tmp_path)
    full = reader.load("holdout", "full")
    assert full.shape == (2, 5, DIM) and full.dtype == np.float32
    assert reader.load("holdout", "crop_view").shape == (2, 5, DIM)
    assert meta["n"] == 5 and meta["conditions"] == ["full", "crop_view"]
    assert all(views == 2 for _, views in fake.calls) and len(fake.calls) == 2 * 3  # 3 batches x 2 conditions


def test_the_features_of_each_image_do_not_depend_on_the_batch_size(tmp_path):
    ids = ["a", "b", "c", "d", "e"]
    run(tmp_path / "x", ids, batch_size=1)
    run(tmp_path / "y", ids, batch_size=5)
    for cond in ("full", "crop_view"):
        assert (np.load(tmp_path / "x" / f"holdout__{cond}.npy") == np.load(tmp_path / "y" / f"holdout__{cond}.npy")).all()


def test_only_the_requested_conditions_are_extracted(tmp_path):
    _, fake = run(tmp_path, ["a", "b"], conditions=("full",))
    assert not (tmp_path / "holdout__crop_view.npy").exists()
    assert len(fake.calls) == 1


def test_crop_448_reaches_the_extractor_at_448(tmp_path):
    _, fake = run(tmp_path, ["a"], conditions=("crop_orig448",), n_views=1)
    assert fake.calls[0][0] == (1, 448, 448, 3)


def test_descriptors_and_fallbacks_are_saved_and_the_counts_reported(tmp_path):
    meta, _ = run(tmp_path, ["a", "empty1", "b"], conditions=("full", "crop_orig224"))
    desc = np.load(tmp_path / "holdout__descriptors.npy")
    flags = np.load(tmp_path / "holdout__fallbacks.npy")
    assert desc.shape == (3, 7) and np.isnan(desc[1]).all() and np.isfinite(desc[0]).all()
    assert flags.shape == (3, len(r.CONDITIONS)) and flags[1].sum() == len(r.CONDITIONS) - 1
    assert meta["fallbacks"]["crop_orig224"] == 1 and meta["fallbacks"]["full"] == 0
    assert meta["descriptors_nan_rows"] == 1


def test_a_unknown_condition_is_refused_before_any_work(tmp_path):
    with pytest.raises(ValueError):
        run(tmp_path, ["a"], conditions=("full", "bogus"))


def test_meta_records_the_sha256_of_every_array(tmp_path):
    meta, _ = run(tmp_path, ["a", "b"])
    assert meta["sha256"]["full"] == st.file_sha256(tmp_path / "holdout__full.npy")
