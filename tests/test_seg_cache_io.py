"""Pure tests for the cache layout, manifest and hold-out split."""

import numpy as np
import pytest
from PIL import Image

import seg_paths  # noqa: F401
import cache_io


def _touch_image(path, color=(10, 20, 30), size=(12, 8)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def test_parse_root_spec_splits_name_and_path():
    name, path = cache_io.parse_root_spec("ham10000=/data/ham")
    assert name == "ham10000"
    assert str(path).replace("\\", "/") == "/data/ham"


@pytest.mark.parametrize("spec", ["nopath", "=x", "name="])
def test_parse_root_spec_rejects_malformed_values(spec):
    with pytest.raises(ValueError):
        cache_io.parse_root_spec(spec)


def test_scan_is_sorted_and_recursive(tmp_path):
    _touch_image(tmp_path / "a" / "ISIC_0002.jpg")
    _touch_image(tmp_path / "a" / "sub" / "ISIC_0001.jpg")
    _touch_image(tmp_path / "a" / "notes.txt".replace("txt", "jpg"))
    result = cache_io.scan_image_roots({"a": tmp_path / "a"})
    ids = [r.image_id for r in result.records]
    assert ids == sorted(ids)
    assert len(ids) == 3
    assert result.duplicates_dropped == 0


def test_scan_drops_duplicate_ids_across_roots_first_root_wins(tmp_path):
    _touch_image(tmp_path / "ham" / "ISIC_1.jpg")
    _touch_image(tmp_path / "i19" / "ISIC_1.jpg")
    _touch_image(tmp_path / "i19" / "ISIC_2.jpg")
    roots = {"ham10000": tmp_path / "ham", "isic2019": tmp_path / "i19"}
    result = cache_io.scan_image_roots(roots, dedupe_by_id=True)
    assert result.duplicates_dropped == 1
    by_id = {r.image_id: r.dataset for r in result.records}
    assert by_id == {"ISIC_1": "ham10000", "ISIC_2": "isic2019"}


def test_scan_keeps_duplicates_when_dedupe_is_off(tmp_path):
    _touch_image(tmp_path / "ham" / "ISIC_1.jpg")
    _touch_image(tmp_path / "i19" / "ISIC_1.jpg")
    roots = {"ham10000": tmp_path / "ham", "isic2019": tmp_path / "i19"}
    result = cache_io.scan_image_roots(roots, dedupe_by_id=False)
    assert len(result.records) == 2


def test_scan_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cache_io.scan_image_roots({"x": tmp_path / "missing"})


def test_assign_splits_holds_out_requested_fraction():
    keys = [f"ds:img{i}" for i in range(1000)]
    splits = cache_io.assign_splits(keys, holdout_fraction=0.02, seed=1)
    assert splits.count("val") == 20
    assert splits.count("train") == 980


def test_assign_splits_always_holds_out_at_least_one_when_possible():
    keys = [f"ds:img{i}" for i in range(64)]
    splits = cache_io.assign_splits(keys, holdout_fraction=0.02, seed=1)
    assert splits.count("val") == 1


def test_assign_splits_single_image_stays_in_train():
    assert cache_io.assign_splits(["only"], 0.5, seed=0) == ["train"]


def test_assign_splits_is_deterministic_and_order_independent():
    keys = [f"ds:img{i}" for i in range(200)]
    first = dict(zip(keys, cache_io.assign_splits(keys, 0.1, seed=3)))
    reversed_keys = list(reversed(keys))
    second = dict(zip(reversed_keys, cache_io.assign_splits(reversed_keys, 0.1, seed=3)))
    assert first == second


def test_assign_splits_changes_with_seed():
    keys = [f"ds:img{i}" for i in range(200)]
    a = cache_io.assign_splits(keys, 0.1, seed=3)
    b = cache_io.assign_splits(keys, 0.1, seed=4)
    assert a != b


def test_assign_splits_rejects_bad_fraction():
    with pytest.raises(ValueError):
        cache_io.assign_splits(["a", "b"], 1.0, seed=0)


def test_manifest_roundtrip(tmp_path):
    records = [
        cache_io.ImageRecord("ham10000", "ISIC_1", "/x/ISIC_1.jpg"),
        cache_io.ImageRecord("isic2019", "ISIC_2", "/x/ISIC_2.jpg"),
    ]
    path = tmp_path / "manifest.csv"
    cache_io.write_manifest(path, records, ["train", "val"])
    rows = cache_io.read_manifest(path)
    assert [r["index"] for r in rows] == [0, 1]
    assert [r["split"] for r in rows] == ["train", "val"]
    assert rows[1]["image_id"] == "ISIC_2"
    assert cache_io.split_indices(rows, "val").tolist() == [1]


def test_manifest_columns_never_include_labels_or_masks():
    joined = " ".join(cache_io.MANIFEST_COLUMNS).lower()
    assert "label" not in joined and "mask" not in joined and "diagnos" not in joined


def test_cache_roundtrip_and_readonly_open(tmp_path):
    path = tmp_path / "images.npy"
    cache = cache_io.create_cache(path, (3, 4, 4, 3), np.uint8)
    cache[1] = 7
    cache.flush()
    reopened = cache_io.open_cache(path)
    assert reopened.shape == (3, 4, 4, 3)
    assert int(reopened[1].mean()) == 7
    with pytest.raises(ValueError):
        reopened[0] = 1


def test_cache_uses_memmap_by_default_and_can_be_preloaded(tmp_path):
    path = tmp_path / "images.npy"
    cache = cache_io.create_cache(path, (2, 4, 4, 3), np.uint8)
    cache[:] = 3
    cache.flush()

    mapped = cache_io.open_cache(path)
    loaded = cache_io.open_cache(path, preload=True)

    assert isinstance(mapped, np.memmap)
    assert isinstance(loaded, np.ndarray)
    assert not isinstance(loaded, np.memmap)
    assert np.array_equal(mapped, loaded)


def test_resize_image_returns_square_uint8(tmp_path):
    path = tmp_path / "img.jpg"
    _touch_image(path, size=(30, 20))
    array = cache_io.load_resized_rgb(path, 16)
    assert array.shape == (16, 16, 3)
    assert array.dtype == np.uint8


def test_pseudo_mask_label_constants_are_distinct():
    values = {cache_io.BACKGROUND, cache_io.FOREGROUND, cache_io.IGNORE}
    assert len(values) == 3
    assert cache_io.IGNORE == 255
