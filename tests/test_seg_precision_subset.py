"""The precision-control subset must be reproducible and aligned across files."""

import csv

import numpy as np
import pytest

import seg_paths  # noqa: F401
import cache_io
import precision_subset as ps


def test_indices_are_sorted_unique_and_reproducible():
    a, b = ps.subset_indices(1000, 50, 7), ps.subset_indices(1000, 50, 7)
    assert np.array_equal(a, b) and len(set(a.tolist())) == 50 and (np.diff(a) > 0).all()
    assert not np.array_equal(a, ps.subset_indices(1000, 50, 8))


def test_more_than_available_refuses():
    with pytest.raises(ValueError):
        ps.subset_indices(10, 11)


def test_subset_files_stay_aligned(tmp_path):
    src = tmp_path / "cache"
    src.mkdir()
    n = 12
    images = cache_io.create_cache(src / "images.npy", (n, 4, 4, 3), "uint8")
    images[:] = np.arange(n, dtype=np.uint8)[:, None, None, None]
    images.flush()
    field = cache_io.create_cache(src / "field.npy", (n, 4, 4), "uint8")
    field[:] = 1
    field.flush()
    sal = cache_io.create_cache(tmp_path / "saliency.npy", (n, 4, 4), np.float16)
    sal[:] = np.arange(n, dtype=np.float16)[:, None, None]
    sal.flush()
    with open(src / "manifest.csv", "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["index", "dataset", "image_id", "path", "split"])
        for i in range(n):
            w.writerow([i, "ham", f"ID{i}", f"/p/{i}.jpg", "train"])
    record = ps.write_subset(src, tmp_path / "saliency.npy", tmp_path / "out", n=5, seed=3)
    idx = record["indices"]
    got_images = cache_io.open_cache(tmp_path / "out" / "cache" / "images.npy")
    got_sal = cache_io.open_cache(tmp_path / "out" / "published_saliency.npy")
    assert got_images[:, 0, 0, 0].tolist() == idx
    assert got_sal[:, 0, 0].astype(int).tolist() == idx
    with open(tmp_path / "out" / "cache" / "manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["image_id"] for r in rows] == [f"ID{i}" for i in idx]
    assert [int(r["index"]) for r in rows] == list(range(5))
