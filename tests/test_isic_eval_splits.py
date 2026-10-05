"""ISIC 2019 / 2020 split loaders on synthetic CSVs; expected supports and sha are parameters."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from dermssl.isic_eval import splits as sp

CLASSES = sp.ISIC2019_CLASSES


@pytest.fixture
def isic2019(tmp_path):
    rows = [{"image": f"ISIC_{k:05d}", **{c: int(c == CLASSES[k % 8]) for c in (*CLASSES, "UNK")}}
            for k in range(160)]
    csv = tmp_path / "gt" / "ISIC_2019_Training_GroundTruth.csv"
    csv.parent.mkdir()
    pd.DataFrame(rows).to_csv(csv, index=False)
    images = tmp_path / "train-image" / "nested"
    images.mkdir(parents=True)
    for row in rows:
        (images / f"{row['image']}.jpg").write_bytes(b"")
    return csv, tmp_path / "train-image", rows


@pytest.fixture
def isic2020(tmp_path):
    rows = [{"isic_id": f"ISIC_{k:07d}", "patient_id": f"IP_{k // 3:04d}", "target": int(k % 10 == 0)}
            for k in range(300)]
    csv = tmp_path / "train-metadata.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    images = tmp_path / "train-image"
    images.mkdir()
    for row in rows:
        (images / f"{row['isic_id']}.jpg").write_bytes(b"")
    return csv, images, rows


def _expected2019(rows):
    ids = [r["image"] for r in rows]
    y = np.array([CLASSES.index(next(c for c in CLASSES if r[c])) for r in rows])
    _, va = next(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y))
    supports = tuple(int(v) for v in np.bincount(y[va], minlength=8))
    return sp.SplitExpectation("isic2019", CLASSES, supports, sp.sorted_ids_sha256([ids[i] for i in va]))


def _expected2020(rows):
    ids = [r["isic_id"] for r in rows]
    y = np.array([r["target"] for r in rows])
    groups = np.array([r["patient_id"] for r in rows])
    _, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(ids, y, groups))
    return sp.SplitExpectation("isic2020", sp.ISIC2020_CLASSES, tuple(int(v) for v in np.bincount(y[va], minlength=2)),
                               sp.sorted_ids_sha256([ids[i] for i in va]))


def test_isic2019_split_matches_expectation_and_records_sklearn(isic2019):
    csv, images, rows = isic2019
    ds = sp.load_isic2019(csv, images, _expected2019(rows))
    assert ds.classes == CLASSES and len(ds.ids) == 160 and ds.groups is None
    assert len(ds.val_idx) == 32 and not set(ds.train_ids) & set(ds.val_ids)
    assert ds.identity["sklearn_version"] and ds.identity["val_supports"] == [4] * 8
    assert ds.identity["val_sha_construction"] == "join_newline"


def test_isic2019_accepts_the_trailing_newline_digest(isic2019):
    csv, images, rows = isic2019
    good = _expected2019(rows)
    ds = sp.load_isic2019(csv, images, good)
    from dermssl import c1_split_identity as si
    alt = sp.SplitExpectation(good.name, good.classes, good.val_supports, si.ids_sha256(sorted(ds.val_ids)))
    assert sp.load_isic2019(csv, images, alt).identity["val_sha_construction"] == "trailing_newline"


def test_isic2019_refuses_wrong_supports(isic2019):
    csv, images, rows = isic2019
    good = _expected2019(rows)
    bad = sp.SplitExpectation(good.name, good.classes, (5,) + good.val_supports[1:], good.val_sha256)
    with pytest.raises(sp.SplitMismatchError, match="val supports"):
        sp.load_isic2019(csv, images, bad)


def test_isic2019_refuses_wrong_sha(isic2019):
    csv, images, rows = isic2019
    good = _expected2019(rows)
    bad = sp.SplitExpectation(good.name, good.classes, good.val_supports, "0" * 64)
    with pytest.raises(sp.SplitMismatchError, match="sha256"):
        sp.load_isic2019(csv, images, bad)


def test_the_committed_constants_are_refused_on_synthetic_data(isic2019):
    csv, images, _ = isic2019
    with pytest.raises(sp.SplitMismatchError, match="scikit-learn"):
        sp.load_isic2019(csv, images)


def test_isic2019_drops_rows_without_image_and_unk(isic2019):
    csv, images, rows = isic2019
    (images / "nested" / f"{rows[0]['image']}.jpg").unlink()
    ds = sp.load_isic2019(csv, images, _expected2019(rows[1:]))
    assert len(ds.ids) == 159 and rows[0]["image"] not in ds.paths


def test_isic2020_split_is_grouped_by_patient(isic2020):
    csv, images, rows = isic2020
    ds = sp.load_isic2020(csv, images, _expected2020(rows))
    assert ds.classes == ("benign", "malignant")
    assert not set(ds.groups[ds.train_idx]) & set(ds.groups[ds.val_idx])
    assert len(ds.val_groups) == len(ds.val_idx) and ds.y_val.sum() == ds.identity["val_supports"][1]


def test_isic2020_refuses_a_different_split(isic2020):
    csv, images, rows = isic2020
    good = _expected2020(rows)
    bad = sp.SplitExpectation(good.name, good.classes, good.val_supports, sp.sorted_ids_sha256(["x"]))
    with pytest.raises(sp.SplitMismatchError, match="not the committed one"):
        sp.load_isic2020(csv, images, bad)


def test_sorted_sha_ignores_input_order():
    assert sp.sorted_ids_sha256(["b", "a"]) == sp.sorted_ids_sha256(["a", "b"])
