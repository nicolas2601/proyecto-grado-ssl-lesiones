"""E5 dataset adapters: ISIC through the verified loaders, HAM10000 with its masks, smoke limits."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import e5_data as data  # noqa: E402

from dermssl.isic_eval import splits as sp  # noqa: E402


def _eval_dataset(n=12) -> sp.EvalDataset:
    ids = tuple(f"ISIC_{i:04d}" for i in range(n))
    labels = np.array([i % 2 for i in range(n)])
    return sp.EvalDataset("isic2020", ("benign", "malignant"), ids, labels,
                          np.array([f"p{i // 2}" for i in range(n)]),
                          {i: Path(f"/d/{i}.jpg") for i in ids},
                          train_idx=np.arange(0, 8), val_idx=np.arange(8, n),
                          identity={"sklearn_version": "1.7.2", "n_train": 8, "n_val": n - 8})


def test_an_isic_spec_keeps_loader_order_labels_splits_and_declares_no_masks():
    spec = data.spec_from_eval_dataset(_eval_dataset())
    assert spec.name == "isic2020" and spec.classes == ("benign", "malignant") and spec.splits == ("train", "val")
    assert [t.image_id for t in spec.tasks] == [f"ISIC_{i:04d}" for i in range(12)]
    assert [t.split for t in spec.tasks] == ["train"] * 8 + ["val"] * 4
    assert [t.label for t in spec.tasks[:3]] == ["benign", "malignant", "benign"]
    assert spec.masks is None and all(t.mask_pos is None for t in spec.tasks)
    assert spec.provenance["split"] == {"sklearn_version": "1.7.2", "n_train": 8, "n_val": 4}


def test_the_smoke_limit_takes_n_per_split_round_robin_over_the_classes():
    spec = data.spec_from_eval_dataset(_eval_dataset(), limit=2)
    by_split = {s: [t for t in spec.tasks if t.split == s] for s in spec.splits}
    assert all(len(v) == 2 for v in by_split.values())
    assert {t.label for t in by_split["train"]} == {"benign", "malignant"}
    assert [t.image_id for t in spec.tasks] == sorted(t.image_id for t in spec.tasks)


def test_ham_tasks_carry_the_mask_position_of_their_id_in_the_manifest():
    split = {"train_ids": ["a", "b"], "val_ids": ["c"], "classes": ["mel", "nv"], "id2label": {"a": 0, "b": 1, "c": 1}}
    paths = {i: Path(f"/h/{i}.jpg") for i in "abc"}
    positions = {"c": 0, "a": 1, "b": 2, "zzz": 3}
    spec = data.spec_from_ham(split, paths, positions, masks_info={"masks_sha256": "ab"}, provenance={"p": 1})
    assert [(t.image_id, t.split, t.label, t.mask_pos) for t in spec.tasks] == [
        ("a", "train", "mel", 1), ("b", "train", "nv", 2), ("c", "val", "nv", 0)]
    assert spec.name == "ham10000" and spec.masks == {"masks_sha256": "ab"}


def test_a_ham_image_without_a_mask_aborts_instead_of_silently_dropping_it():
    split = {"train_ids": ["a", "b"], "val_ids": [], "classes": ["mel"], "id2label": {"a": 0, "b": 0}}
    paths = {"a": Path("/h/a.jpg"), "b": Path("/h/b.jpg")}
    with pytest.raises(data.MaskCoverageError, match="b"):
        data.spec_from_ham(split, paths, {"a": 0}, masks_info={}, provenance={})


def test_an_image_without_a_file_is_a_typed_error():
    split = {"train_ids": ["a", "b"], "val_ids": [], "classes": ["mel"], "id2label": {"a": 0, "b": 0}}
    with pytest.raises(data.MissingImageError, match="b"):
        data.spec_from_ham(split, {"a": Path("/h/a.jpg")}, {"a": 0, "b": 1}, masks_info={}, provenance={})


def test_isic_with_masks_is_refused_because_none_exist_for_it():
    with pytest.raises(SystemExit, match="no tiene máscaras"):
        data.check_mask_args("isic2019", masks="m.npy", masks_manifest="m.json")


def test_ham_without_masks_is_refused_because_the_ring_needs_them():
    with pytest.raises(SystemExit, match="--masks"):
        data.check_mask_args("ham10000", masks=None, masks_manifest=None)
    data.check_mask_args("isic2019", masks=None, masks_manifest=None)
    data.check_mask_args("ham10000", masks="m.npy", masks_manifest="m.json")
