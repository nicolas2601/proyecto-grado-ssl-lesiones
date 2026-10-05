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
    assert spec.masks is None and all(t.mask_pos is None and t.mask_state is None for t in spec.tasks)
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
    assert [(t.image_id, t.split, t.label, t.mask_pos, t.mask_state) for t in spec.tasks] == [
        ("a", "train", "mel", 1, "c1_native"), ("b", "train", "nv", 2, "c1_native"), ("c", "val", "nv", 0, "c1_native")]
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


def test_isic_needs_the_pool_root_and_must_not_get_the_ham_masks():
    with pytest.raises(SystemExit, match="--pool-root"):
        data.check_mask_args("isic2019", masks=None, masks_manifest=None, pool_root=None)
    with pytest.raises(SystemExit, match="HAM10000"):
        data.check_mask_args("isic2019", masks="m.npy", masks_manifest="m.json", pool_root="/p")
    data.check_mask_args("isic2020", masks=None, masks_manifest=None, pool_root="/p")


def test_ham_needs_its_native_masks_and_must_not_get_the_pool():
    with pytest.raises(SystemExit, match="--masks"):
        data.check_mask_args("ham10000", masks=None, masks_manifest=None, pool_root=None)
    with pytest.raises(SystemExit, match="--pool-root"):
        data.check_mask_args("ham10000", masks="m.npy", masks_manifest="m.json", pool_root="/p")
    data.check_mask_args("ham10000", masks="m.npy", masks_manifest="m.json", pool_root=None)


def _pool(tmp_path, entries):
    import json

    for ds, image_id, eligible in entries:
        for kind in ("originals", "masks"):
            (tmp_path / kind / ds).mkdir(parents=True, exist_ok=True)
            (tmp_path / kind / ds / f"{image_id}.png").write_bytes(b"x")
    manifest = [{"dataset": ds, "image_id": i, "eligible": e, "quarantine_reason": None if e else "degenerate_area",
                 "mask_sha256": "ab" * 32} for ds, i, e in entries]
    (tmp_path / "mask_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data.load_pool(tmp_path)


def test_the_pool_index_gives_each_isic_image_its_state_and_its_224_files(tmp_path):
    pool = _pool(tmp_path, [("isic2020", "ISIC_0000", True), ("isic2020", "ISIC_0001", False)])
    ok = pool.fields("isic2020", "ISIC_0000")
    assert ok["mask_state"] == "eligible" and ok["pool_mask"].endswith("masks/isic2020/ISIC_0000.png")
    assert ok["pool_image"].endswith("originals/isic2020/ISIC_0000.png") and ok["mask_sha256"] == "ab" * 32
    q = pool.fields("isic2020", "ISIC_0001")
    assert q["mask_state"] == "quarantine" and q["pool_mask"] is None and q["pool_image"].endswith("ISIC_0001.png")
    assert pool.fields("isic2020", "ISIC_9999") == {"mask_state": "absent", "pool_image": None, "pool_mask": None,
                                                    "mask_sha256": None}


def test_isic2019_looks_in_its_own_pool_folder_then_in_the_ham_passes(tmp_path):
    pool = _pool(tmp_path, [("isic2019", "ISIC_A", True), ("ham10000_p2", "ISIC_B", True), ("ham10000_p1", "ISIC_C", False)])
    assert pool.fields("isic2019", "ISIC_A")["pool_mask"].endswith("masks/isic2019/ISIC_A.png")
    assert pool.fields("isic2019", "ISIC_B")["pool_mask"].endswith("masks/ham10000_p2/ISIC_B.png")
    assert pool.fields("isic2019", "ISIC_C")["mask_state"] == "quarantine"
    assert pool.fields("isic2020", "ISIC_B")["mask_state"] == "absent"  # ISIC 2020 never looks in the HAM passes


def test_an_isic_spec_with_a_pool_carries_states_and_the_pool_provenance(tmp_path):
    ds = _eval_dataset()
    pool = _pool(tmp_path, [("isic2020", ds.ids[0], True), ("isic2020", ds.ids[1], False)])
    spec = data.spec_from_eval_dataset(ds, pool=pool)
    states = [t.mask_state for t in spec.tasks]
    assert states[:2] == ["eligible", "quarantine"] and set(states[2:]) == {"absent"}
    assert spec.masks["n_entries"] == 2 and len(spec.masks["manifest_sha256"]) == 64
    assert "automáticas" in spec.masks["kind"] and spec.masks["found_in"] == {"isic2020": 2}
