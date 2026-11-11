from __future__ import annotations

import json

import numpy as np
import pytest

from dermssl.release import moco_inputs


def test_feature_cache_requires_exact_order_groups_and_dimensions(tmp_path):
    rows = [{"image_id": "a", "identity_component_id": "p1"},
            {"image_id": "b", "identity_component_id": "p2"}]
    path = tmp_path / "features.npz"
    np.savez(path, features=np.ones((2, 4)), image_ids=["a", "b"], groups=["p1", "p2"])
    assert moco_inputs.load_feature_matrix(path, rows, 4).shape == (2, 4)
    for ids, groups, features in ((["b", "a"], ["p1", "p2"], np.ones((2, 4))),
                                  (["a", "b"], ["p2", "p1"], np.ones((2, 4))),
                                  (["a", "b"], ["p1", "p2"], np.ones((2, 3)))):
        np.savez(path, features=features, image_ids=ids, groups=groups)
        with pytest.raises(ValueError, match="alignment|dimension"):
            moco_inputs.load_feature_matrix(path, rows, 4)


def test_closed_roles_are_rejected_before_any_file_access(tmp_path):
    for task in ("PADtarget_test", "HAMholdout", "ham_refit", ""):
        with pytest.raises(ValueError, match="approved"):
            moco_inputs.load_development(tmp_path, task)


def _external_fixture(tmp_path, *, mode="cls4", size=224, rows_dim=3072):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "external.jsonl"
    records = [{"image_id": "e0", "class_ham": "nv", "class_pad": "NEV", "lesion_id": ""},
               {"image_id": "e1", "class_ham": "", "class_pad": "SCC", "lesion_id": "L1"},
               {"image_id": "e2", "class_ham": "mel", "class_pad": "MEL", "lesion_id": "L1"}]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    features = tmp_path / "features"
    features.mkdir()
    np.save(features / f"s42-{mode}-000.npy", np.ones((3, rows_dim), dtype=np.float32))
    status = {"seed": 42, "size": size, "mode": mode, "rows": 3, "chunks": 1, "done": [0], "state": "COMPLETE",
              "manifest_sha256": moco_inputs.sha256_file(manifest), "weights_sha256": moco_inputs.ENCODER_SHA256}
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(status))
    return manifest, features, status_path


def test_external_rows_must_match_the_v10_recipe_cls4_224(tmp_path):
    for mode, size in (("cls4_d4", 448), ("cls4", 448), ("cls4_d4", 224)):
        manifest, features, status = _external_fixture(tmp_path / f"{mode}-{size}", mode=mode, size=size)
        with pytest.raises(ValueError, match="recipe"):
            moco_inputs.load_external_rows(manifest, features, status, "HAM")


def test_external_rows_are_mapped_to_native_classes_and_unmapped_rows_are_counted(tmp_path):
    manifest, features, status = _external_fixture(tmp_path)
    x, labels, groups, provenance = moco_inputs.load_external_rows(manifest, features, status, "HAM")
    assert x.shape == (2, 3072)
    assert list(labels) == ["nv", "mel"]
    assert list(groups) == ["e0", "L1"]
    assert provenance["rows_total"] == 3 and provenance["rows_used"] == 2
    assert provenance["dropped_unmapped"] == 1
    x, labels, groups, _ = moco_inputs.load_external_rows(manifest, features, status, "PAD")
    assert list(labels) == ["NEV", "SCC", "MEL"] and len(groups) == 3
