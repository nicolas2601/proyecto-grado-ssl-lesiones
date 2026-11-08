from __future__ import annotations

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
