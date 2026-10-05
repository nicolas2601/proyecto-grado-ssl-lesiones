"""B71 reference arms read from a real E3 feature directory (built here by e3_run with a fake extractor)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e3b_support as sup  # noqa: E402

from dermssl import feature_cache as fc  # noqa: E402
from dermssl.e3b import b71  # noqa: E402


@pytest.fixture(scope="module")
def e3_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("e3")
    sup.build_e3_dir(out)
    return out


def ids_sha(split):
    return fc.fingerprint_ids([f"{split}_{i}" for i in range(sup.SIZES[split])])


def y_of(split):
    return np.array([sup.label_of(f"{split}_{i}") for i in range(sup.SIZES[split])])


def descriptors(e3_dir, split):
    return np.load(e3_dir / "features" / f"{split}__descriptors.npy")


def test_the_e3_cells_are_verified_against_this_runs_ids_and_masks(e3_dir):
    shas = {"train": ids_sha("train"), "val": ids_sha("val")}
    b71.verify_e3_features(e3_dir, shas, masks_sha256="m" * 64)
    with pytest.raises(b71.E3FeaturesMismatchError, match="ids"):
        b71.verify_e3_features(e3_dir, {**shas, "val": "0" * 64}, masks_sha256="m" * 64)
    with pytest.raises(b71.E3FeaturesMismatchError, match="masks"):
        b71.verify_e3_features(e3_dir, shas, masks_sha256="z" * 64)


def test_a_corrupted_e3_array_is_detected(e3_dir, tmp_path):
    import shutil

    copy = tmp_path / "e3"
    shutil.copytree(e3_dir, copy)
    np.save(copy / "features" / "val__full.npy", np.zeros((8, 18, 6), dtype=np.float32))
    with pytest.raises(b71.E3FeaturesMismatchError, match="sha256"):
        b71.verify_e3_features(copy, {"train": ids_sha("train"), "val": ids_sha("val")}, masks_sha256="m" * 64)


def test_registered_arms_reproduce_the_e3_final_rows_and_notta_arms_use_one_view(e3_dir):
    probas = b71.proposed_probas(e3_dir, y_of("train"), 3, descriptors(e3_dir, "train"),
                                 descriptors(e3_dir, "val"))
    assert set(probas) == {"B71", "B71+F4", "B71_tta", "B71+F4_tta"}
    assert all(p.shape == (18, 3) for p in probas.values())
    final = json.loads((e3_dir / "e3_final_eval.json").read_text())
    check = b71.check_against_e3_final(probas, final, y_of("val"), mel=1)
    assert check["matches"] and set(check["compared"]) == {"B71_tta", "B71+F4_tta"}
    assert not np.allclose(probas["B71"], probas["B71_tta"])  # one view is not the 8-view average


def test_a_mismatch_with_the_e3_final_is_reported_not_hidden(e3_dir):
    probas = b71.proposed_probas(e3_dir, y_of("train"), 3, descriptors(e3_dir, "train"),
                                 descriptors(e3_dir, "val"))
    final = json.loads((e3_dir / "e3_final_eval.json").read_text())
    final["rows"]["F4"]["macro_f1"] += 0.05
    check = b71.check_against_e3_final(probas, final, y_of("val"), mel=1)
    assert not check["matches"] and check["compared"]["B71+F4_tta"]["delta"] == pytest.approx(-0.05, abs=1e-9)
