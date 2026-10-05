"""E3 final: MCC contrasts from a probe refit on cached features, only where the stored points are reproduced."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from dermssl import e3_mcc_rebuild as mr
from dermssl import e3_render as r
from dermssl import e3_stages as sg
from dermssl import e3_store as st

N_CLASSES, MEL, DIM = 3, 1, 8
SIZES = {"train": 120, "val": 60}
SELECTION = {"decision": "adopt", "val_plan": ["B71", "F4"], "reference": "B71", "best_nomask": "B71"}


def _build(tmp_path):
    rng = np.random.default_rng(0)
    proto = rng.normal(size=(N_CLASSES, DIM))
    labels, groups, codes = {}, {}, {}
    for split, n in SIZES.items():
        y = np.arange(n) % N_CLASSES
        labels[split], groups[split], codes[split] = y, np.array([f"{split}{i // 2}" for i in range(n)]), (np.arange(n) // 2) % 3
        views = 2 if split == "val" else 1
        for cond in r.CONDITIONS:
            st.save_array(tmp_path, split, cond, (proto[y][None] + rng.normal(size=(views, n, DIM))).astype(np.float32))
        st.save_array(tmp_path, split, "descriptors", rng.normal(size=(n, 7)))
    return labels, groups, codes


def _stored(tmp_path, labels, groups, codes):
    """A legacy final-eval document: what the stage wrote before MCC joined the contrast metrics."""
    run = sg.Run(feats_dir=tmp_path, n_classes=N_CLASSES, mel=MEL, labels_of=labels.get, groups_of=groups.get,
                 codes_of=codes.get)
    doc = sg._evaluate(run, SELECTION, "sha", SELECTION["val_plan"], st.FeatureReader(tmp_path), 20)
    for row in doc["rows"].values():
        row["ci95_grouped"].pop("mcc")
    doc["contrasts"] = [c for c in doc["contrasts"] if c["metric"] != "mcc"]
    return json.loads(json.dumps(doc))


def _rebuild(tmp_path, stored, labels, groups, codes):
    def train_only(split):
        if split != "train":
            raise AssertionError("the rebuild must not ask the dataset for validation labels")
        return labels["train"]

    return mr.rebuild(stored, tmp_path, y_train=train_only("train"), y_val=labels["val"], groups=groups["val"],
                      codes=codes["val"], n_classes=N_CLASSES, mel=MEL, n_resamples=20)


def test_the_refit_reproduces_the_stored_points_and_adds_mcc_last(tmp_path):
    labels, groups, codes = _build(tmp_path)
    stored = _stored(tmp_path, labels, groups, codes)
    contrasts, report = _rebuild(tmp_path, stored, labels, groups, codes)
    old = [c for c in contrasts if c["metric"] != "mcc"]
    assert old == stored["contrasts"] and report["cells"] == {"B71": True, "F4": True}
    mcc = [c for c in contrasts if c["metric"] == "mcc"]
    assert len(mcc) == 2 and {c["subset"] for c in mcc} == {"all", "dark_tercile"}
    assert all(c.get("available", True) and "ci95_low" in c for c in mcc)
    assert [c["metric"] for c in contrasts[:4]] == ["macro_f1", "balanced_accuracy", "mel_recall", "mcc"]


def test_a_cell_that_does_not_reproduce_publishes_no_mcc_number(tmp_path):
    labels, groups, codes = _build(tmp_path)
    stored = _stored(tmp_path, labels, groups, codes)
    stored["rows"]["F4"]["macro_f1"] += 0.01
    contrasts, report = _rebuild(tmp_path, stored, labels, groups, codes)
    assert report["cells"] == {"B71": True, "F4": False}
    mcc = [c for c in contrasts if c["metric"] == "mcc"]
    assert len(mcc) == 2 and all(c["available"] is False and c["reason"] == mr.UNAVAILABLE_REASON for c in mcc)
    assert all("point_delta" not in c and "ci95_low" not in c for c in mcc)
    assert [c for c in contrasts if c["metric"] != "mcc"] == stored["contrasts"]
    assert mr.UNAVAILABLE_REASON == "no disponible: el refit no reproduce los puntos registrados"


def test_the_tolerance_is_one_e_minus_four_on_the_three_registered_points(tmp_path):
    labels, groups, codes = _build(tmp_path)
    stored = _stored(tmp_path, labels, groups, codes)
    ok = copy.deepcopy(stored)
    ok["rows"]["F4"]["mel_recall"] += 5e-5
    assert _rebuild(tmp_path, ok, labels, groups, codes)[1]["cells"]["F4"] is True
    bad = copy.deepcopy(stored)
    bad["rows"]["F4"]["balanced_accuracy"] -= 2e-4
    assert _rebuild(tmp_path, bad, labels, groups, codes)[1]["cells"]["F4"] is False


def test_the_output_dir_cannot_be_the_one_holding_the_final_eval_and_its_lock(tmp_path):
    final = tmp_path / "e3" / "e3_final_eval.json"
    final.parent.mkdir()
    with pytest.raises(mr.RebuildInputError, match="own"):
        mr.check_out_dir(final.parent, final)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / st.LOCK_NAME).write_text("{}")
    with pytest.raises(mr.RebuildInputError, match="lock"):
        mr.check_out_dir(tmp_path / "other", final)
    mr.check_out_dir(tmp_path / "fresh", final)


def test_the_script_takes_y_val_from_the_artifact_only_for_the_same_val_ids(tmp_path):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import e3_contrasts_rebuild as script

    np.savez(tmp_path / "p.npz", y_val=np.array([0, 1, 2]), val_ids=np.array(["a", "b", "c"]))
    assert list(script._val_labels(tmp_path / "p.npz", ["a", "b", "c"])) == [0, 1, 2]
    with pytest.raises(mr.RebuildInputError, match="val_ids"):
        script._val_labels(tmp_path / "p.npz", ["a", "c", "b"])


def test_the_features_must_be_the_ones_the_run_hashed(tmp_path):
    labels, groups, codes = _build(tmp_path)
    stored = _stored(tmp_path, labels, groups, codes)
    stored["fingerprint"]["feature_sha256"] = {
        split: {cond: st.file_sha256(tmp_path / f"{split}__{cond}.npy") for cond in ("full", "crop_view", "descriptors")}
        for split in ("train", "val")}
    mr.verify_feature_hashes(stored, tmp_path)
    stored["fingerprint"]["feature_sha256"]["val"]["crop_view"] = "0" * 64
    with pytest.raises(mr.RebuildInputError, match="val.*crop_view"):
        mr.verify_feature_hashes(stored, tmp_path)


def test_the_output_keeps_every_field_but_the_contrasts_and_gets_a_fresh_sidecar(tmp_path):
    from dermssl import run_environment as env

    labels, groups, codes = _build(tmp_path)
    stored = _stored(tmp_path, labels, groups, codes)
    contrasts, report = _rebuild(tmp_path, stored, labels, groups, codes)
    out = mr.write_output(tmp_path / "own", stored, contrasts, report)
    written = json.loads(out.read_text())
    assert out.name == "e3_final_eval.json" and written["contrasts"] == contrasts
    assert {k: v for k, v in written.items() if k not in ("contrasts", "contrasts_rebuild")} == \
        {k: v for k, v in stored.items() if k != "contrasts"}
    assert written["contrasts_rebuild"]["cells"] == {"B71": True, "F4": True}
    assert env.sidecar_problems(env.sidecar_path(out)) == []
    assert not (tmp_path / "own" / st.LOCK_NAME).exists()
