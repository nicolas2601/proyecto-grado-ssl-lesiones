"""Select and final on synthetic feature files: the two stages cannot be mixed."""

import json

import numpy as np
import pytest

from dermssl import e3_selection as sel
from dermssl import e3_stages as sg
from dermssl import e3_store as st
from dermssl import e3_render as r

N_CLASSES, MEL, DIM = 3, 1, 8
SIZES = {"fit": 90, "holdout": 45, "train": 120, "val": 60}
VIEWS = {"fit": 1, "holdout": 2, "train": 1, "val": 2}


def build(tmp_path, signal=1.0, conditions=r.CONDITIONS):
    """Feature files for every split; labels, groups and terciles as plain dicts."""
    rng = np.random.default_rng(0)
    labels, groups, codes = {}, {}, {}
    proto = rng.normal(size=(N_CLASSES, DIM))
    for split, n in SIZES.items():
        y = np.arange(n) % N_CLASSES
        labels[split] = y
        groups[split] = np.array([f"{split}{i // 2}" for i in range(n)])  # two images per lesion
        codes[split] = (np.arange(n) // 2) % 3
        for cond in conditions:
            x = proto[y][None] * signal + rng.normal(size=(VIEWS[split], n, DIM))
            st.save_array(tmp_path, split, cond, x.astype(np.float32))
        st.save_array(tmp_path, split, "descriptors", rng.normal(size=(n, 7)))
    return labels, groups, codes


class Tripwire:
    """Records every label/group/tercile access so a test can prove what a stage touched."""

    def __init__(self, table, forbidden=()):
        self.table, self.forbidden, self.seen = table, set(forbidden), []

    def __call__(self, split):
        self.seen.append(split)
        if split in self.forbidden:
            raise AssertionError(f"el split {split!r} no debia leerse")
        return self.table[split]


def make_run(tmp_path, labels, groups, codes, forbidden=()):
    return sg.Run(feats_dir=tmp_path, n_classes=N_CLASSES, mel=MEL, labels_of=Tripwire(labels, forbidden),
                  groups_of=Tripwire(groups, forbidden), codes_of=Tripwire(codes, forbidden))


def run_select(tmp_path, run, variants=("B71", "F1", "F2", "F3", "F4", "F5", "F2_448"), cells=None):
    cells = cells or st.CellStore(tmp_path / "cells")
    return sg.select_stage(run, cells, variants=variants, n_resamples=10), cells


def test_select_never_opens_validation_or_train_arrays_nor_their_labels(tmp_path, monkeypatch):
    labels, groups, codes = build(tmp_path)
    opened = []
    real_load = np.load
    monkeypatch.setattr(np, "load", lambda path, *a, **k: opened.append(str(path)) or real_load(path, *a, **k))
    run = make_run(tmp_path, labels, groups, codes, forbidden=("val", "train"))
    doc, cells = run_select(tmp_path, run)
    assert not cells.failures and opened
    assert not [p for p in opened if "val__" in p or "train__" in p]
    assert set(run.labels_of.seen) <= {"fit", "holdout"} and "val" not in run.codes_of.seen


def test_select_scores_every_variant_on_the_holdout_and_records_the_rule(tmp_path):
    labels, groups, codes = build(tmp_path)
    doc, _ = run_select(tmp_path, make_run(tmp_path, labels, groups, codes))
    assert set(doc["rows"]) == {"B71", "F1", "F2", "F2_448", "F3", "F4", "F5"}
    assert doc["split"] == "holdout" and doc["decision"] in ("adopt", "no_adoption")
    assert doc["rule"] == sel.RULE and doc["manifest_sha256"] == sel.MANIFEST_SHA256
    assert doc["val_plan"][0] == "B71"
    for row in doc["rows"].values():
        assert {"macro_f1", "mel_recall", "dark_macro_f1", "ece", "ci95"} <= set(row)


def test_select_is_resumable_cell_by_cell(tmp_path):
    labels, groups, codes = build(tmp_path)
    run = make_run(tmp_path, labels, groups, codes)
    doc1, cells = run_select(tmp_path, run)
    again, _ = run_select(tmp_path, run, cells=st.CellStore(tmp_path / "cells"))
    assert again["rows"] == doc1["rows"]
    assert not list((tmp_path / "cells").glob("*.stale-*"))


def test_a_variant_without_extracted_features_fails_its_cell_and_is_listed(tmp_path):
    labels, groups, codes = build(tmp_path, conditions=("full", "crop_view", "crop_orig224"))
    cells = st.CellStore(tmp_path / "cells")
    sg.select_stage(make_run(tmp_path, labels, groups, codes), cells, variants=("B71", "F1", "F5"), n_resamples=5)
    assert [name for name, _ in cells.failures] == ["select__F5"]


def test_the_val_plan_is_reference_nomask_best_and_the_winner_or_the_report_candidate():
    adopted = {"decision": "adopt", "winner": "F1", "best_nomask": "B71", "reference": "B71"}
    assert sg.val_plan(adopted, report_candidate="F2") == ["B71", "F1"]
    none = {"decision": "no_adoption", "winner": None, "best_nomask": "B71", "reference": "B71"}
    assert sg.val_plan(none, report_candidate="F2") == ["B71", "F2"]


def test_a_no_mask_bar_from_outside_our_variants_is_not_planned_for_val():
    outside = {"decision": "adopt", "winner": "F1", "best_nomask": "v71:B_soup", "reference": "B71"}
    assert sg.val_plan(outside, report_candidate="F2") == ["B71", "F1"]


def selection(tmp_path, run):
    doc, _ = run_select(tmp_path, run)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(doc))
    return path, st.file_sha256(path)


def test_final_aborts_without_a_selection_and_reads_no_validation_data(tmp_path):
    labels, groups, codes = build(tmp_path)
    run = make_run(tmp_path, labels, groups, codes)
    with pytest.raises(st.SelectionMissingError):
        sg.final_stage(run, selection_path=tmp_path / "nope.json", pinned_sha256="0" * 64,
                       lock=st.ValLock(tmp_path), n_resamples=5)
    assert run.labels_of.seen == [] and not st.ValLock(tmp_path).taken()


def test_final_aborts_when_the_pin_does_not_match(tmp_path):
    labels, groups, codes = build(tmp_path)
    run = make_run(tmp_path, labels, groups, codes)
    path, _ = selection(tmp_path, make_run(tmp_path, labels, groups, codes))
    with pytest.raises(st.SelectionPinError):
        sg.final_stage(run, selection_path=path, pinned_sha256="f" * 64, lock=st.ValLock(tmp_path),
                       n_resamples=5)
    assert not st.ValLock(tmp_path).taken()


def test_final_reads_val_once_writes_contrasts_and_refuses_a_second_read(tmp_path):
    labels, groups, codes = build(tmp_path)
    path, sha = selection(tmp_path, make_run(tmp_path, labels, groups, codes))
    run = make_run(tmp_path, labels, groups, codes)
    doc = sg.final_stage(run, selection_path=path, pinned_sha256=sha, lock=st.ValLock(tmp_path), n_resamples=20)
    assert doc["selection_sha256"] == sha and st.ValLock(tmp_path).taken()
    assert doc["bootstrap"] == {"n_resamples": 20, "seed": 20260923, "grouped_by": "lesion_id"}
    plan = doc["val_plan"]
    assert set(doc["rows"]) == set(plan)
    assert {c["arm"] for c in doc["contrasts"]} == set(plan) - set(doc["controls"])
    assert {c["subset"] for c in doc["contrasts"]} == {"all", "dark_tercile"}
    for contrast in doc["contrasts"]:
        assert {"metric", "control", "ci95_low", "ci95_high", "direction"} <= set(contrast)
    second = make_run(tmp_path, labels, groups, codes)
    with pytest.raises(st.ValAlreadyReadError):
        sg.final_stage(second, selection_path=path, pinned_sha256=sha, lock=st.ValLock(tmp_path), n_resamples=5)
    assert second.labels_of.seen == []  # the refusal comes before any validation label is read


def test_final_checks_the_features_exist_before_taking_the_lock(tmp_path):
    labels, groups, codes = build(tmp_path)
    path, sha = selection(tmp_path, make_run(tmp_path, labels, groups, codes))
    (tmp_path / "val__full.npy").unlink()
    with pytest.raises(st.MissingFeaturesError):
        sg.final_stage(make_run(tmp_path, labels, groups, codes), selection_path=path, pinned_sha256=sha,
                       lock=st.ValLock(tmp_path), n_resamples=5)
    assert not st.ValLock(tmp_path).taken()
