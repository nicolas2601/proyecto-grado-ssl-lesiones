"""Resumable cells, the selection pin, the val lock and the val-free reader of E3."""

import json

import numpy as np
import pytest

from dermssl import e3_store as st


def key(**kw):
    return {"protocol": "B71", **kw}


def test_a_finished_cell_is_skipped_when_its_key_sha_matches(tmp_path):
    store = st.CellStore(tmp_path)
    calls = []
    assert store.run("c1", key(a=1), lambda: calls.append(1) or {"x": 1}) == "done"
    assert store.run("c1", key(a=1), lambda: calls.append(1) or {"x": 2}) == "skipped"
    assert len(calls) == 1 and store.read("c1")["result"] == {"x": 1}


def test_a_cell_with_another_key_is_recomputed_and_the_old_file_is_kept_aside(tmp_path):
    store = st.CellStore(tmp_path)
    store.run("c1", key(a=1), lambda: {"x": 1})
    assert store.run("c1", key(a=2), lambda: {"x": 2}) == "done"
    assert store.read("c1")["result"] == {"x": 2}
    assert len(list(tmp_path.glob("c1.json.stale-*"))) == 1


def test_a_failing_cell_is_retried_twice_then_listed_and_the_rest_continues(tmp_path):
    store = st.CellStore(tmp_path)
    attempts = []

    def boom():
        attempts.append(1)
        raise RuntimeError("cuda out of memory")

    assert store.run("bad", key(), boom) == "failed"
    assert len(attempts) == 1 + st.RETRIES == 3
    assert store.run("good", key(), lambda: {"ok": True}) == "done"
    assert [name for name, _ in store.failures] == ["bad"]
    assert "cuda out of memory" in store.failures[0][1]
    assert not (tmp_path / "bad.json").exists()


def test_a_cell_that_succeeds_on_the_retry_is_done_and_not_listed(tmp_path):
    store = st.CellStore(tmp_path)
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 2:
            raise RuntimeError("transient")
        return {"n": state["n"]}

    assert store.run("c", key(), flaky) == "done" and not store.failures


def test_keyboard_interrupt_is_not_swallowed(tmp_path):
    def stop():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        st.CellStore(tmp_path).run("c", key(), stop)


def test_the_result_must_be_json_serialisable_numpy_scalars_included(tmp_path):
    store = st.CellStore(tmp_path)
    store.run("c", key(), lambda: {"a": np.float32(0.5), "b": np.arange(2)})
    assert store.read("c")["result"] == {"a": 0.5, "b": [0, 1]}


def selection_file(tmp_path, **doc):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"decision": "adopt", "winner": "F1", **doc}))
    return path


def test_final_aborts_without_a_selection_file(tmp_path):
    with pytest.raises(st.SelectionMissingError):
        st.require_selection(tmp_path / "selection.json", "0" * 64)


def test_final_aborts_when_the_selection_sha_pin_does_not_match(tmp_path):
    path = selection_file(tmp_path)
    with pytest.raises(st.SelectionPinError):
        st.require_selection(path, "0" * 64)
    with pytest.raises(st.SelectionPinError):
        st.require_selection(path, "")


def test_the_selection_is_returned_with_its_sha_when_pinned(tmp_path):
    path = selection_file(tmp_path)
    sha = st.file_sha256(path)
    doc, got = st.require_selection(path, sha)
    assert got == sha and doc["winner"] == "F1"


def test_a_selection_without_a_decision_is_refused(tmp_path):
    path = tmp_path / "selection.json"
    path.write_text("{}")
    with pytest.raises(st.SelectionPinError):
        st.require_selection(path, st.file_sha256(path))


def test_the_val_lock_can_be_taken_once_only(tmp_path):
    lock = st.ValLock(tmp_path)
    assert not lock.taken()
    lock.acquire("a" * 64)
    assert lock.taken()
    with pytest.raises(st.ValAlreadyReadError):
        st.ValLock(tmp_path).acquire("a" * 64)
    assert json.loads((tmp_path / st.LOCK_NAME).read_text())["selection_sha256"] == "a" * 64


def test_the_reader_refuses_validation_and_train_arrays_for_selection(tmp_path):
    np.save(tmp_path / "val__full.npy", np.zeros((1, 2, 3)))
    reader = st.FeatureReader(tmp_path, forbidden=("val", "train"))
    with pytest.raises(st.ValAccessError):
        reader.load("val", "full")
    with pytest.raises(st.ValAccessError):
        reader.load("train", "full")


def test_the_reader_loads_the_allowed_splits_and_names_missing_files(tmp_path):
    np.save(tmp_path / "holdout__full.npy", np.ones((1, 2, 3)))
    reader = st.FeatureReader(tmp_path, forbidden=("val",))
    assert reader.load("holdout", "full").shape == (1, 2, 3)
    with pytest.raises(st.MissingFeaturesError):
        reader.load("holdout", "ring")
