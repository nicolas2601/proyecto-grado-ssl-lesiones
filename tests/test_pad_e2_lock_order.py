"""The target_test lock is taken as late as possible: it covers only the label read and the metrics."""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_pad_e2 import _manifest_rows, _split

from dermssl import pad_e2 as e2
from dermssl import pad_target_split as ts

N_RES = 20
FP = {"backbone_sha256": {"D1": "a" * 64}, "manifest_sha256": "m" * 64}


class _Predict:
    def __init__(self, fail: bool = False):
        self.calls, self.fail = 0, fail

    def __call__(self, ids, paths):
        self.calls += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        rng = np.random.RandomState(0)
        return {a: rng.dirichlet(np.ones(7), size=len(ids)) for a in e2.ARMS}


@pytest.fixture
def env(tmp_path):
    rows = _manifest_rows(tmp_path)
    return tmp_path, rows, _split(tmp_path, rows), tmp_path / "out.json"


def _run(env, predict, **kw):
    tmp, rows, split, out = env
    return e2.run_single_evaluation(split, rows, tmp, predict, out, n_resamples=N_RES, fingerprint=FP, **kw)


class TestLockWindow:
    def test_a_failure_during_extraction_leaves_no_lock_and_no_frozen_files(self, env):
        with pytest.raises(RuntimeError, match="CUDA"):
            _run(env, _Predict(fail=True))
        assert not (env[0] / ts.LOCK_NAME).exists()
        assert not list(env[0].glob("out.probas*")) and not list(env[0].glob("*.tmp"))

    def test_a_failure_right_after_taking_the_lock_leaves_it(self, env, monkeypatch):
        monkeypatch.setattr(e2, "evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            _run(env, _Predict())
        assert (env[0] / ts.LOCK_NAME).is_file()

    def test_the_frozen_predictions_exist_before_the_lock_and_the_labels(self, env):
        tmp, rows, split, out = env
        seen = {}

        class Spy(dict):
            def __getitem__(self, key):
                if key == "ham_label":
                    seen.setdefault("at_first_label", (out.with_suffix(".probas.npz").is_file(),
                                                       out.with_suffix(".frozen.json").is_file(),
                                                       (tmp / ts.LOCK_NAME).is_file()))
                return super().__getitem__(key)

        e2.run_single_evaluation(split, [Spy(r) for r in rows], tmp, _Predict(), out, n_resamples=N_RES,
                                 fingerprint=FP)
        assert seen["at_first_label"] == (True, True, True)  # frozen, registered, THEN locked, THEN labels

    def test_the_lock_is_not_taken_while_the_predictions_are_computed(self, env):
        tmp = env[0]

        def predict(ids, paths):
            assert not (tmp / ts.LOCK_NAME).exists()
            return _Predict()(ids, paths)

        _run(env, predict)
        assert (tmp / ts.LOCK_NAME).is_file()

    def test_the_frozen_file_is_written_atomically(self, env, monkeypatch):
        real_replace, targets = e2.os.replace, []
        monkeypatch.setattr(e2.os, "replace", lambda a, b: targets.append(str(b)) or real_replace(a, b))
        _run(env, _Predict())
        assert any(t.endswith(".probas.npz") for t in targets) and any(t.endswith(".frozen.json") for t in targets)


class TestRelaunch:
    def _die_at_the_lock(self, env, monkeypatch):
        def refuse(self, sha):
            raise RuntimeError("killed")
        monkeypatch.setattr(ts.TargetTestLock, "acquire", refuse)
        with pytest.raises(RuntimeError, match="killed"):
            _run(env, _Predict())
        monkeypatch.undo()

    def test_after_dying_before_the_lock_the_frozen_predictions_are_reused(self, env, monkeypatch):
        self._die_at_the_lock(env, monkeypatch)
        again = _Predict()
        doc = _run(env, again)
        assert again.calls == 0 and (env[0] / ts.LOCK_NAME).is_file() and doc["probas_sha256"]

    def test_after_a_failed_extraction_a_relaunch_recomputes_and_finishes(self, env):
        with pytest.raises(RuntimeError):
            _run(env, _Predict(fail=True))
        again = _Predict()
        _run(env, again)
        assert again.calls == 1 and (env[0] / ts.LOCK_NAME).is_file()

    def test_a_tampered_frozen_file_is_recomputed(self, env, monkeypatch):
        self._die_at_the_lock(env, monkeypatch)
        npz = env[3].with_suffix(".probas.npz")
        npz.write_bytes(npz.read_bytes() + b"x")
        again = _Predict()
        _run(env, again)
        assert again.calls == 1

    def test_other_weights_or_manifest_are_recomputed(self, env, monkeypatch):
        self._die_at_the_lock(env, monkeypatch)
        again = _Predict()
        tmp, rows, split, out = env
        e2.run_single_evaluation(split, rows, tmp, again, out, n_resamples=N_RES,
                                 fingerprint={**FP, "backbone_sha256": {"D1": "b" * 64}})
        assert again.calls == 1

    def test_the_reused_run_reports_the_same_document_as_a_fresh_one(self, tmp_path):
        docs = []
        for name, die in (("fresh", False), ("reused", True)):
            sub = tmp_path / name
            sub.mkdir()
            rows = _manifest_rows(sub)
            split = _split(sub, rows)
            out = sub / "out.json"
            if die:
                with pytest.MonkeyPatch.context() as mp:
                    mp.setattr(ts.TargetTestLock, "acquire", lambda self, sha: (_ for _ in ()).throw(RuntimeError("k")))
                    with pytest.raises(RuntimeError):
                        e2.run_single_evaluation(split, rows, sub, _Predict(), out, n_resamples=N_RES, fingerprint=FP)
            docs.append(e2.run_single_evaluation(split, rows, sub, _Predict(), out, n_resamples=N_RES, fingerprint=FP))
        for doc in docs:
            doc.pop("probas_sha256")  # the npz container embeds a timestamp; the content is compared below
        assert json.dumps(docs[0], sort_keys=True) == json.dumps(docs[1], sort_keys=True)

    def test_after_a_success_a_relaunch_aborts_before_predicting(self, env):
        _run(env, _Predict())
        again = _Predict()
        with pytest.raises(ts.TargetTestAlreadyReadError):
            _run(env, again)
        assert again.calls == 0
