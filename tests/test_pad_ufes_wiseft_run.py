"""scripts/pad_ufes_wiseft.py end to end on synthetic features: no torch forward, no image, no git.

What is checked is what surrounds the backbones: public (alpha = 0) is extracted
and fitted once for the three curves, every probe fit runs under
``PROBE_BLAS_THREADS`` and the sidecar says so, and the outputs are written.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
from blas_spy import blas_threads
from env_sidecar import sidecar_env
from test_pad_ufes_wiseft import _grid

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_readings as pr
from dermssl import pad_ufes_wiseft as pw
from dermssl import probe_threads, probes
from scripts import pad_ufes_wiseft as ws

N_CLASSES = len(pr.HAM_CLASSES)
TRAIN_IDS = [f"ISIC_{i:07d}" for i in range(35)]
N_PAD = 60
DIM = 3


class ProbeSpy:
    """Stands in for the logreg: records the BLAS threads it ran under."""

    def __init__(self):
        self.seen: list[int] = []

    def __call__(self, x_train, y_train, x_val, n_classes, **kwargs):
        self.seen.append(blas_threads())
        raw = np.random.default_rng(int(abs(x_val).sum() * 1000) % 2**32).random((len(x_val), n_classes))
        return raw / raw.sum(axis=1, keepdims=True)


class FakeModel:
    def __init__(self, weights: str):
        self.weights = weights

    def to(self, device):
        return self


class FakeBank:
    def __init__(self, weights_dir):
        self.states: list[str] = []

    def state(self, weights: str) -> str:
        self.states.append(weights)
        return weights

    def model(self, state: str) -> FakeModel:
        return FakeModel(state)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "BOOTSTRAP", {"n_resamples": 10, "rng_seed": 20260923})
    id2label = {image_id: k % N_CLASSES for k, image_id in enumerate(TRAIN_IDS)}
    ctx = {"split": {"train_ids": TRAIN_IDS, "val_ids": [], "id2label": id2label},
           "ids": {"train": TRAIN_IDS, "val": []}, "id2label": id2label,
           "classes": list(pr.HAM_CLASSES), "image_index": {}}
    cycle = ["bcc", "mel", "nv", "akiec", "bkl"]
    pad_ids = [f"PAT_{i}" for i in range(N_PAD)]
    pad = {"ids": pad_ids, "paths": {},
           "ham_labels": np.array([cycle[i % 5] for i in range(N_PAD)]),
           "groups": np.array([f"P{i // 3}" for i in range(N_PAD)]),
           "fst_bins": np.array([pu.FST_BINS[i % 4] for i in range(N_PAD)]),
           "fst": [[1, 2, 3, 4, None][i % 5] for i in range(N_PAD)], "manifest_sha256": "m" * 64}
    extracted: list[tuple[str, int]] = []

    def extract_features(model, ids, index, labels, resolution, device, batch_size, num_workers):
        assert (resolution, num_workers) == (224, 0)
        extracted.append((model.weights, len(ids)))
        rng = np.random.default_rng(len(extracted))
        return rng.normal(size=(len(ids), DIM)).astype(np.float32), np.arange(len(ids)) % N_CLASSES

    grid = tmp_path / "holdout_grid.json"
    grid.write_text(json.dumps(_grid()), encoding="utf-8")
    first_pass = tmp_path / "pad_ufes_eval.probas.npz"
    monkeypatch.setattr(ws, "preflight", lambda args: ("r" * 40, "f" * 40))
    monkeypatch.setattr(ws.io, "load_context", lambda ns: ctx)
    monkeypatch.setattr(ws.io, "WeightBank", FakeBank)
    monkeypatch.setattr(ws.v71_val, "_guard_val", lambda args, c, w: {
        "exclude_ids_sha256": "e" * 64, "runs": {"A_s42": {"seed": 42}, "A_s43": {"seed": 43}}})
    monkeypatch.setattr(ws.ev, "load_pad", lambda manifest, root: pad)
    monkeypatch.setattr(ws, "torch_device", lambda name: "cpu")
    monkeypatch.setattr(ws.bank, "extract_features", extract_features)
    spy = ProbeSpy()
    for name in probes.PROBES:
        monkeypatch.setitem(probes.PROBES, name, spy)
    out = tmp_path / "res" / "pad_ufes_wiseft.json"
    argv = ["--single-evaluation", "--ham-grid", str(grid), "--first-pass-probas", str(first_pass)]
    for flag in ("--manifest", "--pad-root", "--ham-root", "--split-index", "--holdout", "--exclude-ids-file",
                 "--weights-dir"):
        argv += [flag, str(tmp_path / flag.strip("-"))]
    argv += ["--out", str(out)]
    np.savez_compressed(first_pass, ids=np.array(pad_ids),
                        **{r: np.full((N_PAD, N_CLASSES), 1 / N_CLASSES) for r in ("public", "plain_s42", "plain_s43")})
    return {"argv": argv, "out": out, "spy": spy, "extracted": extracted, "pad": pad, "first_pass": first_pass}


def test_public_is_extracted_once_and_every_point_once(world):
    assert ws.main(world["argv"]) == 0
    per_weights = Counter(weights for weights, _ in world["extracted"])
    assert per_weights == Counter({role: 2 for role in pw.roles()})  # C1 train and PAD, once each
    assert [n for _, n in world["extracted"][:2]] == [len(TRAIN_IDS), N_PAD]


@pytest.mark.parametrize("pinned", [2, 3])
def test_every_probe_fit_runs_under_the_pinned_threads_and_the_sidecar_says_so(monkeypatch, world, pinned):
    monkeypatch.setattr(probe_threads, "PROBE_BLAS_THREADS", pinned)
    assert ws.main(world["argv"]) == 0
    assert world["spy"].seen == [pinned] * len(pw.roles())
    assert sidecar_env(world["out"])["knobs"]["loky_inner_threads"] == pinned


def test_the_outputs_carry_every_role_the_verdicts_and_the_provenance(world):
    assert ws.main(world["argv"]) == 0
    saved = np.load(world["out"].with_suffix(".probas.npz"))
    assert set(pw.roles()) <= set(saved.files) and list(saved["ids"]) == world["pad"]["ids"]
    document = json.loads(world["out"].read_text(encoding="utf-8"))
    assert set(document["verdicts"]) == {"PAD-W1", "PAD-W2"}
    assert document["registration_commit"] == "r" * 40 and document["code_revision"] == "f" * 40
    assert document["runs"]["A_s43"]["seed"] == 43
    assert document["ham_holdout"]["curves"]["A_s43"]["available"] is False
    assert world["out"].with_suffix(".fst_table.md").is_file()


def test_the_three_shared_points_are_compared_with_the_first_pass(world):
    assert ws.main(world["argv"]) == 0
    agreement = json.loads(world["out"].read_text(encoding="utf-8"))["first_pass_agreement"]
    assert agreement["public"]["first_pass_role"] == "public"
    assert agreement["A_s42_a1.0"]["first_pass_role"] == "plain_s42"
    assert agreement["A_s43_a1.0"]["first_pass_role"] == "plain_s43"
    assert all(entry["max_abs_diff"] > 0 for entry in agreement.values())


def test_a_first_pass_on_other_images_aborts_before_any_extraction(world):
    np.savez_compressed(world["first_pass"], ids=np.array(["X"] * N_PAD))
    with pytest.raises(SystemExit, match="primera pasada"):
        ws.main(world["argv"])
    assert world["extracted"] == []
