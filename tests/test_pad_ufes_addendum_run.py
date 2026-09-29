"""scripts/pad_ufes_addendum.py end to end on synthetic features: no torch forward, no image, no git.

What is checked is what surrounds the backbones: every probe fit (s44 logreg and
the A71 v7.1 fits) runs under ``PROBE_BLAS_THREADS`` and the sidecar says so, A71
is the frozen v7.1 cell read on the D4 views of PAD, and the outputs are written.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from blas_spy import blas_threads
from env_sidecar import sidecar_env

from dermssl import pad_ufes as pu
from dermssl import pad_ufes_addendum as pa
from dermssl import pad_ufes_readings as pr
from dermssl import probe_threads, probes
from dermssl import v71_grid as vg
from dermssl import v71_protocol as vp
from scripts import pad_ufes_addendum as ad

N_CLASSES = len(pr.HAM_CLASSES)
TRAIN_IDS = [f"ISIC_{i:07d}" for i in range(35)]
N_PAD = 60
DIM = 3


class ProbeSpy:
    """Stands in for the s44 logreg: records the BLAS threads it ran under."""

    def __init__(self):
        self.seen: list[int] = []

    def __call__(self, x_train, y_train, x_val, n_classes, **kwargs):
        self.seen.append(blas_threads())
        raw = np.random.default_rng(len(self.seen)).random((len(x_val), n_classes))
        return raw / raw.sum(axis=1, keepdims=True)


class ViewFitSpy:
    """Stands in for ``v71_grid.fit_predict_views``: threads seen and what it returned."""

    def __init__(self):
        self.seen: list[int] = []
        self.returned: list[np.ndarray] = []

    def __call__(self, x_fit, y_fit, x_views, n_classes, class_weight):
        self.seen.append(blas_threads())
        raw = np.random.default_rng(len(self.seen)).random((len(x_views), x_views.shape[1], n_classes))
        self.returned.append(raw / raw.sum(axis=2, keepdims=True))
        return self.returned[-1]


class FakeBank:
    def __init__(self, weights_dir):
        self.states: list[str] = []

    def state(self, weights: str) -> str:
        self.states.append(weights)
        return weights

    def model(self, state: str) -> str:
        return state


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "BOOTSTRAP", {"n_resamples": 15, "rng_seed": 20260923})
    id2label = {image_id: k % N_CLASSES for k, image_id in enumerate(TRAIN_IDS)}
    ctx = {"split": {"train_ids": TRAIN_IDS, "val_ids": [], "id2label": id2label},
           "ids": {"train": TRAIN_IDS, "val": []}, "id2label": id2label,
           "classes": list(pr.HAM_CLASSES), "image_index": {}}
    cycle = ["bcc", "mel", "nv", "akiec", "bkl"]
    pad = {"ids": [f"PAT_{i}" for i in range(N_PAD)], "paths": {},
           "ham_labels": np.array([cycle[i % 5] for i in range(N_PAD)]),
           "groups": np.array([f"P{i // 3}" for i in range(N_PAD)]),
           "fst_bins": np.array([pu.FST_BINS[i % 4] for i in range(N_PAD)]),
           "fst": [[1, 2, 3, 4, None][i % 5] for i in range(N_PAD)], "manifest_sha256": "m" * 64}
    train = tmp_path / "train.npy"
    np.save(train, np.zeros((len(TRAIN_IDS), 2), dtype=np.uint8))
    s44_run = {"code_revision": "c" * 40, "exclude_ids_sha256": "e" * 64, "seed": 44}
    extracted: list[tuple] = []

    def extract_split(model, images, device, batch_size, tta):
        extracted.append((model, len(images), tta))
        rng = np.random.default_rng(len(extracted))
        views = 8 if tta else 1
        return {name: rng.normal(size=(views, len(images), DIM)) for name in ("cls", "attn")}

    monkeypatch.setattr(ad, "preflight", lambda args: ("r" * 40, "f" * 40))
    monkeypatch.setattr(ad.io, "load_context", lambda ns: ctx)
    monkeypatch.setattr(ad.io, "WeightBank", FakeBank)
    monkeypatch.setattr(ad.v71_val, "_guard_val", lambda args, c, w: {
        "exclude_ids_sha256": "e" * 64, "runs": {"A_s42": {"seed": 42}}})
    monkeypatch.setattr(ad.c1, "run_manifests", lambda specs: {
        role: dict(s44_run) for role in specs if role != pr.PUBLIC})
    monkeypatch.setattr(ad, "train_inputs_path", lambda inputs, ids: train)
    monkeypatch.setattr(ad.ev, "load_pad", lambda manifest, root: pad)
    monkeypatch.setattr(ad, "pad_inputs", lambda args, p: np.zeros((N_PAD, 2), dtype=np.uint8))
    monkeypatch.setattr(ad, "torch_device", lambda name: "cpu")
    monkeypatch.setattr(ad, "extract_split", extract_split)
    monkeypatch.setattr(ad.bank, "build_frozen_backbone", lambda *a: {"model": a[1]})
    monkeypatch.setattr(ad.bank, "extract_features", lambda model, ids, index, labels, *rest: (
        np.random.default_rng(len(ids)).normal(size=(len(ids), DIM)).astype(np.float32),
        np.arange(len(ids)) % N_CLASSES))
    probe_spy, view_spy = ProbeSpy(), ViewFitSpy()
    for name in probes.PROBES:
        monkeypatch.setitem(probes.PROBES, name, probe_spy)
    monkeypatch.setattr(vg, "fit_predict_views", view_spy)
    argv = ["--single-evaluation", "--probe-jobs", "1"]
    for flag in ("--manifest", "--pad-root", "--ham-root", "--split-index", "--holdout", "--exclude-ids-file",
                 "--weights-dir", "--v71-inputs", "--pad-cache"):
        argv += [flag, str(tmp_path / flag.strip("-"))]
    argv += ["--out", str(tmp_path / "res" / "pad_ufes_addendum_a71_s44.json"),
             "--plain-s44", "dinov2_vitb14:/v7/final_A_s44/w.pt", "--proposed-s44", "dinov2_vitb14:/v7/final_B_s44/w.pt"]
    return {"argv": argv, "out": tmp_path / "res" / "pad_ufes_addendum_a71_s44.json", "ctx": ctx,
            "probe_spy": probe_spy, "view_spy": view_spy, "extracted": extracted}


@pytest.mark.parametrize("pinned", [2, 3])
def test_every_probe_fit_runs_under_the_pinned_threads_and_the_sidecar_says_so(monkeypatch, world, pinned):
    monkeypatch.setattr(probe_threads, "PROBE_BLAS_THREADS", pinned)
    assert ad.main(world["argv"]) == 0
    assert world["probe_spy"].seen == [pinned] * 3  # public, A* s44, B* s44
    assert world["view_spy"].seen == [pinned] * 4  # A71 and public_A71, flat and balanced each
    assert sidecar_env(world["out"])["knobs"]["loky_inner_threads"] == pinned


def test_a71_is_the_frozen_cell_on_the_d4_views_of_pad(world):
    assert ad.main(world["argv"]) == 0
    saved = np.load(world["out"].with_suffix(".probas.npz"))
    assert set(pa.roles()) <= set(saved.files)
    # First fit of the first unit is A71's flat probe; la0.5 is post-hoc on it, averaged over the views.
    flat = world["view_spy"].returned[0]
    y_fit = np.array([world["ctx"]["id2label"][i] for i in TRAIN_IDS])
    prior = vp.class_prior(y_fit, N_CLASSES)
    expected = np.stack([vp.logit_adjust(v, prior, 0.5) for v in flat]).mean(axis=0)
    np.testing.assert_allclose(saved["A71"], expected)
    assert [(m, n, tta) for m, n, tta in world["extracted"]] == [
        ("A_soup_a1.0", len(TRAIN_IDS), False), ("A_soup_a1.0", N_PAD, True),
        ("public", len(TRAIN_IDS), False), ("public", N_PAD, True)]


def test_the_document_is_descriptive_and_names_its_provenance(world):
    assert ad.main(world["argv"]) == 0
    document = json.loads(world["out"].read_text(encoding="utf-8"))
    pa.assert_descriptive(document)
    assert set(document["readings"]["primary"]["arms"]) == set(pa.roles())
    assert document["protocols"][pa.A71] == pa.A71_CONFIG
    assert document["protocols"][pa.PUBLIC_A71] == pa.PUBLIC_A71_CONFIG
    assert document["selection_sha256"] == pa.SELECTION_SHA256
    assert document["registration_commit"] == "r" * 40 and document["code_revision"] == "f" * 40
    assert document["runs"]["plain_s44"]["seed"] == 44
    assert world["out"].with_suffix(".fst_table.md").is_file()


def test_other_ham_classes_abort_before_any_extraction(world):
    world["ctx"]["classes"] = ["a", "b"]
    with pytest.raises(SystemExit, match="clases"):
        ad.main(world["argv"])
    assert world["extracted"] == [] and world["probe_spy"].seen == []
