"""T7 training (docs/27): A2's recipe plus dark oversampling and darkening, guarded by the registration.

Same tiny fake DINOv2 and CPU pool as test_s4_unet_train. What is pinned:
with k = 1 and p = 0 the T7 path trains A2's exact weights; the record names
the T7 changes and nothing else moves; the registration, the manifest, the r0
pseudo-masks and the cache manifest are checked before training.
"""

from __future__ import annotations

import csv
import hashlib
import json

import numpy as np
import pytest

import seg_paths  # noqa: F401

torch = pytest.importorskip("torch")
from ckpt_io import load_checkpoint  # noqa: E402
from dermssl import luminance_sampling as ls  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from s4_fakes import FakeVit  # noqa: E402
from scripts import s4_unet_train as st  # noqa: E402
from test_s4_unet_train import COUNT, _argv, _pool  # noqa: E402

loop = st.loop
pytestmark = pytest.mark.torch
FST_COLUMNS = ["image_id", "fst_bucket", "tone_tercile", "status", "L_median", "b_median", "ita", "n_skin_px"]
LABELS = ("dark", "medium", "light", "unavailable")
GATE = {"addendum_required": False, "rows_sha256": "f" * 64}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def t7_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "load_backbone", lambda arch, source: FakeVit())
    monkeypatch.setattr(st.rp, "clean_revision", lambda repo: "c0ffee")
    cache, masks = _pool(tmp_path)
    manifest = tmp_path / "fst.csv"
    with open(manifest, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FST_COLUMNS)
        writer.writerows([f"ISIC_{i}", "", LABELS[i % 4], "ok", "70", "10", "40", "100"] for i in range(COUNT))
    monkeypatch.setattr(ls, "check_registration", lambda repo: "r" * 40)
    monkeypatch.setattr(ls, "check_teacher_gate", lambda repo: dict(GATE))
    monkeypatch.setattr(ls, "MANIFEST_SHA256", _sha(manifest))
    monkeypatch.setattr(ls, "PSEUDO_R0_SHA256", _sha(masks))
    monkeypatch.setattr(ls, "CACHE_MANIFEST_SHA256", _sha(cache / "manifest.csv"))
    return (cache, masks), manifest


def _t7_argv(pool, manifest, out, *extra):
    argv = _argv(pool, out, "--tercile-manifest", str(manifest), *extra)
    argv[argv.index("--variant") + 1] = "A2-T7"
    return argv


def _weights(path):
    return load_checkpoint(path)["model"]


def _same(a, b) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_k1_and_p0_train_exactly_a2s_weights(t7_pool, tmp_path, monkeypatch):
    pool, manifest = t7_pool
    st.main(_argv(pool, tmp_path / "a2"))
    monkeypatch.setitem(st.ga.DARKEN_DEFAULTS, "probability", 0.0)  # p = 0 is not a CLI option
    st.main(_t7_argv(pool, manifest, tmp_path / "t7", "--no-oversample"))
    assert _same(_weights(tmp_path / "a2" / "last.pt"), _weights(tmp_path / "t7" / "last.pt"))


def test_the_registered_arm_moves_the_weights_and_records_only_its_changes(t7_pool, tmp_path):
    pool, manifest = t7_pool
    st.main(_argv(pool, tmp_path / "a2"))
    st.main(_t7_argv(pool, manifest, tmp_path / "t7"))
    assert not _same(_weights(tmp_path / "a2" / "last.pt"), _weights(tmp_path / "t7" / "last.pt"))
    a2 = json.loads((tmp_path / "a2" / "run.json").read_text(encoding="utf-8"))
    t7 = json.loads((tmp_path / "t7" / "run.json").read_text(encoding="utf-8"))
    for key in ("pool", "pseudo_masks", "backbone", "schedule", "seed", "training"):
        assert t7[key] == a2[key], key
    assert t7["variant"] == "A2-T7"
    train = [i for i in range(COUNT) if i not in (3, 5, 7)]
    assert t7["t7"] == {"registration_commit": "r" * 40, "tercile_manifest_sha256": ls.MANIFEST_SHA256,
                        "oversample_k": 2, "darken": {"probability": 0.5, "gamma_range": [1.0, 2.2],
                                                      "brightness_range": [0.85, 1.0],
                                                      "contrast_range": [0.85, 1.0], "terciles": ["medium", "light"]},
                        "teacher_check": GATE,
                        "n_dark_train_confident": sum(1 for i in train if LABELS[i % 4] == "dark"),
                        "n_darken_eligible_train_confident": sum(1 for i in train
                                                                 if LABELS[i % 4] in ("medium", "light"))}


def test_the_augmentation_only_arm_keeps_a2s_batches(t7_pool, tmp_path):
    pool, manifest = t7_pool
    st.main(_t7_argv(pool, manifest, tmp_path / "b", "--no-oversample"))
    record = json.loads((tmp_path / "b" / "run.json").read_text(encoding="utf-8"))
    assert record["t7"]["oversample_k"] == 1 and record["t7"]["darken"]["probability"] == 0.5


def test_t7_needs_the_tercile_manifest(t7_pool, tmp_path):
    pool, _ = t7_pool
    argv = _argv(pool, tmp_path / "x")
    argv[argv.index("--variant") + 1] = "A2-T7"
    with pytest.raises(SystemExit, match="--tercile-manifest"):
        st.main(argv)


@pytest.mark.parametrize("constant", ["MANIFEST_SHA256", "PSEUDO_R0_SHA256", "CACHE_MANIFEST_SHA256"])
def test_t7_refuses_unregistered_inputs(t7_pool, tmp_path, monkeypatch, constant):
    pool, manifest = t7_pool
    monkeypatch.setattr(ls, constant, "0" * 64)
    with pytest.raises(SystemExit, match="ABORT"):
        st.main(_t7_argv(pool, manifest, tmp_path / "x"))
    assert not (tmp_path / "x" / "run.json").exists()


def test_t7_refuses_when_the_registration_fails(t7_pool, tmp_path, monkeypatch):
    pool, manifest = t7_pool

    def refuse(repo):
        raise pg.RegistrationError("el bloque registrado cambio")
    monkeypatch.setattr(ls, "check_registration", refuse)
    with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
        st.main(_t7_argv(pool, manifest, tmp_path / "x"))


def test_resuming_with_another_oversampling_factor_is_refused(t7_pool, tmp_path, monkeypatch):
    pool, manifest = t7_pool
    out = tmp_path / "run"
    monkeypatch.setattr(loop, "after_epoch", lambda epoch: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        st.main(_t7_argv(pool, manifest, out))
    with pytest.raises(SystemExit, match="t7"):
        st.main(_t7_argv(pool, manifest, out, "--no-oversample"))


def test_seed_43_is_the_a2_recipe_with_another_seed(t7_pool, tmp_path):
    pool, _ = t7_pool
    st.main(_argv(pool, tmp_path / "s43", "--seed", "43"))
    record = json.loads((tmp_path / "s43" / "run.json").read_text(encoding="utf-8"))
    assert record["variant"] == "A2" and record["seed"] == 43 and "t7" not in record
    assert np.isfinite(record["result"]["total_seconds"])


@pytest.mark.parametrize("extra", [("--dark-oversample", "3"), ("--dark-oversample", "1"), ("--darken-p", "0.4")])
def test_unregistered_factors_or_probabilities_cannot_be_launched(t7_pool, tmp_path, extra):
    pool, manifest = t7_pool
    with pytest.raises(SystemExit):
        st.main(_t7_argv(pool, manifest, tmp_path / "x", *extra))
    assert not (tmp_path / "x" / "run.json").exists()


def test_t7_refuses_when_the_teacher_gate_is_closed(t7_pool, tmp_path, monkeypatch):
    pool, manifest = t7_pool

    def closed(repo):
        raise ls.TeacherGateError("falta la adenda del techo del maestro")
    monkeypatch.setattr(ls, "check_teacher_gate", closed)
    with pytest.raises(SystemExit, match="ABORT: falta la adenda"):
        st.main(_t7_argv(pool, manifest, tmp_path / "x"))
    assert not (tmp_path / "x" / "run.json").exists()


def test_the_t7_arms_only_train_with_seed_42(t7_pool, tmp_path):
    pool, manifest = t7_pool
    with pytest.raises(SystemExit, match="semilla"):
        st.main(_t7_argv(pool, manifest, tmp_path / "x", "--seed", "43"))
    assert not (tmp_path / "x" / "run.json").exists()


def test_resuming_under_another_code_revision_is_refused(t7_pool, tmp_path, monkeypatch):
    pool, manifest = t7_pool
    out = tmp_path / "run"
    monkeypatch.setattr(loop, "after_epoch", lambda epoch: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        st.main(_t7_argv(pool, manifest, out))
    monkeypatch.setattr(loop, "after_epoch", lambda epoch: None)
    monkeypatch.setattr(st.rp, "clean_revision", lambda repo: "beef")
    with pytest.raises(SystemExit, match="code_revision"):
        st.main(_t7_argv(pool, manifest, out))


@pytest.mark.parametrize("extra", [("--no-oversample",), ("--tercile-manifest", "m.csv")])
def test_t7_flags_are_refused_on_other_variants(t7_pool, tmp_path, extra):
    pool, _ = t7_pool
    with pytest.raises(SystemExit, match="A2-T7"):
        st.main(_argv(pool, tmp_path / "x", "--seed", "43", *extra))
    assert not (tmp_path / "x" / "run.json").exists()
