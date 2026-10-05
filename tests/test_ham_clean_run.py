"""End to end of scripts/ham_clean_run.py on a fake HAM10000 and tiny random DINOv2s, CPU only.

The rehearsal and the (monkeypatched-compute) val read run the real code path: guards, lock, extraction,
probes, thresholds from the ORIGINAL selection, paired contrasts grouped by lesion with MCC, JSON + sidecar.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from env_sidecar import sidecar_env  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import ham_clean_run as run  # noqa: E402
import v71_io  # noqa: E402

from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import e3_store as st  # noqa: E402
from dermssl import final_eval as fe  # noqa: E402
from dermssl import ham_clean as hc  # noqa: E402
from dermssl import probe_threads  # noqa: E402
from dermssl import selection_holdout as sh  # noqa: E402
from dermssl import v71_plan as plan  # noqa: E402
from dermssl.isic_eval import protocol as pr  # noqa: E402
from tests.release_helpers import tiny_hub_model  # noqa: E402
from tests.test_v71_run import _fake_ham  # noqa: E402

THRESHOLDS = {"spec80": 0.2, "spec90": 0.3, "sens95": 0.05}
PUBLIC_WINNER = "public@224|cls4|la0.5|tta"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_weights(root: Path, exclude_sha: str) -> None:
    for seed, name in enumerate(("final_B_s42", "final_A_s42", "final_A_s43"), start=21):
        (root / name).mkdir(parents=True)
        torch.save(tiny_hub_model(seed=seed, depth=4).state_dict(), root / name / v71_io.WEIGHTS_FILE)
        (root / name / "run.json").write_text(json.dumps(
            {"code_revision": "f1c2053", "exclude_ids_sha256": exclude_sha, "seed": int(name[-2:])}), encoding="utf-8")


def _reference(directory: Path, name: str, split: str, ids, y, entries: dict) -> None:
    rng = np.random.RandomState(7)
    arrays = {label: rng.dirichlet(np.ones(7), len(ids)) for label in (*hc.ORIGINAL, *hc.PUBLIC)}
    (directory / f"{name}.json").write_text(json.dumps(
        {"split": split, "eval_ids_sha256": si.ids_sha256(list(ids)), "entries": entries, "code_revision": "ref"}),
        encoding="utf-8")
    np.savez_compressed(directory / f"{name}.probas.npz", y_true=y, **arrays)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("hamclean")
    ham = root / "ham"
    ham.mkdir()
    _fake_ham(ham)
    from c1_build_val_index import rebuild_split

    split, meta = rebuild_split(ham)
    (root / "c1_split_index.json").write_text(json.dumps(
        {"train_ids": split["train_ids"], "val_ids": split["val_ids"]}), encoding="utf-8")
    rows = meta.set_index("image_id")
    labels = [split["cls2idx"][d] for d in rows.loc[split["train_ids"], "dx"]]
    fit, hold = sh.grouped_holdout(split["train_ids"], labels, list(rows.loc[split["train_ids"], "lesion_id"]),
                                   forbidden=split["val_ids"])
    (root / "selection_holdout.json").write_text(json.dumps({
        "fit_ids": fit, "holdout_ids": hold, "fit_ids_sha256": si.ids_sha256(fit),
        "holdout_ids_sha256": si.ids_sha256(hold)}), encoding="utf-8")
    v71 = root / "v71"
    v71.mkdir()
    winners = {"A": pr.A71, "B": pr.B71, "public": PUBLIC_WINNER}
    (v71 / "selection.json").write_text(json.dumps(
        {"winners": winners, "holdout_ids_sha256": si.ids_sha256(hold)}), encoding="utf-8")
    (v71 / "holdout_grid.json").write_text(json.dumps(
        {"holdout_ids_sha256": si.ids_sha256(hold), "rows": {c: {"thresholds": THRESHOLDS} for c in (pr.A71, pr.B71)}}),
        encoding="utf-8")
    entries = plan.val_entries({"winners": winners})
    y_of = lambda ids: np.array([split["cls2idx"][rows.loc[i, "dx"]] for i in ids])  # noqa: E731
    _reference(v71, "v71_final_eval", "val", split["val_ids"], y_of(split["val_ids"]), entries)
    _reference(v71, "v71_final_eval_holdout_rehearsal", "holdout", hold, y_of(hold), entries)
    _clean_weights(root / "clean", pr.POOL_EXCLUDE_SHA256["clean"])
    return root


@pytest.fixture(autouse=True)
def _tiny_hub(monkeypatch):
    monkeypatch.setattr(v71_io, "hub_model", lambda arch, pretrained: tiny_hub_model(seed=0 if pretrained else 5, depth=4))
    monkeypatch.setattr(run, "revision", lambda args: "test")


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    from threadpoolctl import threadpool_limits

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(1), pytest.MonkeyPatch.context() as mp:
        mp.setattr(probe_threads, "PROBE_BLAS_THREADS", 1)
        yield
    torch.set_num_threads(threads)


def _argv(world: Path, out: Path, stage: str, *extra: str, weights: str = "clean") -> list[str]:
    return ["--ham-root", str(world / "ham"), "--image-dir", str(world / "ham"),
            "--split-index", str(world / "c1_split_index.json"), "--holdout", str(world / "selection_holdout.json"),
            "--weights-dir", str(world / weights), "--out-dir", str(out), "--v71-dir", str(world / "v71"),
            "--stage", stage, "--ci-resamples", "3", "--probe-jobs", "1", "--decode-workers", "2",
            "--batch-size", "16", "--device", "cpu", *extra]


def _canonical(monkeypatch) -> None:
    monkeypatch.setattr(run, "cuda_available", lambda: True)
    monkeypatch.setattr(hc, "compute_block", lambda d, b, rehearsal: {"device": "cuda", "batch_size": 64,
                                                                      "autocast": "cuda", "canonical": True})
    monkeypatch.setattr(fe, "check_val_protocol", lambda train, val: None)


@pytest.fixture(scope="module")
def rehearsed(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("x") / "ham_clean_rehearsal"
    before = {p.name: _sha(p) for p in (world / "v71").iterdir()}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(v71_io, "hub_model", lambda arch, pretrained: tiny_hub_model(seed=0 if pretrained else 5, depth=4))
        mp.setattr(run, "revision", lambda args: "test")
        for stage in ("extract", "val"):
            assert run.main(_argv(world, out, stage, "--rehearsal")) == 0
    assert before == {p.name: _sha(p) for p in (world / "v71").iterdir()}  # the original artifacts are read only
    return out


def test_rehearsal_never_opens_val_and_takes_no_lock(rehearsed):
    names = [p.name for p in (rehearsed / "inputs").iterdir()]
    assert names and all(n.startswith(("fit_", "holdout_")) for n in names)
    assert not (rehearsed / st.LOCK_NAME).exists()


def test_rehearsal_result_declares_label_assumption_and_discard(rehearsed):
    doc = json.loads((rehearsed / hc.REHEARSAL_JSON).read_text(encoding="utf-8"))
    assert doc["label"] == hc.LABEL and doc["assumptions"] == [hc.THRESHOLD_ASSUMPTION]
    assert doc["rehearsal"] is True and doc["discard"] is True and doc["split"] == "holdout"
    assert doc["entries"] == {"B71_clean": pr.B71, "A71_clean": pr.A71}
    assert doc["bootstrap"]["grouped_by"] == "lesion_id" and doc["bootstrap"]["rng_seed"] == 20260923
    assert doc["pool_check"]["variant"] == "clean" and doc["pool_check"]["verified"] is True
    assert set(doc["weights_sha256"]) == {"B_s42", "A_s42", "A_s43"}


def test_the_five_contrasts_carry_every_metric_with_mcc_last(rehearsed):
    doc = json.loads((rehearsed / hc.REHEARSAL_JSON).read_text(encoding="utf-8"))
    pairs = [(r["arm"], r["control"]) for r in doc["paired"]]
    assert [p for k, p in enumerate(pairs) if p not in pairs[:k]] == list(hc.CONTRASTS)
    for pair in hc.CONTRASTS:
        metrics = [r["metric"] for r in doc["paired"] if (r["arm"], r["control"]) == pair]
        assert metrics == list(hc.CONTRAST_METRICS) and metrics[-1] == "mcc"


def test_thresholds_are_the_original_holdout_ones(rehearsed):
    doc = json.loads((rehearsed / hc.REHEARSAL_JSON).read_text(encoding="utf-8"))
    assert doc["results"]["B71_clean"]["holdout_thresholds"] == THRESHOLDS
    assert "mel_sens_at_thr_spec80" in doc["results"]["A71_clean"]


def test_result_and_extract_have_their_environment_sidecars(rehearsed):
    for name in (hc.REHEARSAL_JSON, hc.EXTRACT_JSON):
        env = sidecar_env(rehearsed / name)
        assert env["torch"]["loaded"] is True and env["knobs"]["batch_size"] == 16
    npz = np.load(rehearsed / hc.REHEARSAL_JSON.replace(".json", ".probas.npz"))
    assert {"y_true", "B71_clean", "A71_clean", "B71", "A71", "public_B71", "public_A71"} <= set(npz.files)


def test_a_second_extract_in_the_same_out_dir_is_refused(world, rehearsed):
    with pytest.raises(SystemExit, match="ya existe"):
        run.main(_argv(world, rehearsed, "extract", "--rehearsal"))


def test_flags_demand_exactly_one_mode_and_a_rehearsal_out_dir(world, tmp_path):
    with pytest.raises(SystemExit, match="exactamente uno"):
        run.main(_argv(world, tmp_path / "o", "extract"))
    with pytest.raises(SystemExit, match="exactamente uno"):
        run.main(_argv(world, tmp_path / "o", "extract", "--rehearsal", "--single-evaluation"))
    with pytest.raises(SystemExit, match="rehearsal"):
        run.main(_argv(world, tmp_path / "plain", "extract", "--rehearsal"))


def test_real_val_refuses_non_canonical_compute_before_lock_and_images(world, tmp_path, monkeypatch):
    monkeypatch.setattr(fe, "check_val_protocol", lambda train, val: None)
    out = tmp_path / "real"
    with pytest.raises(SystemExit, match="canonico"):
        run.main(_argv(world, out, "extract", "--single-evaluation"))
    assert not (out / st.LOCK_NAME).exists() and not (out / "inputs").exists()


def test_a_mislabelled_pool_is_refused(world, tmp_path):
    weights = tmp_path / "w"
    _clean_weights(weights, pr.POOL_EXCLUDE_SHA256["original"])
    argv = _argv(world, tmp_path / "ham_clean_rehearsal", "extract", "--rehearsal")
    argv[argv.index("--weights-dir") + 1] = str(weights)
    with pytest.raises(SystemExit, match="exclude_ids_sha256"):
        run.main(argv)
    assert not (tmp_path / "ham_clean_rehearsal" / "inputs").exists()


def test_an_existing_lock_aborts_before_any_val_image(world, tmp_path, monkeypatch):
    _canonical(monkeypatch)
    out = tmp_path / "locked"
    st.ValLock(out).acquire("x")
    with pytest.raises(SystemExit, match="val ya se leyo"):
        run.main(_argv(world, out, "extract", "--single-evaluation"))
    assert not (out / "inputs").exists()


def test_val_stage_without_the_lock_is_refused(world, tmp_path):
    with pytest.raises(SystemExit, match="extract"):
        run.main(_argv(world, tmp_path / "none", "val", "--single-evaluation"))


def test_the_real_val_path_takes_the_lock_pairs_on_val_ids_and_writes_everything(world, tmp_path, monkeypatch):
    _canonical(monkeypatch)
    out = tmp_path / "real_val"
    assert run.main(_argv(world, out, "extract", "--single-evaluation")) == 0
    assert (out / st.LOCK_NAME).exists()
    assert run.main(_argv(world, out, "val", "--single-evaluation")) == 0
    doc = json.loads((out / "ham_clean_val.json").read_text(encoding="utf-8"))
    assert doc["split"] == "val" and doc["rehearsal"] is False and "discard" not in doc
    assert doc["compute"]["canonical"] is True and doc["n_eval"] == len(np.load(out / "ham_clean_val.probas.npz")["y_true"])
    assert _sha(out / "ham_clean_val.probas.npz") == doc["probas_sha256"]
    assert sidecar_env(out / "ham_clean_val.json")["schema"] == 1
    with pytest.raises(SystemExit, match="ya existe"):
        run.main(_argv(world, out, "val", "--single-evaluation"))


def test_a_reference_on_other_ids_cannot_be_paired(world, tmp_path, monkeypatch):
    _canonical(monkeypatch)
    out = tmp_path / "other_ids"
    assert run.main(_argv(world, out, "extract", "--single-evaluation")) == 0
    other = tmp_path / "ref"
    other.mkdir()
    for name in ("v71_final_eval.json", "v71_final_eval.probas.npz"):
        (other / name).write_bytes((world / "v71" / name).read_bytes())
    doc = json.loads((other / "v71_final_eval.json").read_text(encoding="utf-8"))
    doc["eval_ids_sha256"] = "0" * 64
    (other / "v71_final_eval.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="ids"):
        run.main(_argv(world, out, "val", "--single-evaluation", "--reference-dir", str(other)))
    assert not (out / "ham_clean_val.json").exists()


# --- preflight: nothing that can be checked beforehand may burn the single val read -------------------------

def _clone_weights(world: Path, tmp_path: Path) -> Path:
    root = tmp_path / "w"
    shutil.copytree(world / "clean", root)
    return root


def _real_val_argv(world: Path, tmp_path: Path, weights: Path, *extra: str) -> tuple[list[str], Path]:
    out = tmp_path / "pre"
    argv = _argv(world, out, "extract", "--single-evaluation", *extra)
    argv[argv.index("--weights-dir") + 1] = str(weights)
    return argv, out


def _refused(world, tmp_path, monkeypatch, match, mutate=None, **patches):
    _canonical(monkeypatch)
    weights = _clone_weights(world, tmp_path)
    if mutate:
        mutate(weights)
    for name, value in patches.items():
        monkeypatch.setattr(run, name, value)
    argv, out = _real_val_argv(world, tmp_path, weights)
    with pytest.raises(SystemExit, match=match):
        run.main(argv)
    assert not (out / st.LOCK_NAME).exists() and not (out / "inputs").exists()


def _edit_run(folder: str, **fields):
    def mutate(root: Path) -> None:
        path = root / folder / "run.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **fields}), encoding="utf-8")
    return mutate


def test_a_missing_final_aborts_with_the_lock_untaken(world, tmp_path, monkeypatch):
    _refused(world, tmp_path, monkeypatch, "final_A_s43", lambda r: (r / "final_A_s43" / v71_io.WEIGHTS_FILE).unlink())


def test_a_final_that_does_not_load_strictly_aborts_with_the_lock_untaken(world, tmp_path, monkeypatch):
    _refused(world, tmp_path, monkeypatch, "carga estricta",
             lambda r: torch.save({}, r / "final_B_s42" / v71_io.WEIGHTS_FILE))


def test_a_val_image_without_file_aborts_with_the_lock_untaken(world, tmp_path, monkeypatch):
    real = run.io.load_context

    def without_one(args):
        ctx = real(args)
        ctx["image_index"] = {k: v for k, v in ctx["image_index"].items() if k != ctx["ids"]["val"][0]}
        return ctx

    monkeypatch.setattr(run.io, "load_context", without_one)
    _refused(world, tmp_path, monkeypatch, "sin archivo")


def test_no_cuda_aborts_with_the_lock_untaken(world, tmp_path, monkeypatch):
    _canonical(monkeypatch)
    weights = _clone_weights(world, tmp_path)
    monkeypatch.setattr(run, "cuda_available", lambda: False)
    argv, out = _real_val_argv(world, tmp_path, weights)
    with pytest.raises(SystemExit, match="CUDA"):
        run.main(argv)
    assert not (out / st.LOCK_NAME).exists()


def test_a_final_with_the_wrong_seed_aborts(world, tmp_path, monkeypatch):
    _refused(world, tmp_path, monkeypatch, "seed", _edit_run("final_A_s43", seed=42))


def test_copied_folders_with_identical_weights_abort(world, tmp_path, monkeypatch):
    def copy(root: Path) -> None:
        shutil.copy(root / "final_A_s42" / v71_io.WEIGHTS_FILE, root / "final_A_s43" / v71_io.WEIGHTS_FILE)
    _refused(world, tmp_path, monkeypatch, "identic", copy)


def test_b_and_a_must_come_from_the_same_code_revision(world, tmp_path, monkeypatch):
    _refused(world, tmp_path, monkeypatch, "code_revision", _edit_run("final_B_s42", code_revision="deadbee"))
