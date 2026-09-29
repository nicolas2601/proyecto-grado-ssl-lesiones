"""End to end on a fake HAM10000 and tiny random DINOv2s, CPU only.

The three stages run for real (pre-test, holdout grid, selection, the val
stage in its holdout rehearsal) so the wiring is proven before the pod: the
files each stage promises exist, a relaunch skips finished units, the
pre-test aborts on a WiSE endpoint that is not the final, and val refuses to
run without its guards.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from env_sidecar import sidecar_env  # noqa: E402

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import v71_io  # noqa: E402
import v71_run  # noqa: E402

from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import selection_holdout as sh  # noqa: E402
from tests.release_helpers import tiny_hub_model  # noqa: E402

CLASSES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
WEIGHTS = ("public", "A_s42_a1.0", "A_soup_a0.6", "B_s42_a1.0")


def _fake_ham(root: Path) -> None:
    from PIL import Image

    rng = np.random.RandomState(0)
    rows = ["image_id,lesion_id,dx"]
    for k in range(7 * 12 + 30):
        cls = k % 7 if k < 7 * 12 else CLASSES.index("mel")  # mel-rich: every resample holds mel
        image_id = f"ISIC_{k:07d}"
        rows.append(f"{image_id},HAM_{k:07d},{CLASSES[cls]}")
        pixels = rng.randint(0, 60, (45, 60, 3)) + 25 * cls
        Image.fromarray(pixels.astype(np.uint8)).save(root / f"{image_id}.jpg", quality=95)
    (root / "HAM10000_metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _weights(root: Path) -> None:
    for seed, folder in enumerate(v71_io.FINAL_DIRS.values(), start=11):
        (root / folder).mkdir(parents=True)
        torch.save(tiny_hub_model(seed=seed, depth=4).state_dict(), root / folder / v71_io.WEIGHTS_FILE)
        (root / folder / "run.json").write_text(json.dumps(
            {"code_revision": "f1c2053", "exclude_ids_sha256": "e" * 64, "seed": 42}), encoding="utf-8")


@pytest.fixture(scope="module")
def pod(tmp_path_factory):
    root = tmp_path_factory.mktemp("v71")
    ham = root / "ham"
    ham.mkdir()
    _fake_ham(ham)
    _weights(root / "v7")
    from c1_build_val_index import rebuild_split

    split, meta = rebuild_split(ham)
    (root / "c1_split_index.json").write_text(json.dumps(
        {"train_ids": split["train_ids"], "val_ids": split["val_ids"]}), encoding="utf-8")
    rows = meta.set_index("image_id").loc[split["train_ids"]]
    fit, hold = sh.grouped_holdout(split["train_ids"], [split["cls2idx"][d] for d in rows.dx],
                                   list(rows.lesion_id), forbidden=split["val_ids"])
    (root / "selection_holdout.json").write_text(json.dumps({
        "fit_ids": fit, "holdout_ids": hold, "fit_ids_sha256": si.ids_sha256(fit),
        "holdout_ids_sha256": si.ids_sha256(hold)}), encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _tiny_hub(monkeypatch):
    monkeypatch.setattr(v71_io, "hub_model",
                        lambda arch, pretrained: tiny_hub_model(seed=0 if pretrained else 5, depth=4))


def _argv(pod: Path, out: Path, *extra: str) -> list[str]:
    return ["--ham-root", str(pod / "ham"), "--image-dir", str(pod / "ham"),
            "--split-index", str(pod / "c1_split_index.json"),
            "--holdout", str(pod / "selection_holdout.json"), "--weights-dir", str(pod / "v7"),
            "--out-dir", str(out), "--resolutions", "224", "--weights", *WEIGHTS,
            "--ci-resamples", "3", "--probe-jobs", "1", "--boot-jobs", "1", "--device", "cpu",
            "--batch-size", "16", "--decode-workers", "2", "--allow-dirty", *extra]


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    """One thread each for torch and BLAS: under ``-n 8`` the defaults oversubscribe
    the machine, and this module pushed a neighbour past the 120 s pytest timeout."""
    from threadpoolctl import threadpool_limits

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(1):
        yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def ran(pod, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(v71_io, "hub_model",
                   lambda arch, pretrained: tiny_hub_model(seed=0 if pretrained else 5, depth=4))
        assert v71_run.main(_argv(pod, out, "--stage", "grid")) == 0
        assert v71_run.main(_argv(pod, out, "--stage", "select")) == 0
        assert v71_run.main(_argv(pod, out, "--stage", "val", "--split", "holdout")) == 0
    return out


def test_pretest_passed_on_both_arms_and_both_endpoints(ran):
    pretest = json.loads((ran / "pretest.json").read_text())
    assert pretest["passed"] is True
    assert {(c["arm"], c["alpha"]) for c in pretest["checks"]} == {
        ("A", 0.0), ("A", 1.0), ("B", 0.0), ("B", 1.0)}


def test_grid_has_every_cell_of_every_unit(ran):
    grid = json.loads((ran / "holdout_grid.json").read_text())
    assert len(grid["rows"]) == len(WEIGHTS) * 60
    row = next(iter(grid["rows"].values()))
    assert "mel_sens_at_spec80" in row and "ci95" in row


def test_selection_picks_within_each_arm(ran):
    selection = json.loads((ran / "selection.json").read_text())
    assert selection["winners"]["A"].startswith("A_")
    assert selection["winners"]["B"].startswith("B_")
    assert "phase2" in selection
    assert set(selection["holdout_contrasts"]) == {"A", "B"}


def test_rehearsal_reads_every_entry_and_contrast(ran):
    doc = json.loads((ran / "v71_final_eval_holdout_rehearsal.json").read_text())
    assert doc["split"] == "holdout"
    assert set(doc["results"]) == {"A71", "B71", "public_A71", "public_B71", "A_v7", "B_v7",
                                   "public_v7", "public71"}
    assert "mel_sens_at_thr_spec80" in doc["results"]["A71"]
    assert {(r["arm"], r["control"]) for r in doc["paired"]} >= {("A71", "public_A71")}
    import build_v71_table

    assert "A* v7.1" in build_v71_table.render(doc)  # the table reads what the stage writes


def test_a_relaunch_skips_finished_units(pod, ran, capsys):
    assert v71_run.main(_argv(pod, ran, "--stage", "grid")) == 0
    assert "extraer" not in capsys.readouterr().out


def test_dry_run_prints_the_cost_and_touches_nothing(pod, tmp_path, capsys):
    assert v71_run.main(_argv(pod, tmp_path / "dry", "--stage", "grid", "--dry-run")) == 0
    assert "GPU-h" in capsys.readouterr().out
    assert not (tmp_path / "dry").exists()


def test_val_without_single_evaluation_is_refused(pod, ran):
    with pytest.raises(SystemExit, match="single-evaluation"):
        v71_run.main(_argv(pod, ran, "--stage", "val", "--split", "val"))


def test_pretest_aborts_when_an_endpoint_is_not_the_final(pod, tmp_path, monkeypatch):
    from dermssl import torch_v71 as tv

    real = tv.interpolate_state

    def off_by_a_bit(public, adapted, alpha):
        state = real(public, adapted, alpha)
        key = next(iter(state))
        state[key] = state[key] + 1e-3
        return state

    monkeypatch.setattr(tv, "interpolate_state", off_by_a_bit)
    with pytest.raises(SystemExit, match="pre-test"):
        v71_run.main(_argv(pod, tmp_path / "bad", "--stage", "grid"))


def test_val_guards_run_before_any_val_image_is_decoded(pod, ran):
    """Dirty tree (here) or foreign fingerprints (in CI) must abort the real read
    before a single validation image is touched."""
    exclude = pod / "exclude.txt"
    exclude.write_text("x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="ABORT"):
        v71_run.main(_argv(pod, ran, "--stage", "val", "--split", "val", "--single-evaluation",
                           "--exclude-ids-file", str(exclude)))
    assert not list((ran / "inputs").glob("val_*"))
    assert not (ran / "v71_final_eval.json").exists()


def test_selection_carries_the_resolution_diagnostic(ran):
    selection = json.loads((ran / "selection.json").read_text())
    assert selection["resolution_diagnostic"] == {}  # this smoke ran 224 only


def test_cached_results_from_another_holdout_are_refused(pod, ran):
    """Same --out-dir, different ids (here --limit): nothing cached may be reused."""
    with pytest.raises(SystemExit, match="otro holdout"):
        v71_run.main(_argv(pod, ran, "--stage", "grid", "--limit", "40"))


def test_val_thresholds_read_the_fit80_probe(pod, ran, monkeypatch):
    """The real val branch, guards stubbed (dirty tree, fake ids): thresholded mel
    metrics come from the probe fitted on the 80 % fit split, argmax metrics
    from the probe refitted on the full train."""
    import v71_val

    from dermssl import final_eval as fe
    from dermssl import v71_metrics as vm

    monkeypatch.setattr(v71_run, "_revision", lambda args: "test")
    monkeypatch.setattr(fe, "check_val_protocol", lambda train, val: None)
    monkeypatch.setattr(v71_val.rp, "file_sha256", lambda path: "e" * 64)
    exclude = pod / "exclude.txt"
    exclude.write_text("x\n", encoding="utf-8")
    assert v71_run.main(_argv(pod, ran, "--stage", "val", "--split", "val", "--single-evaluation",
                              "--exclude-ids-file", str(exclude))) == 0
    doc = json.loads((ran / "v71_final_eval.json").read_text())
    saved = np.load(ran / "v71_final_eval.probas.npz")
    y, mel = saved["y_true"], 4
    r = doc["results"]["A71"]
    fixed = vm.fixed_threshold_metrics(y, saved["A71__fit80"], mel=mel, thresholds=r["holdout_thresholds"])
    assert {k: r[k] for k in fixed} == pytest.approx(fixed)
    assert r["macro_f1"] == pytest.approx(vm.metric_bundle(y, saved["A71"], mel=mel)["macro_f1"])
    assert not np.allclose(saved["A71"], saved["A71__fit80"])  # two different probes
    assert "80 %" in doc["note_operating_points"] and "note_thresholds" not in doc


def test_a_missing_holdout_file_aborts_before_anything(pod, tmp_path):
    argv = _argv(pod, tmp_path / "out", "--stage", "grid", "--limit", "70")
    argv[argv.index("--holdout") + 1] = str(tmp_path / "missing.json")
    with pytest.raises(SystemExit, match="holdout"):
        v71_run.main(argv)
    assert not (tmp_path / "out").exists()


def test_a_bad_holdout_digest_aborts_the_smoke_before_extraction(pod, tmp_path):
    doc = json.loads((pod / "selection_holdout.json").read_text())
    doc["holdout_ids_sha256"] = "0" * 64
    bad = tmp_path / "bad_holdout.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    argv = _argv(pod, tmp_path / "out", "--stage", "grid", "--limit", "70")
    argv[argv.index("--holdout") + 1] = str(bad)
    with pytest.raises(SystemExit, match="digest"):
        v71_run.main(argv)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["pretest.json", "holdout_grid.json", "units/public@224/result.json",
                                  "selection.json", "v71_final_eval_holdout_rehearsal.json"])
def test_every_result_json_records_its_environment(ran, name):
    env = sidecar_env(ran / name)
    assert env["schema"] == 1 and env["torch"]["loaded"] is True
    assert env["knobs"]["probe_jobs"] == 1 and env["knobs"]["decode_workers"] == 2


@pytest.mark.parametrize("name", ["holdout_grid.json", "units/public@224/result.json",
                                  "v71_final_eval_holdout_rehearsal.json"])
def test_probe_results_record_the_pinned_probe_blas_threads(ran, name):
    from dermssl.probe_threads import PROBE_BLAS_THREADS  # noqa: PLC0415

    assert sidecar_env(ran / name)["knobs"]["loky_inner_threads"] == PROBE_BLAS_THREADS
