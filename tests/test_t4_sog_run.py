"""T4 end to end on a fake HAM10000 and tiny random DINOv2s, CPU only.

The v7.1 runner first writes its holdout grid for the three T4 weight sets on
the same fake data: that file plays the committed v7.1 grid, so the
reproduction guard is exercised for real (pass here, abort against the real
committed grid, which fake data cannot reproduce). Then grid, select and the
val stage in its holdout rehearsal run, and val refuses to run unguarded.
"""

from __future__ import annotations

import csv
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

import t4_sog_run  # noqa: E402
import v71_io  # noqa: E402
import v71_run  # noqa: E402

from dermssl import t4_sog  # noqa: E402
from tests.release_helpers import tiny_hub_model  # noqa: E402
from tests.test_v71_run import _fake_ham, _weights  # noqa: E402

T4_WEIGHTS = tuple(t4_sog.WEIGHTS.values())


def _hub(arch, pretrained):
    return tiny_hub_model(seed=0 if pretrained else 5, depth=4)


@pytest.fixture(scope="module", autouse=True)
def _one_thread_and_tiny_hub():
    from threadpoolctl import threadpool_limits

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(1), pytest.MonkeyPatch.context() as mp:
        mp.setattr(v71_io, "hub_model", _hub)
        mp.setattr(t4_sog_run, "check_registration", lambda repo: "registration-checked-in-tests")
        yield
    torch.set_num_threads(threads)


def _manifest(path: Path, ids) -> None:
    labels = ("dark", "medium", "light", "unavailable")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "fst_bucket", "tone_tercile", "status"])
        for k, image_id in enumerate(ids):
            writer.writerow([image_id, "", labels[k % 4], "ok"])


@pytest.fixture(scope="module")
def pod(tmp_path_factory):
    from c1_build_val_index import rebuild_split
    from dermssl import c1_split_identity as si
    from dermssl import selection_holdout as sh

    root = tmp_path_factory.mktemp("t4")
    (root / "ham").mkdir()
    _fake_ham(root / "ham")
    _weights(root / "v7")
    split, meta = rebuild_split(root / "ham")
    (root / "c1_split_index.json").write_text(json.dumps(
        {"train_ids": split["train_ids"], "val_ids": split["val_ids"]}), encoding="utf-8")
    rows = meta.set_index("image_id").loc[split["train_ids"]]
    fit, hold = sh.grouped_holdout(split["train_ids"], [split["cls2idx"][d] for d in rows.dx],
                                   list(rows.lesion_id), forbidden=split["val_ids"])
    (root / "selection_holdout.json").write_text(json.dumps({
        "fit_ids": fit, "holdout_ids": hold, "fit_ids_sha256": si.ids_sha256(fit),
        "holdout_ids_sha256": si.ids_sha256(hold)}), encoding="utf-8")
    _manifest(root / "manifest.csv", [*split["train_ids"], *split["val_ids"]])
    return root


def _common(pod: Path) -> list[str]:
    return ["--ham-root", str(pod / "ham"), "--image-dir", str(pod / "ham"),
            "--split-index", str(pod / "c1_split_index.json"),
            "--holdout", str(pod / "selection_holdout.json"), "--weights-dir", str(pod / "v7"),
            "--ci-resamples", "3", "--probe-jobs", "1", "--boot-jobs", "1", "--device", "cpu",
            "--batch-size", "16", "--decode-workers", "2", "--allow-dirty"]


def _t4(pod: Path, out: Path, *extra: str) -> list[str]:
    return [*_common(pod), "--out-dir", str(out), "--tone-manifest", str(pod / "manifest.csv"), *extra]


@pytest.fixture(scope="module")
def v71_grid(pod, tmp_path_factory):
    """The v7.1 runner's own holdout grid on this fake data: the stand-in for the committed one."""
    out = tmp_path_factory.mktemp("v71")
    argv = [*_common(pod), "--out-dir", str(out), "--resolutions", "224", "--weights", *T4_WEIGHTS]
    assert v71_run.main([*argv, "--stage", "grid"]) == 0
    return out / "holdout_grid.json"


@pytest.fixture(scope="module")
def ran(pod, v71_grid, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    for stage in (["--stage", "grid"], ["--stage", "select"], ["--stage", "val", "--split", "holdout"]):
        assert t4_sog_run.main(_t4(pod, out, "--v71-grid", str(v71_grid), *stage)) == 0
    return out


def test_identity_reproduces_the_v71_grid_before_any_sog_unit(ran):
    grid = json.loads((ran / "t4_holdout_grid.json").read_text())
    assert grid["reproduction"]["passed"] is True
    assert set(grid["inputs"]) == {"identity", "sog_p6"}
    assert all(len(rows) == len(T4_WEIGHTS) * 60 for rows in grid["inputs"].values())


def test_the_two_inputs_give_different_holdout_probabilities(ran):
    cid = t4_sog.FIXED_PROTOCOLS["A"]
    plain = t4_sog_run.unit_probas(ran, "identity", cid)[1]
    sog = t4_sog_run.unit_probas(ran, "sog_p6", cid)[1]
    assert plain.shape == sog.shape and not np.allclose(plain, sog)


def test_selection_names_a_sog_winner_per_arm(ran):
    selection = json.loads((ran / "t4_selection.json").read_text())
    for arm, weights in t4_sog.WEIGHTS.items():
        assert selection["winners"][arm].startswith(f"{weights}@224|")
    assert {r["arm"] for r in selection["holdout_contrasts"]} == {f"{a}_sog" for a in t4_sog.WEIGHTS}


def test_the_rehearsal_has_every_entry_contrast_and_gap(ran):
    doc = json.loads((ran / "t4_final_eval_holdout_rehearsal.json").read_text())
    assert set(doc["entries"]) == {f"{a}_{k}" for a in t4_sog.WEIGHTS for k in ("id", "sog", "sogwin")}
    metrics = {(r["arm"], r["metric"]) for r in doc["paired"]}
    for arm, control in (*t4_sog.PRIMARY_CONTRASTS, *t4_sog.SECONDARY_CONTRASTS):
        for metric in ("macro_f1", t4_sog.GAP_METRIC, t4_sog.RANGE_METRIC):
            assert (arm, metric) in metrics
    assert set(doc["terciles"]["A_sog"]) == {"dark", "medium", "light"}
    assert set(doc["verdicts"]) == {"P1", "P2", "P3"}


def test_grid_aborts_when_identity_does_not_reproduce_v71(pod, tmp_path):
    with pytest.raises(SystemExit, match="reproduce"):
        t4_sog_run.main(_t4(pod, tmp_path / "bad", "--stage", "grid"))  # the real committed grid
    assert not (tmp_path / "bad" / "sog_p6").exists()


def test_val_split_needs_single_evaluation(pod, tmp_path):
    with pytest.raises(SystemExit, match="single-evaluation"):
        t4_sog_run.main(_t4(pod, tmp_path, "--stage", "val", "--split", "val"))


def test_val_split_refuses_a_manifest_other_than_the_registered_one(pod, ran):
    argv = _t4(pod, ran, "--stage", "val", "--split", "val", "--single-evaluation",
               "--exclude-ids-file", str(pod / "manifest.csv"))
    argv.remove("--allow-dirty")
    with pytest.raises(SystemExit, match="manifiesto"):
        t4_sog_run.main(argv)


def test_dry_run_prints_the_cost_and_touches_nothing(pod, tmp_path, capsys):
    assert t4_sog_run.main(_t4(pod, tmp_path / "dry", "--stage", "grid", "--dry-run")) == 0
    assert "GPU-h" in capsys.readouterr().out and not (tmp_path / "dry").exists()


def test_val_split_refuses_limit_because_it_skips_the_reproduction_guard(pod, tmp_path):
    with pytest.raises(SystemExit, match="--limit"):
        t4_sog_run.main(_t4(pod, tmp_path, "--stage", "val", "--split", "val", "--single-evaluation",
                            "--limit", "10"))


# --- reproduction diagnosis (Paula, 2026-09-29): identity-only runs, feature hashes, errata flag ---


def test_every_unit_records_the_sha256_of_its_features(ran):
    result = json.loads((ran / "identity" / "units" / "public@224" / "result.json").read_text())
    digests = result["feature_sha256"]
    assert set(digests) == {f"{split}_{p}" for split in ("fit", "hold")
                            for p in ("cls", "mean", "gem", "attn", "cls4")}
    assert all(len(v) == 64 for v in digests.values())


@pytest.fixture(scope="module")
def identity_runs(pod, tmp_path_factory):
    """Two identity-only runs, same flags, against the real committed grid (which fake data fails)."""
    outs = [tmp_path_factory.mktemp(f"id{k}") for k in (1, 2)]
    for out in outs:
        assert t4_sog_run.main(_t4(pod, out, "--stage", "grid", "--inputs", "identity")) == 0
    return outs


def test_identity_only_reports_the_reproduction_without_aborting(identity_runs):
    grid = json.loads((identity_runs[0] / "t4_holdout_grid.json").read_text())
    assert set(grid["inputs"]) == {"identity"} and grid["reproduction"]["passed"] is False
    assert not (identity_runs[0] / "sog_p6").exists()
    env = sidecar_env(identity_runs[0] / "t4_holdout_grid.json")
    assert {"probe_jobs", "boot_jobs", "decode_workers", "loky_inner_threads"} <= set(env["knobs"])
    assert {"torch", "scikit-learn", "numpy"} <= set(env["packages"])
    assert {"cudnn_deterministic", "cudnn_benchmark", "num_threads"} <= set(env["torch"])
    assert env["cpu_count"] and "threadpools" in env and "OMP_NUM_THREADS" in env["env"]


def test_select_refuses_an_identity_only_grid(pod, identity_runs):
    with pytest.raises(SystemExit, match="solo identity"):
        t4_sog_run.main(_t4(pod, identity_runs[0], "--stage", "select"))


def test_two_identity_runs_on_cpu_compare_as_identical(identity_runs, tmp_path):
    import t4_repro_compare

    out = tmp_path / "cmp.json"
    assert t4_repro_compare.main(["--run-a", str(identity_runs[0]), "--run-b", str(identity_runs[1]),
                                  "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["a_vs_b"]["identical"] is True
    assert doc["a_vs_published"]["max_abs_delta"]["macro_f1"] > 0  # fake data vs the real v7.1 grid


@pytest.mark.parametrize("name", ["t4_holdout_grid.json", "t4_selection.json",
                                  "t4_final_eval_holdout_rehearsal.json"])
def test_every_t4_result_json_records_its_environment(ran, name):
    env = sidecar_env(ran / name)
    assert env["schema"] == 1 and env["torch"]["loaded"] is True and env["knobs"]["boot_jobs"] == 1
