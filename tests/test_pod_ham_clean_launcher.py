"""scripts/pod/ham_clean.sh: strict shell, the real exit code of each step, an rc=0 marker only with its artifacts."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCHER = REPO / "scripts/pod/ham_clean.sh"
STUB = """#!/usr/bin/env python3
import os, sys
from pathlib import Path
argv = sys.argv[1:]
arg = lambda flag: argv[argv.index(flag) + 1] if flag in argv else None
out, stage = Path(arg("--out-dir")), arg("--stage")
script = Path(argv[0]).name
if os.environ.get("STUB_LOG"):
    open(os.environ["STUB_LOG"], "a").write(f"{script}:{arg('--stage')}\\n")
if os.environ.get("STUB_FAIL") == f"{script}:{stage}":
    sys.exit(3)
out.mkdir(parents=True, exist_ok=True)
rehearsal = "--rehearsal" in argv
if script == "v71_run.py":
    for n in ("v71_final_eval_holdout_rehearsal.json", "v71_final_eval_holdout_rehearsal.probas.npz"):
        (out / n).write_text("x")
elif stage == "extract":
    (out / "ham_clean_extract.json").write_text("{}")
    if not rehearsal:
        (out / "val_read.lock").write_text("{}")
else:
    name = "ham_clean_holdout_rehearsal" if rehearsal else "ham_clean_val"
    if os.environ.get("STUB_NO_SIDECAR") != "1":
        (out / f"{name}.environment.json").write_text("{}")
    (out / f"{name}.json").write_text("{}")
    (out / f"{name}.probas.npz").write_text("x")
"""


@pytest.fixture()
def pod(tmp_path):
    repo = tmp_path / "repo"
    (repo / "resultados/clasificacion/v71").mkdir(parents=True)
    for name in ("selection.json", "holdout_grid.json"):
        (repo / "resultados/clasificacion/v71" / name).write_text("{}")
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", "--no-verify")
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    stub.chmod(0o755)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    env = {**os.environ, "REPO": str(repo), "PYTHON": str(stub), "P": str(tmp_path / "p"), "OUT_ROOT": str(tmp_path / "out")}
    return env, head, tmp_path / "out"


def _run(env, *args):
    return subprocess.run(["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True)


def test_the_launcher_is_strict_shell_and_does_not_pipe_a_step_through_tail():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert subprocess.run(["bash", "-n", str(LAUNCHER)]).returncode == 0
    assert "set -euo pipefail" in text and "require_commit" in text
    assert "| tail" not in text and "| grep" not in text


def test_val_mode_runs_extract_then_val_and_marks_rc0_with_artifacts(pod):
    env, head, out = pod
    done = _run(env, "val", head)
    assert done.returncode == 0, done.stdout + done.stderr
    marker = (out / "val" / "ham_clean_val.marker").read_text()
    assert marker.strip().endswith("rc=0")
    for name in ("ham_clean_val.json", "ham_clean_val.probas.npz", "ham_clean_val.environment.json", "val_read.lock"):
        assert (out / "val" / name).is_file()


def test_a_failing_step_leaves_no_rc0_marker_and_returns_its_real_code(pod):
    env, head, out = pod
    done = _run({**env, "STUB_FAIL": "ham_clean_run.py:val"}, "val", head)
    assert done.returncode == 3
    marker = out / "val" / "ham_clean_val.marker"
    assert not marker.exists() or "rc=0" not in marker.read_text()


def test_missing_artifact_means_no_rc0_even_if_every_step_exited_zero(pod):
    env, head, out = pod
    done = _run({**env, "STUB_NO_SIDECAR": "1"}, "val", head)
    assert done.returncode != 0
    marker = out / "val" / "ham_clean_val.marker"
    assert not marker.exists() or "rc=0" not in marker.read_text()


def test_a_commit_that_is_not_an_ancestor_of_head_aborts_before_any_step(pod):
    env, _, out = pod
    done = _run(env, "val", "0" * 40)
    assert done.returncode != 0 and not out.exists()


def test_a_dirty_clone_aborts(pod):
    env, head, out = pod
    (Path(env["REPO"]) / "stray.txt").write_text("x")
    assert _run(env, "val", head).returncode != 0 and not out.exists()


def test_val_refuses_an_out_dir_that_already_has_a_lock(pod):
    env, head, out = pod
    (out / "val").mkdir(parents=True)
    (out / "val" / "val_read.lock").write_text("{}")
    assert _run(env, "val", head).returncode != 0


def test_rehearsal_builds_its_own_reference_and_never_creates_a_lock(pod):
    env, head, out = pod
    done = _run(env, "rehearsal", head)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (out / "rehearsal_reference" / "v71_final_eval_holdout_rehearsal.json").is_file()
    assert (out / "ham_clean_rehearsal" / "ham_clean_holdout_rehearsal.json").is_file()
    assert not list(out.rglob("val_read.lock"))
    assert (out / "ham_clean_rehearsal" / "ham_clean_rehearsal.marker").read_text().strip().endswith("rc=0")


def test_rehearsal_can_be_rerun_on_the_same_out_root_without_redoing_the_reference(pod, tmp_path):
    env, head, out = pod
    env = {**env, "STUB_LOG": str(tmp_path / "calls.log")}
    assert _run(env, "rehearsal", head).returncode == 0
    again = _run(env, "rehearsal", head)
    assert again.returncode == 0, again.stdout + again.stderr
    calls = (tmp_path / "calls.log").read_text().split()
    assert calls.count("v71_run.py:val") == 1 and calls.count("ham_clean_run.py:extract") == 2
    assert list(out.glob("ham_clean_rehearsal.prev.*"))
    assert (out / "ham_clean_rehearsal" / "ham_clean_rehearsal.marker").read_text().strip().endswith("rc=0")


def test_val_mode_still_refuses_a_second_run_even_after_a_rehearsal(pod):
    env, head, out = pod
    assert _run(env, "rehearsal", head).returncode == 0
    assert _run(env, "val", head).returncode == 0
    assert _run(env, "val", head).returncode != 0


def test_parallelism_comes_from_the_script_defaults_not_from_magic_numbers_in_the_launcher():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "--probe-jobs" not in text and "--decode-workers" not in text


def test_the_launcher_is_executable_or_runs_with_bash():
    assert shutil.which("bash") and LAUNCHER.is_file()
