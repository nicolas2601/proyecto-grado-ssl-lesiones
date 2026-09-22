"""One command that runs the whole tone pipeline and can be re-run after a crash.

The pod has died four times mid-run. The recovery procedure so far was a human
reading the runbook, working out which stage had finished, and pasting the next
command. This orchestrator makes the recovery a single line: every stage
declares the artefact that proves it finished, a completed stage is skipped,
and an interrupted one is relaunched exactly as it was.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_stage2 as orch  # noqa: E402


def _args(tmp_path, **overrides):
    values = dict(
        repo=tmp_path / "repo",
        outputs=tmp_path / "salidas",
        python=sys.executable,
        ham_root=tmp_path / "ham",
        budget_gpu_hours=12.0,
        nproc_per_node=2,
        batch_size_per_gpu=64,
        num_workers=0,
        wandb_project="proj",
        wandb_group="etapa2",
        seeds=[0, 1, 2],
        only=None,
        dry_run=True,
        max_retries=2,
    )
    values.update(overrides)
    return orch.Settings(**values)


def test_every_stage_declares_the_artefact_that_proves_it_finished(tmp_path):
    stages = orch.build_stages(_args(tmp_path))
    assert [s.name for s in stages] == [
        "export", "tone_pairs", "ssl_plain", "ssl_proposed", "bank"
    ]
    for stage in stages:
        assert stage.done_when, f"{stage.name} has no completion artefact"
        assert stage.argv, f"{stage.name} has no command"


def test_a_stage_whose_artefact_exists_is_skipped(tmp_path):
    settings = _args(tmp_path)
    stages = orch.build_stages(settings)
    export = stages[0]
    export.done_when.parent.mkdir(parents=True, exist_ok=True)
    export.done_when.write_text("{}", encoding="utf-8")

    assert orch.is_done(export) is True
    assert orch.is_done(stages[1]) is False


def test_pending_stages_keep_their_order_and_drop_the_finished_ones(tmp_path):
    settings = _args(tmp_path)
    stages = orch.build_stages(settings)
    for stage in stages[:2]:
        stage.done_when.parent.mkdir(parents=True, exist_ok=True)
        stage.done_when.write_text("{}", encoding="utf-8")

    pending = [s.name for s in orch.pending_stages(stages)]
    assert pending == ["ssl_plain", "ssl_proposed", "bank"]


def test_the_budget_and_the_checkpoint_cadence_reach_both_arms(tmp_path):
    stages = {s.name: s for s in orch.build_stages(_args(tmp_path, budget_gpu_hours=16.0))}
    for name in ("ssl_plain", "ssl_proposed"):
        argv = stages[name].argv
        assert argv[argv.index("--budget-gpu-hours") + 1] == "16.0"
        start = argv.index("--checkpoint-fractions") + 1
        fractions = [float(v) for v in argv[start:start + len(orch.CHECKPOINT_FRACTIONS)]]
        assert fractions == list(orch.CHECKPOINT_FRACTIONS)
        assert fractions[-1] == 1.0
        assert argv[start + len(fractions)].startswith("--")


def test_only_the_proposed_arm_gets_the_transformed_inputs(tmp_path):
    stages = {s.name: s for s in orch.build_stages(_args(tmp_path))}
    assert "--transformed-image-root" in stages["ssl_proposed"].argv
    assert "--transformed-image-root" not in stages["ssl_plain"].argv


def test_the_two_arms_write_to_different_directories(tmp_path):
    stages = {s.name: s for s in orch.build_stages(_args(tmp_path))}
    plain = stages["ssl_plain"].argv
    proposed = stages["ssl_proposed"].argv
    assert plain[plain.index("--output-dir") + 1] != proposed[proposed.index("--output-dir") + 1]


def test_the_bank_reads_both_arms_and_the_public_backbone(tmp_path):
    argv = {s.name: s for s in orch.build_stages(_args(tmp_path))}["bank"].argv
    joined = " ".join(argv)
    assert "dinov2_public=dinov2_vits14:public" in joined
    assert "ssl_plain=dinov2_vits14:" in joined
    assert "ssl_proposed=dinov2_vits14:" in joined
    assert "--strata-column" in argv


def test_only_runs_the_named_stage(tmp_path):
    stages = orch.build_stages(_args(tmp_path, only="bank"))
    assert [s.name for s in stages] == ["bank"]


def test_an_unknown_stage_name_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="desconocida"):
        orch.build_stages(_args(tmp_path, only="no_existe"))


def test_a_dry_run_reports_every_stage_without_launching_anything(tmp_path, capsys):
    settings = _args(tmp_path)
    result = orch.run(settings)
    printed = capsys.readouterr().out
    assert result["dry_run"] is True
    assert result["launched"] == []
    for name in ("export", "tone_pairs", "ssl_plain", "ssl_proposed", "bank"):
        assert name in printed


def test_a_failing_stage_is_retried_and_then_reported(tmp_path, monkeypatch):
    settings = _args(tmp_path, dry_run=False, max_retries=2, only="export")
    attempts = []

    def always_fails(argv, **kwargs):
        attempts.append(argv)
        return subprocess.CompletedProcess(argv, returncode=1)

    monkeypatch.setattr(orch.subprocess, "run", always_fails)
    result = orch.run(settings)

    assert len(attempts) == 2
    assert result["failed"] == ["export"]


def test_a_stage_that_succeeds_is_not_retried(tmp_path, monkeypatch):
    settings = _args(tmp_path, dry_run=False, only="export")
    attempts = []

    def succeeds(argv, **kwargs):
        attempts.append(argv)
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(orch.subprocess, "run", succeeds)
    result = orch.run(settings)

    assert len(attempts) == 1
    assert result["failed"] == []
    assert result["launched"] == ["export"]


def test_the_state_file_records_what_finished(tmp_path, monkeypatch):
    settings = _args(tmp_path, dry_run=False, only="export")
    monkeypatch.setattr(
        orch.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, returncode=0),
    )
    orch.run(settings)

    state = json.loads((settings.outputs / "stage2_state.json").read_text(encoding="utf-8"))
    assert state["stages"]["export"]["status"] == "ok"
    assert "started_at" in state["stages"]["export"]


def test_parse_args_exposes_the_operator_contract():
    args = orch.parse_args([
        "--repo", "/r", "--outputs", "/o", "--ham-root", "/h",
        "--budget-gpu-hours", "16", "--only", "bank", "--dry-run",
    ])
    assert args.budget_gpu_hours == 16.0
    assert args.only == "bank"
    assert args.dry_run is True
    assert args.nproc_per_node == 2


# ------------------------------------------- everything writes to the NFS share
def test_the_environment_pins_every_scratch_path_inside_the_outputs_tree(tmp_path):
    settings = _args(tmp_path)
    env = orch.stage_environment(settings, base_env={"PATH": "/usr/bin"})

    for key in ("TMPDIR", "TORCH_HOME", "WANDB_DIR", "HF_HOME", "XDG_CACHE_HOME"):
        assert key in env, f"{key} left pointing at the overlay"
        assert str(settings.outputs) in env[key], f"{key} escapes the outputs tree"
    assert env["PATH"] == "/usr/bin", "the inherited environment must survive"


def test_the_scratch_directories_are_created_before_a_stage_runs(tmp_path, monkeypatch):
    settings = _args(tmp_path, dry_run=False, only="export")
    monkeypatch.setattr(
        orch.subprocess, "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, returncode=0),
    )
    orch.run(settings)

    for name in ("tmp", "torch", "wandb"):
        assert (settings.outputs / "scratch" / name).is_dir()


def test_the_stage_runs_with_that_environment(tmp_path, monkeypatch):
    settings = _args(tmp_path, dry_run=False, only="export")
    seen = {}

    def capture(argv, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, returncode=0)

    monkeypatch.setattr(orch.subprocess, "run", capture)
    orch.run(settings)

    assert str(settings.outputs) in seen["env"]["TORCH_HOME"]
