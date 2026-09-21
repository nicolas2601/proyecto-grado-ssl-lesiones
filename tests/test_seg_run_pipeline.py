"""Pure tests for the driver that chains the Camino B stages."""

import json


import seg_paths  # noqa: F401
import run_pipeline as rp


def _config(tmp_path, **overrides):
    values = dict(
        workdir=tmp_path / "w",
        preset={
            "pipeline": {"rounds": 2, "epochs_per_round": 5, "seed": 7},
            "cache": {"resolution": 224},
            "contrastive": {"epochs": 10},
            "ccam": {"epochs": 3},
            "pseudo": {"low": 0.35, "high": 0.65},
            "student": {"epochs": 99, "batch_size": 32},
        },
        roots=["ham10000=/d/ham"],
        rounds=None,
        epochs_per_round=None,
        nproc_per_node=1,
        python="py",
        stages=list(rp.ALL_STAGES),
        eval_image_dir="/e/img",
        eval_mask_dir="/e/msk",
        eval_name="isic2018",
        smoke=False,
    )
    values.update(overrides)
    return rp.PipelineConfig(**values)


def _names(stages):
    return [s.name for s in stages]


def _flag(stage, flag):
    """Value argparse would see: the LAST occurrence of a repeated flag wins."""
    position = len(stage.argv) - 1 - stage.argv[::-1].index(flag)
    return stage.argv[position + 1].replace("\\", "/")


def test_stage_order_chains_rounds_with_relabelling(tmp_path):
    stages = rp.build_stage_commands(_config(tmp_path))
    assert _names(stages) == [
        "env", "cache", "contrastive", "ccam", "pseudo_r0",
        "student_r1", "pseudo_r1", "student_r2", "evaluate",
    ]


def test_round_inputs_come_from_the_previous_round(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path))}
    assert _flag(stages["student_r1"], "--pseudo-masks").endswith("pseudo_r0/pseudo_masks.npy")
    assert _flag(stages["pseudo_r1"], "--saliency").endswith("student_r1/predictions.npy")
    assert _flag(stages["pseudo_r1"], "--source") == "student"
    assert _flag(stages["pseudo_r0"], "--source") == "ccam"
    assert _flag(stages["student_r2"], "--pseudo-masks").endswith("pseudo_r1/pseudo_masks.npy")
    assert "--no-predict" in stages["student_r2"].argv
    assert "--no-predict" not in stages["student_r1"].argv


def test_cli_rounds_and_epochs_override_the_preset(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path, rounds=1, epochs_per_round=3))}
    assert "student_r2" not in stages
    assert _flag(stages["student_r1"], "--epochs") == "3"
    assert "--no-predict" in stages["student_r1"].argv


def test_preset_epochs_per_round_beats_the_student_section(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path))}
    assert _flag(stages["student_r1"], "--epochs") == "5"
    assert _flag(stages["student_r1"], "--batch-size") == "32"


def test_seed_is_forwarded_to_every_training_stage(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path))}
    for name in ("cache", "contrastive", "ccam", "student_r1"):
        assert _flag(stages[name], "--seed") == "7"


def test_only_the_evaluation_stage_touches_reference_masks(tmp_path):
    stages = rp.build_stage_commands(_config(tmp_path))
    for stage in stages:
        joined = " ".join(stage.argv).lower()
        if stage.name == "evaluate":
            assert "--mask-dir" in stage.argv
        else:
            assert "--mask-dir" not in stage.argv and "isic2018" not in joined


def test_multi_gpu_uses_torch_distributed_only_for_training_stages(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path, nproc_per_node=2))}
    for name in ("contrastive", "ccam", "student_r1"):
        assert stages[name].argv[1:4] == ["-m", "torch.distributed.run", "--nproc_per_node=2"]
    for name in ("cache", "pseudo_r0", "evaluate"):
        assert "torch.distributed.run" not in stages[name].argv


def test_stage_subset_keeps_order_and_skips_the_rest(tmp_path):
    stages = rp.build_stage_commands(_config(tmp_path, stages=["student", "evaluate"]))
    assert _names(stages) == ["student_r1", "pseudo_r1", "student_r2", "evaluate"]


def test_evaluate_stage_is_skipped_without_reference_data(tmp_path):
    stages = rp.build_stage_commands(_config(tmp_path, eval_image_dir="", eval_mask_dir=""))
    assert "evaluate" not in _names(stages)


def test_smoke_mode_uses_synthetic_data_two_rounds_and_a_synthetic_eval_set(tmp_path):
    config = rp.smoke_config(tmp_path / "w", python="py", nproc_per_node=1)
    stages = {s.name: s for s in rp.build_stage_commands(config)}
    assert "--synthetic" in stages["cache"].argv
    assert "student_r2" in stages and "evaluate" in stages and "synthetic_eval" in stages
    assert _flag(stages["contrastive"], "--epochs") == "1"
    assert _flag(stages["student_r1"], "--epochs") == "4"


def test_shipped_pilot_preset_drives_a_valid_command_list(tmp_path):
    preset = json.loads((seg_paths.COMUN / "configs" / "piloto.json").read_text(encoding="utf-8"))
    config = _config(tmp_path, preset=preset)
    stages = rp.build_stage_commands(config)
    assert all(isinstance(part, str) for stage in stages for part in stage.argv)


def test_parse_args_accepts_smoke_and_dry_run():
    args = rp.parse_args(["--smoke", "--dry-run"])
    assert args.smoke and args.dry_run


def test_no_preload_is_forwarded_to_cache_and_training_stages(tmp_path):
    config = _config(tmp_path, no_preload=True)
    stages = {stage.name: stage for stage in rp.build_stage_commands(config)}
    for name in ("contrastive", "ccam", "student_r1"):
        assert "--no-preload" in stages[name].argv


def test_preload_is_forwarded_without_changing_checkpoint_selection(tmp_path):
    config = _config(tmp_path, preload=True)
    stages = {stage.name: stage for stage in rp.build_stage_commands(config)}
    for name in ("contrastive", "ccam", "student_r1"):
        assert "--preload" in stages[name].argv
    assert "--selection" not in stages["student_r1"].argv
    assert 'default="best-val"' in (
        (seg_paths.COMUN / "estudiante" / "train_student.py").read_text(encoding="utf-8")
    )


def test_smoke_cleanup_only_targets_the_default_workdir(tmp_path):
    default_workdir = tmp_path / "salidas" / "smoke"
    custom_workdir = tmp_path / "custom"
    marker_default = default_workdir / "marker.txt"
    marker_custom = custom_workdir / "marker.txt"
    marker_default.parent.mkdir(parents=True)
    marker_custom.parent.mkdir(parents=True)
    marker_default.write_text("keep?", encoding="utf-8")
    marker_custom.write_text("keep", encoding="utf-8")

    rp.clean_smoke_workdir(default_workdir, default_workdir)
    rp.clean_smoke_workdir(custom_workdir, default_workdir)

    assert not default_workdir.exists()
    assert marker_custom.is_file()


def test_final_preset_carries_placeholders_that_the_driver_detects():
    final = json.loads((seg_paths.COMUN / "configs" / "final.json").read_text(encoding="utf-8"))
    pending = rp.find_placeholders(final)
    assert "pipeline.rounds" in pending and "contrastive.epochs" in pending
    assert rp.find_placeholders(json.loads(
        (seg_paths.COMUN / "configs" / "piloto.json").read_text(encoding="utf-8"))) == []
