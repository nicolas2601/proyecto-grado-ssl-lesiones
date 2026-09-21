"""Pure tests for the driver that chains the Camino B stages."""

import json

import pytest

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
            "pseudo": {"low": 0.35, "high": 0.65, "max_area_fraction": 0.6},
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
        "student_r1", "pseudo_r1", "student_r2", "fst", "evaluate_isic2018",
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
        if stage.name.startswith("evaluate"):
            assert "--mask-dir" in stage.argv
        else:
            assert "--mask-dir" not in stage.argv and "isic2018" not in joined


def test_multi_gpu_uses_torch_distributed_only_for_training_stages(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path, nproc_per_node=2))}
    for name in ("contrastive", "ccam", "student_r1"):
        assert stages[name].argv[1:4] == ["-m", "torch.distributed.run", "--nproc_per_node=2"]
    for name in ("cache", "pseudo_r0", "fst", "evaluate_isic2018"):
        assert "torch.distributed.run" not in stages[name].argv


def test_stage_subset_keeps_order_and_skips_the_rest(tmp_path):
    stages = rp.build_stage_commands(_config(tmp_path, stages=["student", "evaluate"]))
    assert _names(stages) == ["student_r1", "pseudo_r1", "student_r2", "evaluate_isic2018"]


def test_evaluate_stage_is_skipped_without_reference_data(tmp_path):
    stages = rp.build_stage_commands(_config(tmp_path, eval_image_dir="", eval_mask_dir=""))
    assert not any(name.startswith("evaluate") for name in _names(stages))


def test_smoke_mode_uses_synthetic_data_two_rounds_and_a_synthetic_eval_set(tmp_path):
    config = rp.smoke_config(tmp_path / "w", python="py", nproc_per_node=1)
    stages = {s.name: s for s in rp.build_stage_commands(config)}
    assert "--synthetic" in stages["cache"].argv
    assert "student_r2" in stages and "evaluate_synthetic" in stages and "synthetic_eval" in stages
    assert _flag(stages["contrastive"], "--epochs") == "1"
    assert _flag(stages["student_r1"], "--epochs") == "4"


def test_smoke_mode_also_exercises_the_field_flags(tmp_path):
    config = rp.smoke_config(tmp_path / "w", python="py", nproc_per_node=1)
    stages = {s.name: s for s in rp.build_stage_commands(config)}
    assert "--field-black-threshold" in stages["cache"].argv
    assert "--field-margin-fraction" in stages["cache"].argv
    assert _flag(stages["pseudo_r0"], "--field").endswith("cache/field.npy")
    # The smoke runs the same cap as the cluster presets, so a regression in the
    # field detector or in the orientation shows up here and not only in production.
    pilot = _cluster_preset("cluster-piloto.json")["pseudo"]["max_area_fraction"]
    assert float(_flag(stages["pseudo_r0"], "--max-area-fraction")) == pilot


def test_smoke_mode_skips_the_fst_stage_because_there_are_no_files_on_disk(tmp_path):
    config = rp.smoke_config(tmp_path / "w", python="py", nproc_per_node=1)
    assert "fst" not in _names(rp.build_stage_commands(config))


def test_a_smoke_preset_without_cache_keys_falls_back_instead_of_crashing(tmp_path):
    config = rp.smoke_config(tmp_path / "w", python="py", nproc_per_node=1)
    config.preset = {**config.preset, "cache": {"threads": 4}}
    stages = {s.name: s for s in rp.build_stage_commands(config)}
    assert _flag(stages["cache"], "--synthetic") == str(rp.SMOKE_SYNTHETIC_COUNT)
    assert _flag(stages["synthetic_eval"], "--size") == str(rp.SMOKE_RESOLUTION)


def test_shipped_pilot_preset_drives_a_valid_command_list(tmp_path):
    preset = json.loads((seg_paths.COMUN / "configs" / "piloto.json").read_text(encoding="utf-8"))
    config = _config(tmp_path, preset=preset)
    stages = rp.build_stage_commands(config)
    assert all(isinstance(part, str) for stage in stages for part in stage.argv)


def test_parse_args_accepts_smoke_and_dry_run():
    args = rp.parse_args(["--smoke", "--dry-run"])
    assert args.smoke and args.dry_run


def test_smoke_alone_still_fills_in_the_defaults():
    args = rp.parse_args(["--smoke"])
    assert args.config == str(rp.DEFAULT_CONFIG)
    assert args.stages == list(rp.ALL_STAGES)


@pytest.mark.parametrize("extra", [["--config", "otro.json"], ["--stages", "cache"]])
def test_smoke_refuses_the_flags_it_would_have_ignored(extra, capsys):
    with pytest.raises(SystemExit):
        rp.parse_args(["--smoke", *extra])
    assert "--smoke" in capsys.readouterr().err


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


# --- field ROI, several evaluation sets, FST stage and live tracking ---------


def test_pseudo_stages_receive_the_field_and_the_area_cap(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path))}
    for name in ("pseudo_r0", "pseudo_r1"):
        assert _flag(stages[name], "--field").endswith("cache/field.npy")
        assert _flag(stages[name], "--max-area-fraction") == "0.6"


def test_the_field_is_omitted_when_the_preset_disables_it(tmp_path):
    preset = _config(tmp_path).preset
    preset["cache"] = {"resolution": 224, "no_field": True}
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path, preset=preset))}
    assert "--field" not in stages["pseudo_r0"].argv
    assert "--no-field" in stages["cache"].argv


def test_parse_eval_set_reads_name_images_and_masks():
    parsed = rp.parse_eval_set("isic2018=/data/img:/data/msk")
    assert (parsed.name, parsed.image_dir, parsed.mask_dir) == ("isic2018", "/data/img", "/data/msk")


def test_parse_eval_set_rejects_a_malformed_specification():
    for spec in ("isic2018", "isic2018=/data/img", "=/a:/b"):
        try:
            rp.parse_eval_set(spec)
        except ValueError:
            continue
        raise AssertionError(f"{spec!r} should be rejected")


def test_every_eval_set_becomes_its_own_evaluation_stage(tmp_path):
    sets = [rp.parse_eval_set("ham10000=/h/img:/h/msk"), rp.parse_eval_set("ph2=/p/img:/p/msk")]
    config = _config(tmp_path, eval_image_dir="", eval_mask_dir="", eval_sets=sets)
    stages = {s.name: s for s in rp.build_stage_commands(config)}
    assert "evaluate_ham10000" in stages and "evaluate_ph2" in stages
    stage = stages["evaluate_ph2"]
    assert _flag(stage, "--eval-name") == "ph2"
    assert _flag(stage, "--image-dir") == "/p/img"
    assert _flag(stage, "--mask-dir") == "/p/msk"
    assert _flag(stage, "--output-dir").endswith("evaluacion_ph2")
    assert _flag(stage, "--overlay-dir").endswith("evaluacion_ph2/overlays")


def test_legacy_eval_flags_keep_working_as_one_more_set(tmp_path):
    sets = [rp.parse_eval_set("ph2=/p/img:/p/msk")]
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path, eval_sets=sets))}
    assert "evaluate_isic2018" in stages and "evaluate_ph2" in stages
    legacy = stages["evaluate_isic2018"]
    assert _flag(legacy, "--image-dir") == "/e/img"
    assert _flag(legacy, "--eval-name") == "isic2018"


def test_the_fst_manifest_reaches_every_evaluation_stage(tmp_path):
    sets = [rp.parse_eval_set("ph2=/p/img:/p/msk")]
    config = _config(tmp_path, eval_sets=sets, fst_manifest="/m/fst.csv")
    stages = rp.build_stage_commands(config)
    evaluations = [s for s in stages if s.name.startswith("evaluate")]
    assert len(evaluations) == 2
    for stage in evaluations:
        assert _flag(stage, "--fst-manifest") == "/m/fst.csv"
    for stage in stages:
        if not stage.name.startswith("evaluate"):
            assert "--fst-manifest" not in stage.argv


def test_the_fst_stage_reads_the_last_pseudo_round_and_the_cache(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path))}
    stage = stages["fst"]
    assert _flag(stage, "--manifest").endswith("cache/manifest.csv")
    assert _flag(stage, "--pseudo-masks").endswith("pseudo_r1/pseudo_masks.npy")
    assert _flag(stage, "--field").endswith("cache/field.npy")
    assert _flag(stage, "--output-json").endswith("fst/fst_pseudo_manifest.json")


def test_the_fst_stage_also_writes_the_csv_that_the_evaluation_reads(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path))}
    assert _flag(stages["fst"], "--output-csv").endswith("fst/fst_pseudo_manifest.csv")


def test_the_evaluation_defaults_to_the_csv_produced_by_the_fst_stage(tmp_path):
    stages = rp.build_stage_commands(_config(tmp_path))
    evaluations = [s for s in stages if s.name.startswith("evaluate")]
    assert evaluations
    for stage in evaluations:
        assert _flag(stage, "--fst-manifest").endswith("fst/fst_pseudo_manifest.csv")


def test_an_explicit_fst_manifest_replaces_the_generated_one(tmp_path):
    config = _config(tmp_path, fst_manifest="/m/propio.csv")
    stage = next(s for s in rp.build_stage_commands(config) if s.name.startswith("evaluate"))
    assert stage.argv.count("--fst-manifest") == 1
    assert _flag(stage, "--fst-manifest") == "/m/propio.csv"


def test_without_the_fst_stage_the_evaluation_gets_no_manifest(tmp_path):
    config = _config(tmp_path, stages=["student", "evaluate"])
    stage = next(s for s in rp.build_stage_commands(config) if s.name.startswith("evaluate"))
    assert "--fst-manifest" not in stage.argv


def test_the_fst_stage_follows_a_single_round_pipeline(tmp_path):
    stages = {s.name: s for s in rp.build_stage_commands(_config(tmp_path, rounds=1))}
    assert _flag(stages["fst"], "--pseudo-masks").endswith("pseudo_r0/pseudo_masks.npy")


def test_the_wandb_project_reaches_training_and_evaluation_stages(tmp_path):
    config = _config(tmp_path, wandb_project="dermssl-segmentacion")
    stages = {s.name: s for s in rp.build_stage_commands(config)}
    for name in ("contrastive", "ccam", "student_r1", "student_r2", "evaluate_isic2018"):
        assert _flag(stages[name], "--wandb-project") == "dermssl-segmentacion"
    for name in ("env", "cache", "pseudo_r0", "fst"):
        assert "--wandb-project" not in stages[name].argv


def test_without_a_project_no_stage_mentions_weights_and_biases(tmp_path):
    for stage in rp.build_stage_commands(_config(tmp_path)):
        assert not any(part.startswith("--wandb") for part in stage.argv)


def test_child_env_exports_the_project_and_the_run_group(tmp_path):
    config = _config(tmp_path, wandb_project="dermssl-segmentacion")
    env = rp.child_env(config)
    assert env == {"WANDB_PROJECT": "dermssl-segmentacion", "WANDB_RUN_GROUP": "w"}


def test_child_env_is_empty_without_a_project(tmp_path):
    assert rp.child_env(_config(tmp_path)) == {}


def test_no_stage_or_environment_ever_carries_the_api_key(tmp_path):
    config = _config(tmp_path, wandb_project="dermssl-segmentacion")
    for stage in rp.build_stage_commands(config):
        assert not any("WANDB_API_KEY" in part for part in stage.argv)
    assert "WANDB_API_KEY" not in rp.child_env(config)


def test_parse_args_collects_repeated_eval_sets_and_tracking_flags():
    args = rp.parse_args([
        "--eval-set", "isic2018=/i/img:/i/msk",
        "--eval-set", "ham10000=/h/img:/h/msk",
        "--fst-manifest", "/m/fst.csv",
        "--wandb-project", "dermssl-segmentacion",
    ])
    assert args.eval_set == ["isic2018=/i/img:/i/msk", "ham10000=/h/img:/h/msk"]
    assert args.fst_manifest == "/m/fst.csv"
    assert args.wandb_project == "dermssl-segmentacion"


def _cluster_preset(name):
    return json.loads((seg_paths.COMUN / "configs" / name).read_text(encoding="utf-8"))


def test_cluster_presets_are_complete_and_carry_the_new_sections():
    for name in ("cluster-piloto.json", "cluster-final.json"):
        preset = _cluster_preset(name)
        assert rp.find_placeholders(preset) == [], name
        assert preset["pseudo"]["max_area_fraction"] == 0.6
        assert preset["cache"]["field_black_threshold"] > 0
        assert preset["cache"]["field_margin_fraction"] > 0
        assert preset["evaluate"]["batch_size"] == 32


def test_cluster_presets_size_the_pilot_below_the_final_run():
    pilot, final = _cluster_preset("cluster-piloto.json"), _cluster_preset("cluster-final.json")
    assert pilot["pipeline"]["rounds"] == 1 and final["pipeline"]["rounds"] == 3
    assert pilot["contrastive"]["epochs"] < final["contrastive"]["epochs"]
    assert pilot["ccam"]["epochs"] < final["ccam"]["epochs"]
    assert final["contrastive"]["momentum"] == 0.999


def test_only_the_final_preset_pays_for_hausdorff(tmp_path):
    assert "hausdorff" not in _cluster_preset("cluster-piloto.json")["evaluate"]
    config = _config(tmp_path, preset=_cluster_preset("cluster-final.json"))
    stage = next(s for s in rp.build_stage_commands(config) if s.name.startswith("evaluate"))
    # store_true on the evaluation CLI, so the preset boolean must render bare.
    assert stage.argv.count("--hausdorff") == 1
    following = stage.argv[stage.argv.index("--hausdorff") + 1:]
    assert not following or following[0].startswith("--")


def test_cluster_pilot_preset_builds_every_stage(tmp_path):
    config = _config(tmp_path, preset=_cluster_preset("cluster-piloto.json"))
    stages = {s.name: s for s in rp.build_stage_commands(config)}
    assert set(stages) >= {"cache", "contrastive", "ccam", "pseudo_r0", "student_r1", "fst"}
    assert _flag(stages["pseudo_r0"], "--max-area-fraction") == "0.6"
