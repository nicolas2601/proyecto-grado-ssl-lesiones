"""The v7 final recipe rebuilds the s43 finals exactly; a replica changes seed and folder only.

``S43`` is the Weights & Biases config of the two seed-43 finals
(4672e3ff0604 and b8e85ebc8ece), the config fields ``scripts/pretrain.py``
turns into a ``PretrainConfig``. The budget is part of it: 8 GPU-h for A,
12 for B, and a different cap changes when the clock stops a slow run.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pretrain  # noqa: E402

from dermssl.pretrain import v7_finals as vf  # noqa: E402

POD = Path("/home/jovyan/work/proyectonp/salidas")
_COMMON = {
    "method": "simsiam_dinov2", "backbone": "dinov2_vitb14", "initial_weights": "dinov2-public-lvd142m",
    "image_root": str(POD / "tone/inputs/originals"), "split_hash": vf.SPLIT_HASH,
    "data_revision": "pool-final", "max_steps": 5700, "batch_size_per_gpu": 128, "num_workers": 0,
    "freeze_blocks": 8, "layer_decay": 0.75, "hue_jitter": 0.02, "anchor_l2sp": 0.0001,
    "anchor_distill": 0.0, "head_warmup_steps": 150, "checkpoint_every_steps": 500,
    "checkpoint_fractions": (0.25, 0.5, 0.75, 1.0), "precision": "bf16", "resolution": 224,
    "consistency_weight": 1.0, "lesion_guided_crops": True, "mlflow_experiment": "obj2-ssl-pretraining",
    "tracking_uri": "", "resume_checkpoint": "", "seed": 43,
    "exclude_ids_file": str(POD / "c1/selection/pretrain_exclude_ids_finals.txt"),
}
S43 = {
    "A": {**_COMMON, "output_dir": str(POD / "v7/final_A_s43"), "budget_gpu_hours": 8.0,
          "transformed_image_root": "", "transformed_manifest_path": "", "transformed_data_revision": "",
          "tone_consistency_weight": 0.0, "grad_checkpoint": False},
    "B": {**_COMMON, "output_dir": str(POD / "v7/final_B_s43"), "budget_gpu_hours": 12.0,
          "transformed_image_root": str(POD / "tone/shift2"),
          "transformed_manifest_path": str(POD / "tone/shift2/tone_transform_manifest.json"),
          "transformed_data_revision": vf.TONE_MANIFEST_REVISION,
          "tone_consistency_weight": 0.1, "grad_checkpoint": True},
}


def _config(arm: str, seed: int, **kwargs) -> dict:
    args = pretrain.parse_args(vf.pretrain_argv(arm, seed, salidas=POD, **kwargs))
    # W&B logged the validated config, where the initial weights are resolved.
    return dataclasses.asdict(pretrain.config_mod.validate(pretrain.build_config(args)))


@pytest.mark.parametrize("arm", vf.ARMS)
def test_the_recipe_rebuilds_the_s43_config_field_by_field(arm):
    built = _config(arm, 43)
    assert {k: built[k] for k in S43[arm]} == S43[arm]


def test_the_budget_is_the_one_s43_ran_per_arm():
    assert _config("A", 44, num_workers=4)["budget_gpu_hours"] == 8.0
    assert _config("B", 44, num_workers=4)["budget_gpu_hours"] == 12.0


@pytest.mark.parametrize("arm", vf.ARMS)
def test_s44_changes_only_seed_folder_and_the_declared_workers(arm):
    s43, s44 = _config(arm, 43), _config(arm, 44, num_workers=4)
    changed = {k for k in s43 if s43[k] != s44[k]}
    assert changed == {"seed", "output_dir", "num_workers"}
    assert s44["output_dir"] == str(POD / f"v7/final_{arm}_s44")


def test_the_base_seed_writes_to_the_unsuffixed_folder():
    assert vf.final_dir(POD, "A", 42) == POD / "v7/final_A"


def test_only_a_non_recipe_worker_count_is_reported_as_a_deviation():
    assert vf.deviations(0) == []
    assert "num_workers=4" in vf.deviations(4)[0]


def test_resume_is_passed_only_when_asked():
    assert "--resume" not in vf.pretrain_argv("A", 44, salidas=POD)
    assert "--resume" in vf.pretrain_argv("A", 44, salidas=POD, resume=True)


def test_an_unknown_arm_refuses():
    with pytest.raises(ValueError, match="brazo"):
        vf.pretrain_argv("C", 44, salidas=POD)


def test_the_launcher_prints_the_exact_command_without_running_it(capsys):
    import v7_final_launch as launch  # noqa: PLC0415

    assert launch.main(["--arm", "B", "--seed", "44", "--salidas", str(POD), "--num-workers", "4",
                        "--print-only"]) == 0
    out = capsys.readouterr().out
    assert "scripts/pretrain.py" in out and "--budget-gpu-hours 12" in out and "--seed 44" in out
    assert "DESVIACION" in out
