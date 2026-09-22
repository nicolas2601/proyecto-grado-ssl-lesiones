"""The orchestrator's command must satisfy the contract the trainer enforces.

Two launches died in a row on different mandatory fields - split_hash, then
transformed_data_revision - each costing a bundle and hours of idle GPU.
That is not two oversights: it is one missing check. The orchestrator built
the command, and nothing compared it against what PretrainConfig demands
until a GPU had already been reserved.

This test closes the class instead of the instance. It takes the argv each
arm would actually be launched with, parses it with the trainer's own
parser, builds the config and validates it. Any future mandatory field fails
here, in CI, in milliseconds - not at 07:16 with two GPUs idle.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import run_stage2  # noqa: E402

pretrain = pytest.importorskip("pretrain", reason="needs the trainer's parser")
config_mod = pytest.importorskip("dermssl.pretrain.config")

SHIFT_REVISION = "a754961bfd97bee21b4e48edd35b71314ed82b50cc6355480252b4d075e60e51"


def _settings(tmp_path: Path, archs=("dinov2_vits14", "dinov2_vitb14")) -> run_stage2.Settings:
    return run_stage2.Settings(
        repo=REPO, outputs=tmp_path / "out", python="python",
        ham_root=tmp_path / "ham", archs=list(archs),
    )


def _fixtures(settings: run_stage2.Settings) -> None:
    """The artefacts the placeholders are resolved from."""
    pool = settings.outputs / "tone" / "inputs" / "originals" / "ham10000_p1"
    pool.mkdir(parents=True)
    (pool / "a.png").write_bytes(b"x")

    shift = settings.outputs / "tone" / "shift2"
    shift.mkdir(parents=True)
    (shift / "tone_transform_manifest.json").write_text(
        json.dumps({"revision": SHIFT_REVISION}), encoding="utf-8"
    )


def _trainer_args(argv: list[str]) -> list[str]:
    """Drop the torch.distributed.run wrapper, keep the pretrain.py flags."""
    script = next(i for i, item in enumerate(argv) if item.endswith("pretrain.py"))
    return argv[script + 1:]


def _arms(settings: run_stage2.Settings):
    return [s for s in run_stage2.all_stages(settings) if s.name.startswith("ssl_")]


@pytest.mark.parametrize("arm_index", range(4))
def test_each_arm_builds_a_config_the_trainer_accepts(tmp_path, arm_index):
    settings = _settings(tmp_path)
    _fixtures(settings)
    arm = _arms(settings)[arm_index]

    argv = run_stage2.resolve_argv(arm.argv, settings)
    args = pretrain.parse_args(_trainer_args(argv))
    cfg = pretrain.build_config(args)

    config_mod.validate(cfg)   # raises on any mandatory field the command omits


def test_the_paired_arms_declare_the_manifest_revision(tmp_path):
    settings = _settings(tmp_path)
    _fixtures(settings)
    for arm in (a for a in _arms(settings) if "proposed" in a.name):
        argv = run_stage2.resolve_argv(arm.argv, settings)
        cfg = pretrain.build_config(pretrain.parse_args(_trainer_args(argv)))
        assert cfg.transformed_data_revision == SHIFT_REVISION
        assert cfg.adaptation_mode == "original_transformed_consistency"


def test_the_control_arms_stay_unpaired(tmp_path):
    settings = _settings(tmp_path)
    _fixtures(settings)
    for arm in (a for a in _arms(settings) if "plain" in a.name):
        argv = run_stage2.resolve_argv(arm.argv, settings)
        cfg = pretrain.build_config(pretrain.parse_args(_trainer_args(argv)))
        assert cfg.adaptation_mode == "two_view_ssl"
        assert cfg.split_hash != "0" * 64, "el control entreno con el hash en ceros"


def test_a_missing_revision_in_the_manifest_is_refused(tmp_path):
    settings = _settings(tmp_path)
    _fixtures(settings)
    manifest = settings.outputs / "tone" / "shift2" / "tone_transform_manifest.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        run_stage2.shift_manifest_revision(settings)
