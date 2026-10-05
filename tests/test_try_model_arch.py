"""The architecture comes from the weights' own manifest, not from a default.

try_model.py is the first thing anyone runs after downloading a checkpoint.
More than one architecture is published, and the old fixed ViT-S default made
a ViT-B download fail with a shape error that reads like a corrupt file
instead of a missing flag.

These tests need no torch: resolve_arch is pure path and JSON work, and
try_model imports torch lazily inside the functions that use it.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import try_model  # noqa: E402


def _flat(tmp_path: Path, arm: str, backbone: str) -> Path:
    weights = tmp_path / f"{arm}_backbone.pt"
    weights.write_bytes(b"weights")
    (tmp_path / f"{arm}_run.json").write_text(
        json.dumps({"backbone": backbone}), encoding="utf-8"
    )
    return weights


def test_the_architecture_is_read_from_the_flat_manifest(tmp_path):
    weights = _flat(tmp_path, "ssl_proposed_vitb14", "dinov2_vitb14")
    assert try_model.resolve_arch(str(weights), None) == "dinov2_vitb14"


def test_the_architecture_is_read_from_the_nested_manifest(tmp_path):
    arm = tmp_path / "ssl_plain_vitb14"
    arm.mkdir()
    weights = arm / "simsiam_dinov2_backbone_final.pt"
    weights.write_bytes(b"weights")
    (arm / "run.json").write_text(json.dumps({"backbone": "dinov2_vitb14"}), encoding="utf-8")
    assert try_model.resolve_arch(str(weights), None) == "dinov2_vitb14"


def test_an_explicit_flag_wins_over_the_manifest(tmp_path):
    weights = _flat(tmp_path, "ssl_proposed_vitb14", "dinov2_vitb14")
    assert try_model.resolve_arch(str(weights), "dinov2_vits14") == "dinov2_vits14"


def test_without_a_manifest_it_falls_back_to_the_default(tmp_path):
    weights = tmp_path / "suelto_backbone.pt"
    weights.write_bytes(b"weights")
    assert try_model.resolve_arch(str(weights), None) == try_model.ARCH


def test_the_public_weights_use_the_default(tmp_path):
    assert try_model.resolve_arch("public", None) == try_model.ARCH


def test_a_manifest_without_the_field_does_not_crash(tmp_path):
    weights = tmp_path / "ssl_plain_vits14_backbone.pt"
    weights.write_bytes(b"weights")
    (tmp_path / "ssl_plain_vits14_run.json").write_text("{}", encoding="utf-8")
    assert try_model.resolve_arch(str(weights), None) == try_model.ARCH


def test_the_arm_manifest_is_preferred_over_a_generic_one(tmp_path):
    weights = _flat(tmp_path, "ssl_proposed_vitb14", "dinov2_vitb14")
    (tmp_path / "run.json").write_text(
        json.dumps({"backbone": "dinov2_vits14"}), encoding="utf-8"
    )
    assert try_model.resolve_arch(str(weights), None) == "dinov2_vitb14", (
        "con varios checkpoints en un directorio, el run.json del brazo manda"
    )
