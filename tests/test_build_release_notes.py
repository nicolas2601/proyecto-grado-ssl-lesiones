"""A published checkpoint must carry the run that produced it.

These tests pin two things: the notes describe the models actually in the
release (read from each run.json, never transcribed), and a weights file
without its manifest stops the publication instead of shipping an
unreproducible artifact.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_release_notes as gen  # noqa: E402


def _run(**overrides):
    payload = {
        "backbone": "dinov2_vits14",
        "adaptation_mode": "paired",
        "budget_gpu_hours": 12.0,
        "seed": 42,
        "label_free": True,
        "execution": {"steps_executed": 2045},
    }
    payload.update(overrides)
    return payload


def _flat_arm(directory: Path, arm: str, **overrides):
    (directory / f"{arm}_backbone.pt").write_bytes(b"weights")
    (directory / f"{arm}_run.json").write_text(json.dumps(_run(**overrides)), encoding="utf-8")


def test_a_checkpoint_without_its_manifest_is_refused(tmp_path):
    (tmp_path / "ssl_proposed_vits14_backbone.pt").write_bytes(b"weights")
    with pytest.raises(gen.MissingEvidence, match="run.json"):
        gen.discover_arms(tmp_path)


def test_an_empty_directory_is_refused_rather_than_publishing_nothing(tmp_path):
    with pytest.raises(gen.MissingEvidence, match="publicable"):
        gen.discover_arms(tmp_path)


def test_the_flat_artifact_layout_is_discovered(tmp_path):
    _flat_arm(tmp_path, "ssl_plain_vits14")
    _flat_arm(tmp_path, "ssl_proposed_vitb14")
    arms = gen.discover_arms(tmp_path)
    assert set(arms) == {"ssl_plain_vits14", "ssl_proposed_vitb14"}


def test_the_nested_training_layout_is_discovered(tmp_path):
    arm = tmp_path / "ssl_proposed_vitb14"
    arm.mkdir()
    (arm / gen.NESTED_WEIGHTS).write_bytes(b"weights")
    (arm / gen.NESTED_RUN).write_text(json.dumps(_run()), encoding="utf-8")
    arms = gen.discover_arms(tmp_path)
    assert arms["ssl_proposed_vitb14"]["weights"].name == gen.NESTED_WEIGHTS


def test_a_nested_checkpoint_without_its_manifest_is_refused(tmp_path):
    arm = tmp_path / "ssl_plain_vits14"
    arm.mkdir()
    (arm / gen.NESTED_WEIGHTS).write_bytes(b"weights")
    with pytest.raises(gen.MissingEvidence, match="run.json"):
        gen.discover_arms(tmp_path)


def test_the_architecture_comes_from_the_manifest_not_from_prose(tmp_path):
    _flat_arm(tmp_path, "ssl_proposed_vitb14", backbone="dinov2_vitb14")
    notes = gen.render_notes("v1.0", gen.discover_arms(tmp_path), "")
    assert "dinov2_vitb14" in notes
    assert "ViT-S/14" not in notes, "the old notes hardcoded one architecture"


def test_every_published_arm_gets_a_row(tmp_path):
    _flat_arm(tmp_path, "ssl_plain_vits14", adaptation_mode="none")
    _flat_arm(tmp_path, "ssl_proposed_vits14")
    _flat_arm(tmp_path, "ssl_plain_vitb14", backbone="dinov2_vitb14", adaptation_mode="none")
    _flat_arm(tmp_path, "ssl_proposed_vitb14", backbone="dinov2_vitb14")
    rows = [line for line in gen.arm_table(gen.discover_arms(tmp_path)) if line.startswith("| `")]
    assert len(rows) == 4


def test_the_label_free_claim_is_dropped_when_a_manifest_denies_it(tmp_path):
    _flat_arm(tmp_path, "ssl_plain_vits14")
    _flat_arm(tmp_path, "ssl_proposed_vits14", label_free=False)
    notes = gen.render_notes("v1.0", gen.discover_arms(tmp_path), "")
    assert "preentrenados sin etiquetas" not in notes
    assert "revisar `label_free`" in notes


def test_the_checksums_are_embedded_verbatim(tmp_path):
    _flat_arm(tmp_path, "ssl_plain_vits14")
    notes = gen.render_notes("v1.0", gen.discover_arms(tmp_path), "abc123  ./x.pt\n")
    assert "abc123  ./x.pt" in notes


def test_a_missing_field_renders_a_dash_not_a_crash(tmp_path):
    (tmp_path / "ssl_plain_vits14_backbone.pt").write_bytes(b"weights")
    (tmp_path / "ssl_plain_vits14_run.json").write_text("{}", encoding="utf-8")
    row = [line for line in gen.arm_table(gen.discover_arms(tmp_path)) if line.startswith("| `")][0]
    assert "| - |" in row


def test_the_mode_legend_names_the_modes_the_manifests_carry(tmp_path):
    """The legend used to hardcode `paired`; the arms report
    `original_transformed_consistency`, so the word matched nothing in the
    table. It now derives from the data."""
    _flat_arm(tmp_path, "ssl_plain_vitb14", backbone="dinov2_vitb14",
              adaptation_mode="two_view_ssl")
    _flat_arm(tmp_path, "ssl_proposed_vitb14", backbone="dinov2_vitb14",
              adaptation_mode="original_transformed_consistency")
    notes = gen.render_notes("v1.0", gen.discover_arms(tmp_path), "")
    assert "`original_transformed_consistency`" in notes
    assert "`two_view_ssl`" in notes
    assert "`paired`" not in notes


def test_the_legend_does_not_invent_a_method_for_an_unknown_mode(tmp_path):
    _flat_arm(tmp_path, "ssl_plain_vits14", adaptation_mode="something_new")
    notes = gen.render_notes("v1.0", gen.discover_arms(tmp_path), "")
    assert "`something_new`" in notes
    assert "metodo propuesto" not in notes
