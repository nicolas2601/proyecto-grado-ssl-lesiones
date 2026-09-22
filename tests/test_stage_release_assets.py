"""Four arms, four distinct checkpoints in the release.

All four training runs write the same file name. Copied naively into one
directory they overwrite each other and the release still looks right by
file count, so these tests pin the renaming and the refusal to clobber.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import stage_release_assets as stage  # noqa: E402
from build_release_notes import MissingEvidence, discover_arms  # noqa: E402

ARMS = ("ssl_plain_vits14", "ssl_proposed_vits14", "ssl_plain_vitb14", "ssl_proposed_vitb14")


def _nested_run(root: Path, arm: str, payload: bytes):
    directory = root / arm
    directory.mkdir()
    (directory / "simsiam_dinov2_backbone_final.pt").write_bytes(payload)
    (directory / "run.json").write_text(json.dumps({"backbone": "dinov2_vits14"}), encoding="utf-8")


def test_four_arms_keep_four_distinct_checkpoints(tmp_path):
    src, staging = tmp_path / "src", tmp_path / "staging"
    src.mkdir()
    staging.mkdir()
    for index, arm in enumerate(ARMS):
        _nested_run(src, arm, f"weights-{index}".encode())

    stage.stage_arms(discover_arms(src), staging)

    checkpoints = sorted(staging.glob("*_backbone.pt"))
    assert len(checkpoints) == 4
    contents = {path.read_bytes() for path in checkpoints}
    assert len(contents) == 4, "un arm piso a otro: el release llevaria el mismo peso repetido"


def test_each_checkpoint_is_named_after_its_arm(tmp_path):
    src, staging = tmp_path / "src", tmp_path / "staging"
    src.mkdir()
    staging.mkdir()
    _nested_run(src, "ssl_proposed_vitb14", b"weights")

    stage.stage_arms(discover_arms(src), staging)

    assert (staging / "ssl_proposed_vitb14_backbone.pt").is_file()
    assert (staging / "ssl_proposed_vitb14_run.json").is_file()


def test_a_name_collision_stops_the_staging(tmp_path):
    src, staging = tmp_path / "src", tmp_path / "staging"
    src.mkdir()
    staging.mkdir()
    _nested_run(src, "ssl_plain_vits14", b"weights")
    (staging / "ssl_plain_vits14_backbone.pt").write_bytes(b"algo previo")

    with pytest.raises(MissingEvidence, match="colisionan"):
        stage.stage_arms(discover_arms(src), staging)


def test_the_bank_outputs_travel_with_the_models(tmp_path):
    src, staging = tmp_path / "src", tmp_path / "staging"
    (src / "bank").mkdir(parents=True)
    staging.mkdir()
    for name in ("bank.json", "bank_rows.csv", "strata.json", "bank.md"):
        (src / "bank" / name).write_text("x", encoding="utf-8")

    written = stage.stage_bank(src, staging)

    assert len(written) == 4
    assert (staging / "bank_rows.csv").is_file()


def test_missing_bank_outputs_are_not_fatal(tmp_path):
    src, staging = tmp_path / "src", tmp_path / "staging"
    src.mkdir()
    staging.mkdir()
    assert stage.stage_bank(src, staging) == []
