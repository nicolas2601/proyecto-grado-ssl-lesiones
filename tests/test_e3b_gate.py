"""E3b refuses to run unless the pinned E3 selection names F4 (docs/33 section 3, E3b)."""

import json

import pytest

from dermssl import e3_store as st
from dermssl.e3b import protocol as proto


def write_selection(tmp_path, winner="F4", decision="adopt"):
    path = tmp_path / "e3_selection.json"
    path.write_text(json.dumps({"decision": decision, "winner": winner, "complete": True}), encoding="utf-8")
    return path, st.file_sha256(path)


def test_the_committed_selection_is_the_one_this_stage_was_written_for():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "resultados" / "banco_completo" / "e3" / "e3_selection.json"
    doc = proto.require_pinned_selection(path, proto.PINNED_SELECTION_SHA256)
    assert doc["winner"] == proto.ADOPTED_VARIANT == "F4"


def test_a_pinned_selection_with_the_expected_winner_is_accepted(tmp_path):
    path, sha = write_selection(tmp_path)
    assert proto.require_pinned_selection(path, sha)["winner"] == "F4"


def test_a_wrong_sha_a_missing_file_or_another_winner_are_refused(tmp_path):
    path, sha = write_selection(tmp_path)
    with pytest.raises(st.SelectionPinError):
        proto.require_pinned_selection(path, "0" * 64)
    with pytest.raises(st.SelectionMissingError):
        proto.require_pinned_selection(tmp_path / "nope.json", sha)
    other, other_sha = write_selection(tmp_path, winner="F2_448")
    with pytest.raises(proto.WinnerMismatchError, match="F2_448"):
        proto.require_pinned_selection(other, other_sha)


def test_a_selection_without_adoption_is_refused(tmp_path):
    path, sha = write_selection(tmp_path, decision="keep_reference")
    with pytest.raises(proto.WinnerMismatchError):
        proto.require_pinned_selection(path, sha)


def test_widths_come_from_the_frozen_bank_protocol():
    assert set(proto.MODELS) == {"alexnet", "vgg19", "resnet50", "efficientnet_b0", "vit_b16", "dinov2_vits14",
                                 "dinov2_vitb14", "simsiam_r50"}
    assert proto.FEATURE_DIMS["vgg19"] == 512
    with pytest.raises(proto.FeatureWidthError, match="4096"):
        proto.check_width("vgg19", 4096)
    proto.check_width("vgg19", 512)
