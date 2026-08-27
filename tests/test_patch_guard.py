"""The 256-vs-14 trap must fail loudly, and snapping must stay conservative."""

import pytest

from dermssl import patch_guard as pg


def test_dinov2_patch_size_is_14():
    assert pg.patch_size_for("dinov2_vitb14") == 14
    assert pg.patch_size_for("vit_base_patch16_224") == 16


def test_resnet_has_no_patch_constraint():
    assert pg.patch_size_for("resnet50") is None
    assert pg.is_compatible(257, None) is True
    assert pg.assert_compatible(257, "resnet50") == 257


def test_unknown_backbone_falls_back_to_name_heuristic():
    assert pg.patch_size_for("some_new_vit14") == 14
    assert pg.patch_size_for("mystery_net") is None


def test_224_is_valid_for_patch_14():
    assert 224 % 14 == 0
    assert pg.assert_compatible(224, "dinov2_vits14") == 224


def test_256_is_rejected_for_patch_14():
    """256 / 14 = 18.28... - the exact bug this guard exists to prevent."""
    with pytest.raises(pg.PatchSizeError) as err:
        pg.assert_compatible(256, "dinov2_vits14")
    message = str(err.value)
    assert "252" in message and "266" in message  # both fixes are suggested


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("down", 252), ("up", 266), ("nearest", 252)],
)
def test_snap_modes(mode, expected):
    assert pg.snap_to_patch(256, 14, mode) == expected


def test_snap_nearest_rounds_up_when_closer():
    assert pg.snap_to_patch(265, 14, "nearest") == 266


def test_snap_never_returns_zero():
    assert pg.snap_to_patch(5, 14, "down") == 14


def test_snap_is_identity_when_already_valid():
    for mode in ("down", "up", "nearest"):
        assert pg.snap_to_patch(224, 14, mode) == 224


def test_resolve_resolution_reports_the_change():
    res, note = pg.resolve_resolution(256, "dinov2_vitb14")
    assert res == 252
    assert "256 -> 252" in note

    res, note = pg.resolve_resolution(224, "dinov2_vitb14")
    assert (res, note) == (224, "ok")


def test_resolve_resolution_leaves_resnet_alone():
    assert pg.resolve_resolution(256, "resnet50") == (256, "ok")


def test_num_patch_tokens():
    assert pg.num_patch_tokens(224, 14) == 16 * 16
    assert pg.num_patch_tokens(224, None) == 0


def test_invalid_mode_and_resolution_are_errors():
    with pytest.raises(ValueError):
        pg.snap_to_patch(224, 14, "sideways")
    with pytest.raises(pg.PatchSizeError):
        pg.snap_to_patch(0, 14)


def test_classic_cnns_have_no_patch_constraint():
    """alexnet/vgg19/efficientnet_b0 are convolutional: any resolution is
    valid and vgg19's trailing digits must not trigger the name heuristic."""
    for name in ("alexnet", "vgg19", "efficientnet_b0"):
        assert pg.patch_size_for(name) is None, name
        res, _ = pg.resolve_resolution(224, name)
        assert res == 224
