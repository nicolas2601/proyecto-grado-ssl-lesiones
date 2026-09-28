"""S4-UNet (docs/27): frozen DINOv2 + a light DPT-style decoder.

A tiny fake ViT stands in for DINOv2: it exposes the same
``get_intermediate_layers`` contract, so no real weights and no GPU are
needed. The tests pin what the registration fixes: blocks 3, 6, 9 and 12,
a frozen encoder, and only the decoder in the checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from dermssl import torch_s4_unet as s4  # noqa: E402
from s4_fakes import FakeVit  # noqa: E402

pytestmark = pytest.mark.torch


def _model(**kwargs):
    torch.manual_seed(0)
    return s4.S4UNet(FakeVit(), features=16, **kwargs)


def test_registered_blocks_are_3_6_9_12_one_based():
    assert s4.REGISTERED_BLOCKS == (3, 6, 9, 12)
    assert s4.zero_based(s4.REGISTERED_BLOCKS) == [2, 5, 8, 11]


def test_the_encoder_asks_dinov2_for_the_registered_blocks_reshaped_and_normed():
    model = _model()
    model(torch.zeros(1, 3, 56, 56))
    assert model.backbone.requested == [((2, 5, 8, 11), True, True)]


@pytest.mark.parametrize("size", [56, 224])
def test_logits_come_back_at_the_input_resolution(size):
    logits = _model()(torch.randn(2, 3, size, size))
    assert logits.shape == (2, 1, size, size)


def test_only_the_decoder_trains_and_the_backbone_gets_no_gradient():
    model = _model()
    model(torch.randn(2, 3, 56, 56)).mean().backward()
    assert all(not p.requires_grad and p.grad is None for p in model.backbone.parameters())
    assert all(p.grad is not None for p in model.decoder.parameters())
    assert s4.trainable_parameters(model) == sum(p.numel() for p in model.decoder.parameters())


def test_train_mode_never_switches_the_frozen_backbone_out_of_eval():
    model = _model().train()
    assert model.decoder.training and not model.backbone.training


def test_blocks_other_than_four_distinct_ascending_indices_are_refused():
    for blocks in ((3, 6, 9), (3, 3, 9, 12), (12, 9, 6, 3), (0, 6, 9, 12), (3, 6, 9, 13)):
        with pytest.raises(ValueError):
            s4.S4UNet(FakeVit(), blocks=blocks, features=16)


def test_input_not_divisible_by_the_patch_size_is_refused():
    with pytest.raises(ValueError, match="patch"):
        _model()(torch.zeros(1, 3, 50, 50))


def test_checkpoint_holds_the_decoder_only_and_reloads_to_the_same_output():
    model = _model().eval()
    config = s4.checkpoint_config(
        model, backbone_arch="dinov2_vitb14", backbone_source="public",
        backbone_sha256="public", resolution=56,
    )
    state = {"model": s4.decoder_state(model), "config": config}
    assert not any(key.startswith("backbone") for key in state["model"])
    assert config["arch"] == s4.ARCH and config["blocks"] == [3, 6, 9, 12]

    backbone = model.backbone
    reloaded = s4.model_from_checkpoint(state, lambda arch, source: backbone)
    x = torch.randn(1, 3, 56, 56)
    assert torch.equal(model(x), reloaded(x))
    assert not reloaded.training


def test_the_vit_b_decoder_is_lighter_than_the_student_unet():
    """docs/27 asks for a light decoder; the student U-Net has 7.77 M parameters."""
    decoder = s4.DPTDecoder(768)
    assert sum(p.numel() for p in decoder.parameters()) < 7_770_000
