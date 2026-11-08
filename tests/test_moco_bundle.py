import torch
import pytest

from dermssl.release.moco_bundle import (
    MoCoClassifier,
    OUTPUT_NAMES,
    RECIPE,
    TASK_CLASSES,
    load_checkpoint,
    save_checkpoint,
)
from dermssl.release.plain_dinov2 import PlainDinov2


def _model(depth=4):
    backbone = PlainDinov2(8, depth, 2, 8, 16)
    features = 4 * backbone.layernorm.normalized_shape[0]
    heads = {
        "HAM": {"weight": torch.randn(7, features), "bias": torch.randn(7)},
        "PAD": {"weight": torch.randn(6, features), "bias": torch.randn(6)},
    }
    return MoCoClassifier(backbone, heads)


def _config(depth=4):
    return {
        "hidden_size": 8,
        "num_hidden_layers": depth,
        "num_attention_heads": 2,
        "patch_size": 8,
        "image_size": 16,
    }


def test_two_heads_return_normalized_probabilities_and_fixed_contract():
    model = _model().eval()
    ham, pad = model(torch.randn(2, 3, 16, 16))

    assert OUTPUT_NAMES == ("ham7_probabilities", "pad6_probabilities")
    assert tuple(TASK_CLASSES) == ("HAM", "PAD")
    assert ham.shape == (2, 7)
    assert pad.shape == (2, 6)
    torch.testing.assert_close(ham.sum(-1), torch.ones(2))
    torch.testing.assert_close(pad.sum(-1), torch.ones(2))
    torch.testing.assert_close(model.features(torch.randn(2, 3, 16, 16)).norm(dim=-1), torch.ones(2))


def test_checkpoint_recipe_uses_tau_for_disabled_prior_correction():
    assert RECIPE["tau"] == 0.0
    assert RECIPE["softmax_temperature"] == 1.0
    assert "temperature" not in RECIPE


def test_cls4_concatenates_final_four_normalized_block_cls_and_preserves_legacy_forward():
    backbone = PlainDinov2(8, 6, 2, 8, 16).eval()
    pixels = torch.randn(1, 3, 16, 16)
    tokens = backbone.embeddings(pixels)
    expected = []
    for index, layer in enumerate(backbone.encoder.layer):
        tokens = layer(tokens)
        if index >= 2:
            expected.append(backbone.layernorm(tokens)[:, 0])

    torch.testing.assert_close(backbone.forward_cls4(pixels), torch.cat(expected, dim=-1))
    torch.testing.assert_close(backbone(pixels), backbone.layernorm(tokens)[:, 0])


def test_checkpoint_roundtrip_restores_backbone_and_both_heads(tmp_path):
    model = _model().eval()
    path = tmp_path / "moco.pt"
    save_checkpoint(path, model, _config(), {"seed": 12})
    restored, metadata = load_checkpoint(path)
    pixels = torch.randn(1, 3, 16, 16)

    assert metadata["schema"] == "dermssl.moco-classifier.v1"
    assert metadata["classes"] == {key: list(value) for key, value in TASK_CLASSES.items()}
    assert "state_dict" not in metadata
    for actual, expected in zip(restored(pixels), model(pixels)):
        torch.testing.assert_close(actual, expected)


def test_checkpoint_is_exclusive_and_rejects_invalid_head_depth_or_nonfinite(tmp_path):
    model = _model()
    path = tmp_path / "moco.pt"
    save_checkpoint(path, model, _config(), {})
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, _config(), {})

    with pytest.raises(ValueError, match="four"):
        _model(depth=3)
    heads = {
        "HAM": {"weight": torch.full((7, 32), float("nan")), "bias": torch.zeros(7)},
        "PAD": {"weight": torch.zeros(6, 32), "bias": torch.zeros(6)},
    }
    with pytest.raises(ValueError, match="finite"):
        MoCoClassifier(PlainDinov2(8, 4, 2, 8, 16), heads)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("classes", {"HAM": ["wrong"], "PAD": []}, "classes"),
     ("recipe", {"features": "cls"}, "recipe")],
)
def test_checkpoint_rejects_class_or_recipe_contract_changes(tmp_path, field, value, message):
    path = tmp_path / "moco.pt"
    save_checkpoint(path, _model(), _config(), {})
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(path)


@pytest.mark.parametrize("corruption", ["nonfinite", "shape", "missing"])
def test_checkpoint_rejects_nonfinite_or_corrupt_model_state(tmp_path, corruption):
    path = tmp_path / "moco.pt"
    save_checkpoint(path, _model(), _config(), {})
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    if corruption == "nonfinite":
        state["ham_bias"][0] = float("inf")
    elif corruption == "shape":
        state["pad_weight"] = torch.zeros(6, 31)
    else:
        del state["backbone.embeddings.cls_token"]
    torch.save(checkpoint, path)

    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_constructor_rejects_incorrect_head_shape():
    backbone = PlainDinov2(8, 4, 2, 8, 16)
    heads = {
        "HAM": {"weight": torch.zeros(7, 31), "bias": torch.zeros(7)},
        "PAD": {"weight": torch.zeros(6, 32), "bias": torch.zeros(6)},
    }

    with pytest.raises(ValueError, match="shapes"):
        MoCoClassifier(backbone, heads)


def test_checkpoint_save_rejects_nonprimitive_metadata_and_mismatched_config(tmp_path):
    model = _model()
    with pytest.raises(ValueError, match="primitive"):
        save_checkpoint(tmp_path / "invalid.pt", model, _config(), {"opaque": object()})
    with pytest.raises(ValueError, match="config"):
        save_checkpoint(tmp_path / "mismatch.pt", model, _config(depth=5), {})
    assert not (tmp_path / "invalid.pt").exists()
    assert not (tmp_path / "mismatch.pt").exists()


def test_checkpoint_load_rejects_nonprimitive_provenance(tmp_path):
    path = tmp_path / "moco.pt"
    save_checkpoint(path, _model(), _config(), {})
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["provenance"] = {"tensor": torch.zeros(1)}
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="primitive"):
        load_checkpoint(path)
