"""Single-view 448 px contract: one CLS4 pass, schema-separated checkpoints and ONNX parity (tiny model)."""
from __future__ import annotations

import pytest
import torch

from dermssl.release import moco_bundle, moco_d4, moco_single
from dermssl.release.plain_dinov2 import PlainDinov2


def _config():
    return {"hidden_size": 8, "num_hidden_layers": 4, "num_attention_heads": 2, "patch_size": 8, "image_size": 16}


def _heads(width):
    return {"HAM": {"weight": torch.randn(7, width), "bias": torch.randn(7)},
            "PAD": {"weight": torch.randn(6, width), "bias": torch.randn(6)}}


def _d4_model(seed=0):
    torch.manual_seed(seed)
    backbone = PlainDinov2.from_config(_config())
    return moco_d4.D4MoCoClassifier(backbone, _heads(4 * backbone.layernorm.normalized_shape[0])).eval()


def test_recipe_is_the_d4_recipe_with_one_view():
    assert moco_single.RECIPE == {**moco_d4.RECIPE, "views": 1, "tta": "none"}
    assert moco_single.RECIPE["resolution"] == 448


def test_features_are_the_normalized_cls4_of_the_identity_view_only():
    model = moco_single.from_d4(_d4_model())
    pixels = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        expected = torch.nn.functional.normalize(model.backbone.forward_cls4(pixels), p=2, dim=-1)
        torch.testing.assert_close(model.features(pixels), expected)


def test_from_d4_keeps_encoder_and_heads_and_can_swap_the_ham_head():
    d4 = _d4_model()
    weight, bias = torch.randn(7, 32), torch.randn(7)
    model = moco_single.from_d4(d4, ham_weight=weight, ham_bias=bias)
    for name, tensor in d4.state_dict().items():
        expected = {"ham_weight": weight, "ham_bias": bias}.get(name, tensor)
        torch.testing.assert_close(model.state_dict()[name], expected)


def test_checkpoint_round_trips_and_rejects_d4_and_v1_checkpoints(tmp_path):
    model = moco_single.from_d4(_d4_model())
    path = moco_single.save_checkpoint(tmp_path / "classifier.pt", model, _config(), {"release": "test"})
    loaded, metadata = moco_single.load_checkpoint(path)
    pixels = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        for a, b in zip(model(pixels), loaded(pixels), strict=True):
            torch.testing.assert_close(a, b)
    assert metadata["schema"] == "dermssl.moco-classifier.v3"
    assert metadata["recipe"]["tta"] == "none"
    d4_path = moco_d4.save_checkpoint(tmp_path / "d4.pt", _d4_model(), _config(), {"release": "test"})
    with pytest.raises(ValueError, match="schema"):
        moco_single.load_checkpoint(d4_path)
    v1 = moco_bundle.save_checkpoint(tmp_path / "v1.pt", model, _config(), {"release": "test"})
    with pytest.raises(ValueError, match="schema"):
        moco_single.load_checkpoint(v1)


def test_single_view_graph_exports_to_onnx_and_matches_torch(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    from dermssl.release import torch_export
    model = moco_single.from_d4(_d4_model())
    path = torch_export.export_onnx(model, tmp_path / "single.onnx", resolution=16,
                                    output_names=moco_bundle.OUTPUT_NAMES)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pixels = torch.randn(3, 3, 16, 16)
    with torch.no_grad():
        expected = model(pixels)
    actual = session.run(list(moco_bundle.OUTPUT_NAMES), {"pixel_values": pixels.numpy()})
    for want, got in zip(expected, actual, strict=True):
        assert abs(want.numpy() - got).max() < 1e-4
