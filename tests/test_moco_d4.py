"""D4 candidate contract: views, averaged CLS4 features and schema-separated checkpoints (tiny model)."""
from __future__ import annotations

import pytest
import torch

from dermssl.release import moco_bundle, moco_d4
from dermssl.release.plain_dinov2 import PlainDinov2


def _config():
    return {"hidden_size": 8, "num_hidden_layers": 4, "num_attention_heads": 2, "patch_size": 8, "image_size": 16}


def _model(seed=0):
    torch.manual_seed(seed)
    backbone = PlainDinov2.from_config(_config())
    width = 4 * backbone.layernorm.normalized_shape[0]
    heads = {"HAM": {"weight": torch.randn(7, width), "bias": torch.randn(7)},
             "PAD": {"weight": torch.randn(6, width), "bias": torch.randn(6)}}
    return moco_d4.D4MoCoClassifier(backbone, heads).eval()


def test_d4_produces_eight_distinct_dihedral_views():
    pixels = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    views = moco_d4.d4_views(pixels)
    assert len(views) == 8
    assert torch.equal(views[0], pixels)
    assert torch.equal(views[1], torch.flip(pixels, dims=(3,)))
    assert torch.equal(views[2], torch.rot90(pixels, 1, dims=(2, 3)))
    assert len({tuple(v.flatten().tolist()) for v in views}) == 8


def test_features_are_normalized_mean_of_cls4_over_all_views():
    model = _model()
    pixels = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        expected = torch.stack([model.backbone.forward_cls4(v) for v in moco_d4.d4_views(pixels)]).mean(0)
        expected = torch.nn.functional.normalize(expected, p=2, dim=-1)
        actual = model.features(pixels)
    torch.testing.assert_close(actual, expected)


def test_d4_checkpoint_round_trips_and_rejects_v1_checkpoints(tmp_path):
    model = _model()
    path = moco_d4.save_checkpoint(tmp_path / "classifier.pt", model, _config(), {"release": "test"})
    loaded, metadata = moco_d4.load_checkpoint(path)
    pixels = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        for a, b in zip(model(pixels), loaded(pixels), strict=True):
            torch.testing.assert_close(a, b)
    assert metadata["recipe"]["tta"] == "d4-mean"
    v1_model = moco_bundle.MoCoClassifier(PlainDinov2.from_config(_config()), {
        "HAM": {"weight": torch.zeros(7, 32), "bias": torch.zeros(7)},
        "PAD": {"weight": torch.zeros(6, 32), "bias": torch.zeros(6)}})
    v1_path = moco_bundle.save_checkpoint(tmp_path / "v1.pt", v1_model, _config(), {"release": "test"})
    with pytest.raises(ValueError, match="schema"):
        moco_d4.load_checkpoint(v1_path)


def test_d4_graph_exports_to_onnx_and_matches_torch(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    from dermssl.release import torch_export
    model = _model()
    path = torch_export.export_onnx(model, tmp_path / "d4.onnx", resolution=16,
                                    output_names=moco_bundle.OUTPUT_NAMES)
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pixels = torch.randn(3, 3, 16, 16)
    with torch.no_grad():
        expected = model(pixels)
    actual = session.run(list(moco_bundle.OUTPUT_NAMES), {"pixel_values": pixels.numpy()})
    for want, got in zip(expected, actual, strict=True):
        assert abs(want.numpy() - got).max() < 1e-4


def test_exportable_rotation_matches_torch_rot90_for_all_quarter_turns():
    pixels = torch.randn(2, 3, 5, 7)
    for k in range(4):
        assert torch.equal(moco_d4._rot90(pixels, k), torch.rot90(pixels, k, dims=(2, 3)))
