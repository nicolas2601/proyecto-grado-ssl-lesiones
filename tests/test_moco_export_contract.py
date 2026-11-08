import torch

from dermssl.release.onnx_tools import INPUT_NAME, OUTPUT_NAME
from dermssl.release.torch_export import export_onnx


def test_export_supports_two_named_outputs_with_dynamic_batch_axes(monkeypatch, tmp_path):
    calls = {}

    def fake_export(model, args, f, input_names=None, output_names=None,
                    opset_version=None, dynamic_axes=None, do_constant_folding=None,
                    dynamo=True):
        calls.update(locals())

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    path = tmp_path / "nested" / "model.onnx"
    names = ("ham7_probabilities", "pad6_probabilities")

    assert export_onnx(torch.nn.Identity(), path, output_names=names) == path
    assert calls["output_names"] == list(names)
    assert calls["dynamic_axes"] == {
        INPUT_NAME: {0: "batch"}, names[0]: {0: "batch"}, names[1]: {0: "batch"}
    }
    assert calls["dynamo"] is False


def test_export_keeps_single_legacy_output_by_default(monkeypatch, tmp_path):
    calls = {}

    def fake_export(model, args, f, input_names=None, output_names=None,
                    opset_version=None, dynamic_axes=None, do_constant_folding=None,
                    dynamo=True):
        calls.update(locals())

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    export_onnx(torch.nn.Identity(), tmp_path / "model.onnx")

    assert calls["output_names"] == [OUTPUT_NAME]
    assert calls["dynamic_axes"][OUTPUT_NAME] == {0: "batch"}
