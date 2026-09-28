"""ONNX export of the release classifier (preprocessed image -> probabilities)."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from dermssl.release.onnx_tools import INPUT_NAME, OUTPUT_NAME

OPSET = 17
RESOLUTION = 224


def export_onnx(model: torch.nn.Module, path, opset: int = OPSET,
                resolution: int = RESOLUTION) -> Path:
    """fp32 graph with a dynamic batch axis; input is the normalised NCHW batch.

    Uses the TorchScript exporter (``dynamo=False`` where the argument
    exists): it needs no onnxscript and handles the plain ViT graph.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32)
    kwargs = dict(
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME], opset_version=opset,
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(model.eval(), (dummy,), str(path), **kwargs)
    return path
