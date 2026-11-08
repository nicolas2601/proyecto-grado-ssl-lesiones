"""ONNX export of the release classifier (preprocessed image -> probabilities)."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from dermssl.release.onnx_tools import INPUT_NAME, OUTPUT_NAME

OPSET = 17
RESOLUTION = 224


def export_onnx(model: torch.nn.Module, path, opset: int = OPSET,
                resolution: int = RESOLUTION,
                output_names: tuple[str, ...] | None = None) -> Path:
    """fp32 graph with a dynamic batch axis; input is the normalised NCHW batch.

    Uses the TorchScript exporter (``dynamo=False`` where the argument
    exists): it needs no onnxscript and handles the plain ViT graph.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, resolution, resolution, dtype=torch.float32)
    names = [OUTPUT_NAME] if output_names is None else list(output_names)
    if len(names) not in (1, 2) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("output_names must contain one or two non-empty names")
    kwargs = dict(
        input_names=[INPUT_NAME], output_names=names, opset_version=opset,
        dynamic_axes={INPUT_NAME: {0: "batch"},
                      **{name: {0: "batch"} for name in names}},
        do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    with torch.no_grad():
        torch.onnx.export(model.eval(), (dummy,), str(path), **kwargs)
    return path
