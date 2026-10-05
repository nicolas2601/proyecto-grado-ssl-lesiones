"""Write the release forms of one model and run every form on the same images.

Layout of ``<model_dir>``::

    hf/config.json, hf/model.safetensors   Dinov2Model (forms a and b)
    hf/head.safetensors                    fused head W', b' + classes + preprocessing
    onnx/model.onnx, onnx/model_fp16.onnx  end-to-end classifier (form c)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dermssl.release import fuse, onnx_tools, package, plain_dinov2, torch_convert, torch_export


def write_model_forms(hub_model, weight, bias, classes, out_dir, preprocessing: dict,
                      fp16: bool = True) -> dict:
    out_dir = Path(out_dir)
    hf_dir = out_dir / "hf"
    torch_convert.save_hf_backbone(hub_model, hf_dir)
    package.write_head(hf_dir / "head.safetensors", weight, bias, classes, preprocessing)
    classifier, _ = plain_dinov2.load_classifier(hf_dir)
    paths = {"hf": hf_dir, "onnx": torch_export.export_onnx(classifier, out_dir / "onnx" / "model.onnx")}
    if fp16:
        paths["onnx_fp16"] = onnx_tools.convert_fp16(paths["onnx"], out_dir / "onnx" / "model_fp16.onnx")
    return paths


def hub_probas(hub_model, weight, bias, batch: np.ndarray, device="cpu") -> np.ndarray:
    """The evaluated code path in fp32: hub forward (CLS) -> fused head (float64)."""
    with torch.no_grad():
        feats = hub_model.to(device)(torch.from_numpy(batch).to(device)).float().cpu().numpy()
    return fuse.fused_predict_proba(feats, weight, bias)


class FormRunner:
    """Loads the torch (a), transformers (b) and ONNX (c) forms of one model dir."""

    def __init__(self, model_dir, device="cpu", with_fp16: bool = True):
        model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.plain, _ = plain_dinov2.load_classifier(model_dir / "hf")
        self.plain.to(self.device)
        head = package.read_head(model_dir / "hf" / "head.safetensors")
        self.weight, self.bias = head["weight"].astype(np.float64), head["bias"].astype(np.float64)
        self.hf = _load_hf(model_dir / "hf").to(self.device)
        self.onnx = onnx_tools.session(model_dir / "onnx" / "model.onnx")
        fp16_path = model_dir / "onnx" / "model_fp16.onnx"
        self.onnx_fp16 = onnx_tools.session(fp16_path) if with_fp16 and fp16_path.exists() else None

    def predict(self, batch: np.ndarray) -> dict:
        x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            out = {"torch": self.plain(x).double().cpu().numpy()}
            pooled = self.hf(pixel_values=x).pooler_output.float().cpu().numpy()
        out["transformers"] = fuse.fused_predict_proba(pooled, self.weight, self.bias)
        out["onnx"] = onnx_tools.predict(self.onnx, batch)
        if self.onnx_fp16 is not None:
            out["onnx_fp16"] = onnx_tools.predict(self.onnx_fp16, batch)
        return out


def _load_hf(hf_dir):
    from transformers import AutoModel  # noqa: PLC0415

    return AutoModel.from_pretrained(str(hf_dir)).eval()


def run_forms(runner: FormRunner, batches) -> dict:
    """Concatenate every form's probabilities over an iterable of batches."""
    collected: dict = {}
    for batch in batches:
        for name, proba in runner.predict(batch).items():
            collected.setdefault(name, []).append(proba)
    return {name: np.concatenate(parts) for name, parts in collected.items()}
