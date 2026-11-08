"""DINOv2 ViT in plain PyTorch, reading Hugging Face ``Dinov2Model`` weights.

Standalone (torch + safetensors only) so it ships verbatim in the release:
the torch form loads the same ``model.safetensors`` the transformers form
uses, without installing transformers. Supports the release's own layout
only: 224 x 224 input, the position grid stored already interpolated, MLP
feed-forward, no register tokens.

    from dinov2_plain import load_classifier
    model, classes = load_classifier("B_s42/hf")
    probabilities = model(batch)          # batch: float32 (N, 3, 224, 224)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        # Python numbers, not traced shape arithmetic: a traced ``dim // heads``
        # exports Div + Cast(int64) chains that break the fp16 ONNX copy.
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim ** -0.5
        self.query, self.key, self.value = (nn.Linear(dim, dim) for _ in range(3))

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        return t.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Explicit softmax(q k^T / sqrt(d)) v, as the hub computes it without xFormers.
        query = self._split(self.query(x)) * self.scale
        weights = (query @ self._split(self.key(x)).transpose(-2, -1)).softmax(dim=-1)
        out = weights @ self._split(self.value(x))
        return out.transpose(1, 2).flatten(2)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention = _SelfAttention(dim, heads)
        self.output = nn.Module()
        self.output.dense = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output.dense(self.attention(x))


class _LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Layer(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, eps: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=eps), nn.LayerNorm(dim, eps=eps)
        self.attention, self.mlp = _Attention(dim, heads), _Mlp(dim, hidden)
        self.layer_scale1, self.layer_scale2 = _LayerScale(dim), _LayerScale(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x)))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class _Embeddings(nn.Module):
    def __init__(self, dim: int, patch: int, grid: int, channels: int):
        super().__init__()
        self.patch, self.grid = patch, grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, grid * grid + 1, dim))
        self.patch_embeddings = nn.Module()
        self.patch_embeddings.projection = nn.Conv2d(channels, dim, patch, stride=patch)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        expected = self.grid * self.patch
        if tuple(pixels.shape[-2:]) != (expected, expected):
            raise ValueError(f"input must be {expected}x{expected}, got {tuple(pixels.shape[-2:])}")
        tokens = self.patch_embeddings.projection(pixels).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        return torch.cat((cls, tokens), dim=1) + self.position_embeddings


class PlainDinov2(nn.Module):
    """``forward(pixels) -> CLS`` after the final LayerNorm (= hub ``x_norm_clstoken``)."""

    def __init__(self, dim: int, depth: int, heads: int, patch: int, image_size: int,
                 mlp_ratio: float = 4.0, eps: float = 1e-6, channels: int = 3):
        super().__init__()
        self.embeddings = _Embeddings(dim, patch, image_size // patch, channels)
        self.encoder = nn.Module()
        hidden = int(dim * mlp_ratio)
        self.encoder.layer = nn.ModuleList(_Layer(dim, heads, hidden, eps) for _ in range(depth))
        self.layernorm = nn.LayerNorm(dim, eps=eps)

    @classmethod
    def from_config(cls, config: dict) -> "PlainDinov2":
        if config.get("use_swiglu_ffn") or config.get("num_register_tokens"):
            raise ValueError("only the MLP, register-free DINOv2 layout is supported")
        return cls(config["hidden_size"], config["num_hidden_layers"],
                   config["num_attention_heads"], config["patch_size"], config["image_size"],
                   config.get("mlp_ratio", 4), config.get("layer_norm_eps", 1e-6),
                   config.get("num_channels", 3))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixels)
        for layer in self.encoder.layer:
            x = layer(x)
        return self.layernorm(x)[:, 0]

    def forward_cls4(self, pixels: torch.Tensor) -> torch.Tensor:
        """Concatenate normalized CLS tokens from the final four blocks."""
        x = self.embeddings(pixels)
        depth = len(self.encoder.layer)
        if depth < 4:
            raise ValueError("CLS4 requires at least four encoder layers")
        features = []
        for index, layer in enumerate(self.encoder.layer):
            x = layer(x)
            if index >= depth - 4:
                features.append(self.layernorm(x)[:, 0])
        return torch.cat(features, dim=-1)


class Classifier(nn.Module):
    """Backbone CLS -> fused linear head -> softmax."""

    def __init__(self, backbone: nn.Module, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("weight", weight.float())
        self.register_buffer("bias", bias.float())

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        # No .float() here: an explicit Cast breaks the fp16 ONNX copy; the backbone is fp32.
        logits = self.backbone(pixels) @ self.weight.T + self.bias
        return torch.softmax(logits, dim=-1)


def load_backbone(hf_dir) -> PlainDinov2:
    from safetensors.torch import load_file  # noqa: PLC0415

    hf_dir = Path(hf_dir)
    config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    module = PlainDinov2.from_config(config)
    module.load_state_dict(load_file(str(hf_dir / "model.safetensors")), strict=True)
    return module.eval()


def load_classifier(hf_dir) -> tuple[Classifier, list[str]]:
    """``(model, classes)`` from a release ``hf/`` directory."""
    from safetensors import safe_open  # noqa: PLC0415

    head_path = str(Path(hf_dir) / "head.safetensors")
    with safe_open(head_path, framework="pt") as head:
        weight, bias = head.get_tensor("weight"), head.get_tensor("bias")
        classes = json.loads(head.metadata()["classes"])
    return Classifier(load_backbone(hf_dir), weight, bias).eval(), classes
