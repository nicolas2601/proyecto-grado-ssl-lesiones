"""Hub DINOv2 (``torch.hub`` ``dinov2_vitb14``) -> Hugging Face ``Dinov2Model`` layout.

Two decisions a reviewer should check:

1. ``attn.qkv`` is one Linear with rows ``[q; k; v]``; HF keeps three.
   ``chunk(3)`` splits it in that order, which is how the hub reshapes it
   (``reshape(B, N, 3, heads, dim)``).
2. The hub was trained with a 37x37 position grid and interpolates it to
   16x16 at 224 with ``scale_factor=(16.1/37)`` and no antialias; HF
   interpolates with ``size=16``. Those differ. The release therefore stores
   the grid the hub itself computes at 224 (``interpolate_pos_encoding``) and
   sets ``image_size=224``, so HF and the plain module take it as-is and the
   CLS token is the evaluated one. Other resolutions are not supported.
"""

from __future__ import annotations

import json
from pathlib import Path

RESOLUTION = 224
_BLOCK_MAP = {
    "norm1.weight": "norm1.weight", "norm1.bias": "norm1.bias",
    "attn.proj.weight": "attention.output.dense.weight",
    "attn.proj.bias": "attention.output.dense.bias",
    "ls1.gamma": "layer_scale1.lambda1", "ls2.gamma": "layer_scale2.lambda1",
    "norm2.weight": "norm2.weight", "norm2.bias": "norm2.bias",
    "mlp.fc1.weight": "mlp.fc1.weight", "mlp.fc1.bias": "mlp.fc1.bias",
    "mlp.fc2.weight": "mlp.fc2.weight", "mlp.fc2.bias": "mlp.fc2.bias",
}
_TOP_MAP = {
    "cls_token": "embeddings.cls_token", "mask_token": "embeddings.mask_token",
    "patch_embed.proj.weight": "embeddings.patch_embeddings.projection.weight",
    "patch_embed.proj.bias": "embeddings.patch_embeddings.projection.bias",
    "norm.weight": "layernorm.weight", "norm.bias": "layernorm.bias",
}


def _check_supported(model) -> None:
    if getattr(model, "num_register_tokens", 0):
        raise ValueError("register tokens are not part of Dinov2Model")
    if getattr(model, "chunked_blocks", False):
        raise ValueError("chunked blocks: build the hub model with block_chunks=0")
    if not hasattr(model.blocks[0].mlp, "fc1"):
        raise ValueError("only the MLP feed-forward maps to Dinov2Model (no SwiGLU)")


def hf_config(model, resolution: int = RESOLUTION) -> dict:
    """``config.json`` content for a hub ``DinoVisionTransformer``."""
    _check_supported(model)
    fc1 = model.blocks[0].mlp.fc1
    return {
        "architectures": ["Dinov2Model"], "model_type": "dinov2",
        "hidden_size": int(model.embed_dim), "num_hidden_layers": len(model.blocks),
        "num_attention_heads": int(model.num_heads), "patch_size": int(model.patch_size),
        "image_size": resolution, "num_channels": 3,
        "mlp_ratio": fc1.out_features // fc1.in_features, "hidden_act": "gelu",
        "layer_norm_eps": float(model.norm.eps), "qkv_bias": model.blocks[0].attn.qkv.bias is not None,
        "layerscale_value": 1.0, "use_swiglu_ffn": False, "drop_path_rate": 0.0,
        "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02, "apply_layernorm": True, "reshape_hidden_states": True,
        "use_mask_token": True, "torch_dtype": "float32",
    }


def position_grid(model, resolution: int = RESOLUTION):
    """The position embeddings the hub adds at ``resolution``, computed by the hub."""
    import torch  # noqa: PLC0415

    tokens = (resolution // model.patch_size) ** 2 + 1
    with torch.no_grad():
        dummy = torch.zeros(1, tokens, model.embed_dim, dtype=model.pos_embed.dtype)
        return model.interpolate_pos_encoding(dummy, resolution, resolution).detach().clone()


def hub_to_hf_state(model, resolution: int = RESOLUTION) -> dict:
    """Every hub tensor under its ``Dinov2Model`` name; unknown keys are an error."""
    _check_supported(model)
    state, out = model.state_dict(), {}
    for key, tensor in state.items():
        if key in _TOP_MAP:
            out[_TOP_MAP[key]] = tensor
        elif key == "pos_embed":
            out["embeddings.position_embeddings"] = position_grid(model, resolution)
        elif key.startswith("blocks."):
            _map_block_key(key, tensor, out)
        else:
            raise ValueError(f"no Dinov2Model counterpart for hub key {key!r}")
    return {k: v.detach().cpu().contiguous().float() for k, v in out.items()}


def _map_block_key(key: str, tensor, out: dict) -> None:
    _, index, rest = key.split(".", 2)
    prefix = f"encoder.layer.{index}."
    if rest in ("attn.qkv.weight", "attn.qkv.bias"):
        kind = rest.rsplit(".", 1)[1]
        for name, part in zip(("query", "key", "value"), tensor.chunk(3, dim=0)):
            out[f"{prefix}attention.attention.{name}.{kind}"] = part
    elif rest in _BLOCK_MAP:
        out[prefix + _BLOCK_MAP[rest]] = tensor
    else:
        raise ValueError(f"no Dinov2Model counterpart for hub key {key!r}")


def save_hf_backbone(model, out_dir, resolution: int = RESOLUTION) -> dict:
    """Write ``config.json`` + ``model.safetensors`` loadable by ``AutoModel.from_pretrained``."""
    from safetensors.torch import save_file  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = hf_config(model, resolution)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_file(hub_to_hf_state(model, resolution), str(out_dir / "model.safetensors"),
              metadata={"format": "pt"})
    return config
