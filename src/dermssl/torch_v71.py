"""The torch half of v7.1 (docs/29 §2.1, §2.2, §2.7): weights, views, features.

* WiSE-FT: ``(1 - alpha) * public + alpha * adapted``, written in that order so
  alpha = 0 and alpha = 1 return the endpoint tensors bit for bit (the
  pre-test of §2.1 checks it on the holdout before anything else runs).
* Seed soup: the mean of two finals of the same arm.
* Features: one forward gives the five parts every §2.2 / §2.7 feature is
  built from: the CLS (identical to the bank's ``model(x)``), the mean and the
  GeM (p = 3) of the final-norm patch tokens, the patch tokens pooled by the
  last block's CLS->patch attention averaged over heads, and the normed CLS of
  the last four blocks concatenated.
* D4: the eight rotations/flips of the already centre-cropped square input,
  which equals transforming the image before the centred square crop.

Images travel as uint8 after the bank's resize + centre crop; normalisation is
the same float32 ops as ``ToTensor`` + ``Normalize``, so the CLS path equals
``cluster_frozen_bank.eval_transform`` exactly.
"""

from __future__ import annotations

import contextlib

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_FACTOR = 1.14  # the bank's int(resolution * 1.14) shorter side
GEM_P = 3.0  # §2.7 item 3: fixed, not tuned
GEM_EPS = 1e-6
N_LAST = 4
PARTS = ("cls", "mean", "gem", "attn", "cls4")
N_D4 = 8


def _check_keys(a: dict, b: dict) -> None:
    if set(a) != set(b):
        raise ValueError(f"claves distintas entre pesos: {sorted(set(a) ^ set(b))[:5]}")
    for key in a:
        if a[key].shape != b[key].shape:
            raise ValueError(f"{key}: forma {tuple(a[key].shape)} contra {tuple(b[key].shape)}")


def interpolate_state(public: dict, adapted: dict, alpha: float) -> dict:
    """WiSE-FT. Non-float tensors are copied and must agree between the two."""
    _check_keys(public, adapted)
    out = {}
    for key, pub in public.items():
        ada = adapted[key]
        if not pub.is_floating_point():
            if not torch.equal(pub, ada):
                raise ValueError(f"{key}: tensor no flotante distinto entre publico y adaptado")
            out[key] = pub.clone()
            continue
        out[key] = (1.0 - alpha) * pub + alpha * ada.to(pub.dtype)
    return out


def average_states(a: dict, b: dict) -> dict:
    """Seed soup of two finals: the midpoint, via the same interpolation."""
    return interpolate_state(a, b, 0.5)


def max_abs_diff(a: dict, b: dict) -> float:
    _check_keys(a, b)
    return max(float((a[k].double() - b[k].double()).abs().max()) if a[k].numel() else 0.0 for k in a)


def resize_crop_uint8(image, resolution: int) -> np.ndarray:
    """The bank's geometry (bilinear resize of the shorter side, centre crop), as HWC uint8."""
    from torchvision import transforms  # noqa: PLC0415

    geometry = transforms.Compose([
        transforms.Resize(int(resolution * RESIZE_FACTOR),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
    ])
    return np.asarray(geometry(image), dtype=np.uint8)


def normalize_uint8(batch: np.ndarray) -> torch.Tensor:
    """NHWC uint8 -> NCHW normalised float32, the ToTensor + Normalize arithmetic."""
    x = torch.from_numpy(np.array(batch, dtype=np.uint8, copy=True)).permute(0, 3, 1, 2).contiguous()
    x = x.to(torch.float32).div(255)
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=torch.float32)[None, :, None, None]
    std = torch.as_tensor(IMAGENET_STD, dtype=torch.float32)[None, :, None, None]
    return x.sub(mean).div(std)


def d4_views(x: torch.Tensor) -> list[torch.Tensor]:
    """Identity first, then the other seven elements of the dihedral group."""
    return [torch.rot90(x.flip(-1) if flip else x, k, dims=(-2, -1))
            for flip in (False, True) for k in range(4)]


def gem(tokens: torch.Tensor, p: float = GEM_P, eps: float = GEM_EPS) -> torch.Tensor:
    """Generalised mean over tokens (dim 1); values clamped at ``eps`` as in Radenovic et al."""
    return tokens.clamp(min=eps).pow(p).mean(dim=1).pow(1.0 / p)


def last_block(model):
    block = model.blocks[-1]
    return block if hasattr(block, "attn") else block[-1]  # chunked models nest blocks


def _cls_attention(attn, x: torch.Tensor, first_patch: int) -> torch.Tensor:
    """Head-averaged softmax row of the CLS query over the patch tokens, renormalised."""
    batch, n_tokens, dim = x.shape
    heads = attn.num_heads
    with torch.autocast(device_type=x.device.type, enabled=False):  # fp32 even inside the bank's autocast
        qkv = attn.qkv(x.float()).reshape(batch, n_tokens, 3, heads, dim // heads)
        q_cls, keys = qkv[:, 0, 0] * attn.scale, qkv[:, :, 1]
        weights = torch.einsum("bhd,bnhd->bhn", q_cls, keys).softmax(dim=-1).mean(dim=1)
        patches = weights[:, first_patch:]
        return patches / patches.sum(dim=1, keepdim=True)


@contextlib.contextmanager
def _capture_attention_input(model):
    captured = {}
    handle = last_block(model).attn.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("x", inputs[0]))
    try:
        yield captured
    finally:
        handle.remove()


def extract_parts(model, x: torch.Tensor) -> dict:
    """The five feature parts of one normalised batch, float32, on ``x``'s device."""
    first_patch = 1 + int(getattr(model, "num_register_tokens", 0))
    with torch.no_grad(), _capture_attention_input(model) as captured:
        layers = model.get_intermediate_layers(x, n=N_LAST, return_class_token=True, norm=True)
        patches, cls = layers[-1][0].float(), layers[-1][1].float()
        weights = _cls_attention(last_block(model).attn, captured["x"], first_patch)
        return {
            "cls": cls,
            "mean": patches.mean(dim=1),
            "gem": gem(patches),
            "attn": torch.einsum("bp,bpc->bc", weights, patches),
            "cls4": torch.cat([layer[1].float() for layer in layers], dim=1),
        }


def _autocast(device: torch.device):
    return torch.amp.autocast("cuda") if device.type == "cuda" else contextlib.nullcontext()


def extract_split(model, images: np.ndarray, *, device: torch.device, batch_size: int,
                  n_views: int = 1) -> dict:
    """``part -> (n_views, n_images, dim)`` over NHWC uint8 ``images``; autocast on cuda like the bank."""
    if n_views not in (1, N_D4):
        raise ValueError(f"n_views={n_views}: solo identidad (1) o el grupo D4 completo ({N_D4})")
    model.eval().to(device)
    chunks: dict = {part: [[] for _ in range(n_views)] for part in PARTS}
    for start in range(0, len(images), batch_size):
        x = normalize_uint8(images[start:start + batch_size]).to(device)
        for view, xv in enumerate(d4_views(x)[:n_views]):
            with _autocast(device):
                parts = extract_parts(model, xv)
            for part, value in parts.items():
                chunks[part][view].append(value.float().cpu().numpy())
    return {part: np.stack([np.concatenate(v) for v in views]) for part, views in chunks.items()}
