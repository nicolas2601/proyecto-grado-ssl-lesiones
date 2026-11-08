"""MoCo-v3-style SSL arm on the public DINOv2 ViT-B/14 backbone.

Momentum key encoder (EMA teacher), symmetric InfoNCE with in-batch negatives (no memory queue),
blocks 0-7 and patch/pos/cls frozen. Hyperparameters are local adaptations, not an official MoCo v3 recipe.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

from moco_v3_20261029_helpers import (  # noqa: E402
    ALLOWED_ROLES,
    apply_freeze_plan,
    build_heads,
    build_teacher,
    check_manifest_rows,
    cosine_anchor_loss,
    ema_update,
    make_view_transform,
    moco_symmetric_loss,
)

PUBLIC_SHA = "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73"
ALLOWED_SEEDS = {42, 43, 44}
FREEZE_BLOCKS = 8
MOMENTUM = 0.99
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 0.05
LOG_EVERY = 20
EXPECTED_BLOCKS = 12
EXPECTED_WIDTH = 768


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    required = {"manifest", "manifest_sha256", "manifest_roles", "batch_size", "base_config"}
    if not required <= config.keys():
        raise ValueError(f"missing config fields: {sorted(required - config.keys())}")
    base = config["base_config"]
    base_required = {"initial_weights", "output_dir", "max_steps", "seed"}
    if not isinstance(base, dict) or not base_required <= base.keys():
        raise ValueError("base_config requires initial_weights, output_dir, max_steps, and seed")
    if base["seed"] not in ALLOWED_SEEDS:
        raise ValueError("seed must be one of 42, 43, or 44")
    if not isinstance(base["max_steps"], int) or base["max_steps"] <= 0:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(config["batch_size"], int) or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")
    if set(config["manifest_roles"]) != ALLOWED_ROLES:
        raise ValueError("manifest_roles must declare source_fit and pad_adapt only")
    if base.get("teacher_momentum", MOMENTUM) != MOMENTUM:
        raise ValueError("teacher momentum is fixed at 0.99 for this arm")
    if not isinstance(base.get("cosine_anchor", False), bool):
        raise ValueError("cosine_anchor must be a boolean")
    sha = config["manifest_sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    return base, config["batch_size"]


def _read_manifest(config):
    manifest = Path(config["manifest"])
    if digest(manifest) != config["manifest_sha256"]:
        raise ValueError("Manifest digest mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    check_manifest_rows(rows)
    return rows


def _configure_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _load_backbone(initial_weights):
    if digest(initial_weights) != PUBLIC_SHA:
        raise ValueError("Public DINOv2 initializer SHA does not match PUBLIC_SHA")
    hub = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
    if not (hub / "dinov2/models/vision_transformer.py").is_file():
        raise FileNotFoundError(f"local DINOv2 torch.hub source not found: {hub}")
    backbone = torch.hub.load(str(hub), "dinov2_vitb14", source="local", pretrained=False)
    backbone.load_state_dict(torch.load(initial_weights, map_location="cpu", weights_only=True), strict=True)
    if len(backbone.blocks) != EXPECTED_BLOCKS or backbone.pos_embed.shape[-1] != EXPECTED_WIDTH:
        raise ValueError("expected DINOv2 ViT-B/14 with 12 blocks and 768-dim embeddings")
    return backbone


def _cls(backbone, images):
    return backbone.forward_features(images)["x_norm_clstoken"]


def _max_memory_gib(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (2**30 if sys.platform == "darwin" else 1024**2)


def _open_rgb(row):
    with Image.open(row["image_path"]) as source:
        return source.convert("RGB")


def _build_batch(rows, batch_size, transform, device):
    batch_rows = random.sample(rows, batch_size)
    return [torch.stack([transform(_open_rgb(row)) for row in batch_rows]).to(device) for _ in range(2)]


def _training_step(models, optimizer, views):
    student, teacher, proj, pred, t_proj, anchor_model = models
    with torch.autocast(device_type=views[0].device.type, dtype=torch.bfloat16,
                        enabled=views[0].device.type == "cuda"):
        with torch.no_grad():
            k1 = t_proj(_cls(teacher, views[0]))
            k2 = t_proj(_cls(teacher, views[1]))
        c1, c2 = _cls(student, views[0]), _cls(student, views[1])
        q1, q2 = pred(proj(c1)), pred(proj(c2))
        loss = moco_symmetric_loss(q1.float(), q2.float(), k1.float(), k2.float())
        if anchor_model is not None:
            with torch.no_grad():
                anchor_cls = _cls(anchor_model, views[0]).float()
            loss = loss + cosine_anchor_loss(c1.float(), anchor_cls)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss


def _ema_step(student, teacher, proj, t_proj):
    ema_update(teacher, student, MOMENTUM)
    ema_update(t_proj, proj, MOMENTUM)


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    base, batch_size = validate_config(config)
    rows = _read_manifest(config)
    _configure_seed(base["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("MoCo-v3-style runtime requires CUDA BF16 autocast support")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    output = Path(base["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    student = _load_backbone(base["initial_weights"])
    apply_freeze_plan(student, FREEZE_BLOCKS)
    student = student.to(device)
    proj, pred = (m.to(device) for m in build_heads())
    teacher, t_proj = build_teacher(student), build_teacher(proj)
    anchor_model = None
    if base.get("cosine_anchor", False):
        anchor_model = _load_backbone(base["initial_weights"]).to(device).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in student.parameters() if p.requires_grad], "lr": BACKBONE_LR},
         {"params": list(proj.parameters()) + list(pred.parameters()), "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )
    transform = make_view_transform()
    models = (student, teacher, proj, pred, t_proj, anchor_model)
    started = time.perf_counter()

    for step in range(1, base["max_steps"] + 1):
        views = _build_batch(rows, batch_size, transform, device)
        loss = _training_step(models, optimizer, views)
        _ema_step(student, teacher, proj, t_proj)
        if step % LOG_EVERY == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "max_memory_gib": _max_memory_gib(device)}), flush=True)

    checkpoint = output / "simsiam_dinov2_backbone_final.pt"
    torch.save(student.state_dict(), checkpoint)
    wall_hours = (time.perf_counter() - started) / 3600
    result = {
        "status": "COMPLETE",
        "steps": base["max_steps"],
        "backbone_sha256": digest(checkpoint),
        "allocated_gpu_hours": wall_hours if device.type == "cuda" else 0.0,
        "wall_hours": wall_hours,
        "max_memory_gib": _max_memory_gib(device),
        "public_weights_sha256": PUBLIC_SHA,
        "manifest_sha256": config["manifest_sha256"],
        "role_counts": dict(Counter(row["role"] for row in rows)),
        "seed": base["seed"],
        "batch_size": batch_size,
        "momentum": MOMENTUM,
        "recipe_deviations": [
            "MoCo-v3-style adaptation, not a reproduction of the official MoCo v3 recipe.",
            "Negatives are in-batch only (no memory queue, which would be MoCo v1/v2).",
            "Pretrained DINOv2 ViT-B/14 backbone with blocks 0-7 and patch/pos/cls frozen.",
            "Fixed teacher momentum 0.99 is a local adaptation, not a published schedule.",
            "Batch size 16 is a local memory choice; no official recipe batch is claimed.",
            f"{base['max_steps']}-step budget is a local audit choice.",
            f"Cosine anchor to public DINOv2 CLS features: {'enabled (weight 1)' if anchor_model else 'disabled'}.",
        ],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python moco_v3_20261029_runtime.py CONFIG.json")
    run(sys.argv[1])
