"""Configurable MoCo-v3-style SSL arm (v4) on the public DINOv2 ViT-B/14 backbone, for ablations.

With the default base_config it follows moco_v3_20261029 step for step (same RNG consumption, augmentations,
optimizer and EMA), so the same seed reproduces the v3 backbone_sha256. v3 files are not modified.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hashlib  # noqa: E402
import json  # noqa: E402
from collections import Counter  # noqa: E402
from importlib import metadata  # noqa: E402
from pathlib import Path  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import moco_v4_20261107_helpers as helpers  # noqa: E402
from moco_v3_20261029_runtime import (  # noqa: E402
    PUBLIC_SHA, _cls, _load_backbone, _max_memory_gib, _open_rgb,
)
from moco_v4_20261107_helpers import (  # noqa: E402
    apply_freeze_plan, build_heads, build_teacher, check_manifest_rows, cosine_anchor_loss, ema_update,
    l2sp_penalty, make_batch_sampler, make_view_transform, moco_symmetric_loss, snapshot_trainable,
    validate_config,
)

VERSION = "moco_v4_20261107"
LOG_EVERY = 20
FINAL_CHECKPOINT = "simsiam_dinov2_backbone_final.pt"
AUGMENTATIONS = {
    "random_resized_crop": {"size": 224, "scale": [0.4, 1.0]}, "horizontal_flip": True,
    "normalize": "imagenet", "color_constancy": False, "hue_jitter": False,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def _package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _trainable_summary(student):
    blocks = [i for i, b in enumerate(student.blocks) if any(p.requires_grad for p in b.parameters())]
    return {"trainable_blocks": blocks, "final_norm_trainable": student.norm.weight.requires_grad,
            "trainable_parameters": sum(p.numel() for p in student.parameters() if p.requires_grad)}


def resolved_params(config, options, student, device):
    """Every effective setting, defaults included, as written to resolved-params.json."""
    return {
        "runtime_version": VERSION, "runtime_sha256": digest(__file__),
        "helpers_sha256": digest(helpers.__file__), "torch": torch.__version__,
        "torchvision": _package_version("torchvision"), "device": device.type,
        "seed": options["seed"], "deterministic": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")},
        "manifest": config["manifest"], "manifest_sha256": config["manifest_sha256"],
        "batch_size": config["batch_size"], "max_steps": options["max_steps"],
        "init_weights": options["initial_weights"], "init_weights_sha256": PUBLIC_SHA,
        "freeze_blocks": options["freeze_blocks"], **_trainable_summary(student),
        "optimizer": "AdamW", "backbone_lr": options["backbone_lr"], "head_lr": options["head_lr"],
        "weight_decay": options["weight_decay"], "lr_schedule": "constant", "scheduler": None, "warmup": None,
        "precision": "bf16 autocast on cuda, fp32 on cpu", "augmentations": AUGMENTATIONS,
        "temperature": options["temperature"], "teacher_momentum": options["teacher_momentum"],
        "cosine_anchor": options["cosine_anchor"], "l2sp": options["l2sp"],
        "clinical_quota": options["clinical_quota"], "checkpoint_steps": options["checkpoint_steps"],
    }


def _build_batch(sampler, transform, device):
    batch_rows = sampler()
    return [torch.stack([transform(_open_rgb(row)) for row in batch_rows]).to(device) for _ in range(2)]


def _training_step(models, optimizer, views, options, l2sp_reference):
    student, teacher, proj, pred, t_proj, anchor_model = models
    with torch.autocast(device_type=views[0].device.type, dtype=torch.bfloat16,
                        enabled=views[0].device.type == "cuda"):
        with torch.no_grad():
            k1 = t_proj(_cls(teacher, views[0]))
            k2 = t_proj(_cls(teacher, views[1]))
        c1, c2 = _cls(student, views[0]), _cls(student, views[1])
        q1, q2 = pred(proj(c1)), pred(proj(c2))
        loss = moco_symmetric_loss(q1.float(), q2.float(), k1.float(), k2.float(), options["temperature"])
        if anchor_model is not None:
            with torch.no_grad():
                anchor_cls = _cls(anchor_model, views[0]).float()
            loss = loss + cosine_anchor_loss(c1.float(), anchor_cls)
    if options["l2sp"] > 0:
        loss = loss + l2sp_penalty(student, l2sp_reference, options["l2sp"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss


def _ema_step(student, teacher, proj, t_proj, momentum):
    ema_update(teacher, student, momentum)
    ema_update(t_proj, proj, momentum)


def _deviations(options, batch_size, anchor_on):
    quota = options["clinical_quota"]
    return [
        "MoCo-v3-style adaptation, not a reproduction of the official MoCo v3 recipe.",
        "Negatives are in-batch only (no memory queue, which would be MoCo v1/v2).",
        f"Pretrained DINOv2 ViT-B/14 backbone with blocks 0-{options['freeze_blocks'] - 1} and patch/pos/cls frozen.",
        f"Fixed teacher momentum {options['teacher_momentum']} is a local adaptation, not a published schedule.",
        f"Temperature {options['temperature']}; constant lr, no scheduler or warmup.",
        f"Batch size {batch_size} is a local memory choice; no official recipe batch is claimed.",
        f"{options['max_steps']}-step budget is a local audit choice.",
        f"Cosine anchor to public DINOv2 CLS features: {'enabled (weight 1)' if anchor_on else 'disabled'}.",
        f"L2-SP strength {options['l2sp']} toward the public initial weights.",
        f"Batch composition: {'uniform over manifest' if quota is None else f'{batch_size - quota} source_fit + {quota} pad_adapt'}.",
    ]


def _build_models(options, device):
    student = _load_backbone(options["initial_weights"])
    apply_freeze_plan(student, options["freeze_blocks"])
    student = student.to(device)
    proj, pred = (m.to(device) for m in build_heads())
    teacher, t_proj = build_teacher(student), build_teacher(proj)
    anchor = None
    if options["cosine_anchor"]:
        anchor = _load_backbone(options["initial_weights"]).to(device).eval().requires_grad_(False)
    return student, teacher, proj, pred, t_proj, anchor


def _build_optimizer(student, proj, pred, options):
    return torch.optim.AdamW(
        [{"params": [p for p in student.parameters() if p.requires_grad], "lr": options["backbone_lr"]},
         {"params": list(proj.parameters()) + list(pred.parameters()), "lr": options["head_lr"]}],
        weight_decay=options["weight_decay"])


def _train(models, optimizer, options, sampler, device, l2sp_reference):
    student, teacher, proj, _, t_proj, _ = models
    transform, checkpoints = make_view_transform(), set(options["checkpoint_steps"])
    ckpt_dir = Path(options["output_dir"]) / "checkpoints"
    for step in range(1, options["max_steps"] + 1):
        views = _build_batch(sampler, transform, device)
        loss = _training_step(models, optimizer, views, options, l2sp_reference)
        _ema_step(student, teacher, proj, t_proj, options["teacher_momentum"])
        if step % LOG_EVERY == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach()),
                              "max_memory_gib": _max_memory_gib(device)}), flush=True)
        if step in checkpoints:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(student.state_dict(), ckpt_dir / f"backbone_step{step}.pt")


def run(config_path):
    config = json.loads(Path(config_path).read_text())
    options, batch_size = validate_config(config)
    rows = _read_manifest(config)
    sampler = make_batch_sampler(rows, batch_size, options["clinical_quota"])
    _configure_seed(options["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("MoCo-v3-style runtime requires CUDA BF16 autocast support")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(options["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    models = _build_models(options, device)
    student, _, proj, pred, _, anchor = models
    reference = snapshot_trainable(student) if options["l2sp"] > 0 else None
    params = resolved_params(config, options, student, device)
    (output / "resolved-params.json").write_text(json.dumps(params, indent=2) + "\n")
    started = time.perf_counter()
    _train(models, _build_optimizer(student, proj, pred, options), options, sampler, device, reference)
    checkpoint = output / FINAL_CHECKPOINT
    torch.save(student.state_dict(), checkpoint)
    wall_hours = (time.perf_counter() - started) / 3600
    result = {
        "status": "COMPLETE", "steps": options["max_steps"], "backbone_sha256": digest(checkpoint),
        "allocated_gpu_hours": wall_hours if device.type == "cuda" else 0.0, "wall_hours": wall_hours,
        "max_memory_gib": _max_memory_gib(device), "public_weights_sha256": PUBLIC_SHA,
        "manifest_sha256": config["manifest_sha256"],
        "role_counts": dict(Counter(row["role"] for row in rows)), "seed": options["seed"],
        "batch_size": batch_size, "momentum": options["teacher_momentum"],
        "recipe_deviations": _deviations(options, batch_size, anchor is not None),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python moco_v4_20261107_runtime.py CONFIG.json")
    run(sys.argv[1])
