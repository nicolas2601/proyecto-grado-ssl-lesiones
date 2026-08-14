"""The v6 training loop: budget-driven, DDP-ready, MLflow-tracked.

Deliberately thin: every decision with logic in it (config validation, budget
math, schedules, checkpoint tags, tracking) lives in the tested pure modules.
This file only wires torch to those decisions.

Invariants enforced at runtime, loudly:
* zero labels: ``config.assert_label_free`` runs before anything else, and the
  datasets structurally cannot yield targets;
* checkpoints happen at fixed COMPUTE fractions (equal-compute curves);
* only rank 0 talks to MLflow;
* a crash-resume never repeats compute: the elapsed wall time is part of the
  checkpoint state.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dermssl.pretrain import budget as budget_mod
from dermssl.pretrain import config as config_mod
from dermssl.pretrain import schedules, torch_data, torch_dino, torch_simsiam, tracking

LOG_EVERY_STEPS = 50


def ddp_env() -> tuple[int, int, int]:
    """(rank, world_size, local_rank) from torchrun env; single-GPU defaults."""
    return (
        int(os.environ.get("RANK", 0)),
        int(os.environ.get("WORLD_SIZE", 1)),
        int(os.environ.get("LOCAL_RANK", 0)),
    )


def _autocast(precision: str, device: torch.device):
    if precision == "fp32" or device.type != "cuda":
        return torch.autocast(device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _build(cfg: config_mod.PretrainConfig, device: torch.device):
    """(model, loss_fn, optimizer, dataset, collate) for the configured method."""
    paths = config_mod.collect_images(cfg.image_root)
    if cfg.method == "simsiam":
        model = torch_simsiam.SimSiam().to(device)
        dataset = torch_data.SimSiamPairDataset(
            paths, cfg.resolution, cfg.color_constancy_probs, cfg.seed
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.0, momentum=0.9, weight_decay=cfg.weight_decay
        )
        return model, None, optimizer, dataset, None

    model = torch_dino.DinoContinual(cfg.backbone).to(device)
    loss_fn = torch_dino.DinoLoss().to(device)
    dataset = torch_data.DinoMultiCropDataset(
        paths,
        cfg.resolution,
        cfg.local_resolution,
        cfg.n_local_crops,
        cfg.color_constancy_probs,
        cfg.lesion_guided_crops,
        cfg.seed,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.student_parameters() if p.requires_grad),
        lr=0.0,
        weight_decay=cfg.weight_decay,
    )
    return model, loss_fn, optimizer, dataset, torch_data.dino_collate


def _save_checkpoint(
    path: Path, model, optimizer, elapsed: float, saved_fractions: set[float]
) -> None:
    raw = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "elapsed_wall_seconds": elapsed,
            "saved_fractions": sorted(saved_fractions),
        },
        path,
    )


def _export_backbone(path: Path, model, method: str) -> None:
    """Backbone-only weights in the shape the v5 evaluation loads."""
    raw = model.module if isinstance(model, DDP) else model
    backbone = raw.backbone if method == "simsiam" else raw.student_backbone
    torch.save(backbone.state_dict(), path)


def train(cfg: config_mod.PretrainConfig, mlflow_module=None) -> Path:
    cfg = config_mod.validate(cfg)
    config_mod.assert_label_free(cfg)

    rank, world_size, local_rank = ddp_env()
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )
    torch.manual_seed(cfg.seed + rank)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, loss_fn, optimizer, dataset, collate = _build(cfg, device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)
    sampler = (
        DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_per_gpu,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
        collate_fn=collate,
    )

    compute = budget_mod.ComputeBudget(
        cfg.budget_gpu_hours, world_size, cfg.checkpoint_fractions
    )
    base_lr = config_mod.effective_lr(cfg, world_size)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")

    tracker = tracking.build_tracker(
        mlflow_module,
        experiment=cfg.mlflow_experiment,
        tracking_uri=cfg.tracking_uri,
        run=tracking.run_name(cfg.method, cfg.backbone, cfg.seed),
        rank=rank,
    )
    if rank == 0 and isinstance(tracker, tracking.NullTracker):
        print("WARNING: mlflow unavailable - this run is NOT being tracked.")
    params = config_mod.mlflow_params(cfg)
    params["world_size"] = world_size
    params["effective_lr"] = base_lr
    tracker.start(params)

    elapsed_offset, saved_fractions = 0.0, set()
    resume_path = out_dir / "resume.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        raw = model.module if isinstance(model, DDP) else model
        raw.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        elapsed_offset = state["elapsed_wall_seconds"]
        saved_fractions = set(state["saved_fractions"])
        if rank == 0:
            print(f"resumed at {compute.gpu_hours_elapsed(elapsed_offset):.2f} GPU-hours")

    start = time.monotonic()
    elapsed = lambda: elapsed_offset + (time.monotonic() - start)  # noqa: E731
    step, images_seen, epoch = 0, 0, 0

    while not compute.should_stop(elapsed()):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            fraction = compute.fraction_done(elapsed())
            lr = schedules.lr_at(fraction, base_lr, cfg.warmup_fraction)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg.precision, device):
                if cfg.method == "simsiam":
                    v1, v2 = batch[0].to(device, non_blocking=True), batch[1].to(
                        device, non_blocking=True
                    )
                    p1, p2, z1, z2 = model(v1, v2)
                    loss = torch_simsiam.simsiam_loss(p1, p2, z1, z2)
                    health = torch_simsiam.collapse_indicator(z1)
                    batch_size, guided_rate = v1.shape[0], -1.0
                else:
                    crops, guided = batch
                    crops = [c.to(device, non_blocking=True) for c in crops]
                    raw = model.module if isinstance(model, DDP) else model
                    student_out = raw.student_forward(crops)
                    teacher_out = raw.teacher_forward(crops[:2])
                    loss = loss_fn(student_out, teacher_out)
                    health = float(teacher_out[0].std(dim=0).mean().item())
                    batch_size, guided_rate = crops[0].shape[0], float(guided.mean())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 3.0
            )
            scaler.step(optimizer)
            scaler.update()

            if cfg.method == "dino_continual":
                raw = model.module if isinstance(model, DDP) else model
                raw.ema_update(schedules.ema_momentum_at(fraction, cfg.momentum_teacher))

            step += 1
            images_seen += batch_size * world_size
            if rank == 0 and step % LOG_EVERY_STEPS == 0:
                gpu_hours = compute.gpu_hours_elapsed(elapsed())
                tracker.log_metric("ssl_loss", float(loss.item()), step=step)
                tracker.log_metric("lr", lr, step=step)
                tracker.log_metric("embedding_std", health, step=step)
                tracker.log_metric("gpu_hours", gpu_hours, step=step)
                tracker.log_metric("images_per_gpu_hour", images_seen / max(gpu_hours, 1e-9), step=step)
                if guided_rate >= 0:
                    tracker.log_metric("lesion_guided_rate", guided_rate, step=step)

            if rank == 0:
                for fraction_due in compute.due_checkpoints(elapsed(), saved_fractions):
                    saved_fractions.add(fraction_due)
                    tag = budget_mod.checkpoint_tag(fraction_due)
                    ckpt = out_dir / f"{cfg.method}_{tag}.pt"
                    _export_backbone(ckpt, model, cfg.method)
                    tracker.log_artifact(str(ckpt))
                    _save_checkpoint(resume_path, model, optimizer, elapsed(), saved_fractions)
                    print(f"[{tag}] saved at {compute.gpu_hours_elapsed(elapsed()):.2f} GPU-hours")

            if compute.should_stop(elapsed()):
                break
        epoch += 1

    final_path = out_dir / f"{cfg.method}_backbone_final.pt"
    if rank == 0:
        _export_backbone(final_path, model, cfg.method)
        tracker.log_artifact(str(final_path))
        tracker.log_metric("total_gpu_hours", compute.gpu_hours_elapsed(elapsed()))
        tracker.log_metric("total_images_seen", float(images_seen))
        tracker.end()
    if world_size > 1:
        dist.destroy_process_group()
    return final_path
