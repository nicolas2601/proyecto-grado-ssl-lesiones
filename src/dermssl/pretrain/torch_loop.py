"""Bucle de entrenamiento v6: gobernado por presupuesto, listo para DDP y MLflow.

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
from dermssl.pretrain import (
    run_contract,
    schedules,
    torch_data,
    torch_dino,
    torch_simsiam,
    tracking,
)

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


def _build(
    cfg: config_mod.PretrainConfig,
    device: torch.device,
    transformed_evidence: config_mod.TransformedManifestEvidence | None = None,
):
    """(model, loss_fn, optimizer, dataset, collate) for the configured method."""
    paths = config_mod.collect_images(cfg.image_root)
    if cfg.method in config_mod.SIMSIAM_METHODS:
        if cfg.adaptation_mode == "original_transformed_consistency":
            if transformed_evidence is None:
                config_mod.validate_transformed_manifest(cfg)
            paths = config_mod.collect_paired_images(
                cfg.image_root,
                cfg.transformed_image_root,
                manifest_path=cfg.transformed_manifest_path,
                source_revision=cfg.data_revision,
                transformed_revision=cfg.transformed_data_revision,
            )
        pretrained = cfg.method in ("simsiam_dinov2", "simsiam_imagenet")
        model = torch_simsiam.SimSiam(cfg.backbone, pretrained=pretrained).to(device)
        dataset = torch_data.SimSiamPairDataset(
            paths, cfg.resolution, cfg.color_constancy_probs, cfg.seed
        )
        if pretrained:
            # Converged ViT weights: AdamW, like the continual DINO arm.
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.0, weight_decay=cfg.weight_decay
            )
        else:
            # Chen & He 2021 from-scratch recipe: SGD with momentum.
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.0, momentum=0.9, weight_decay=cfg.weight_decay
            )
        return model, None, optimizer, dataset, None

    model = torch_dino.DinoContinual(
        cfg.backbone, pretrained=cfg.method == "dino_continual"
    ).to(device)
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
    backbone = (
        raw.backbone
        if method in config_mod.SIMSIAM_METHODS
        else raw.student_backbone
    )
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
    gpu_count = world_size if device.type == "cuda" else 0
    torch.manual_seed(cfg.seed + rank)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transformed_evidence = (
        config_mod.validate_transformed_manifest(cfg)
        if cfg.adaptation_mode == "original_transformed_consistency"
        else None
    )
    manifest = run_contract.build_run_manifest(
        cfg,
        gpu_count=gpu_count,
        code_revision=os.environ.get("GIT_COMMIT", ""),
        transformed_manifest=transformed_evidence,
    )
    if rank == 0:
        run_contract.write_run_manifest(manifest, out_dir / "run.json")

    model, loss_fn, optimizer, dataset, collate = _build(
        cfg, device, transformed_evidence
    )
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
        cfg.budget_gpu_hours, gpu_count, cfg.checkpoint_fractions
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
        print("ADVERTENCIA: MLflow no está disponible; esta ejecución no tendrá seguimiento.")
    params = config_mod.mlflow_params(cfg)
    params["world_size"] = world_size
    params["effective_lr"] = base_lr
    tracker.start(params)

    elapsed_offset, saved_fractions = 0.0, set()
    resume_path = Path(cfg.resume_checkpoint) if cfg.resume_checkpoint else out_dir / "resume.pt"
    if cfg.resume_checkpoint and not resume_path.is_file():
        raise FileNotFoundError(f"el checkpoint de reanudación no existe: {resume_path}")
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        raw = model.module if isinstance(model, DDP) else model
        raw.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        elapsed_offset = state["elapsed_wall_seconds"]
        saved_fractions = set(state["saved_fractions"])
        if rank == 0:
            if gpu_count:
                print(
                    f"Reanudado en {compute.gpu_hours_elapsed(elapsed_offset):.2f} horas-GPU"
                )
            else:
                print(f"Reanudado tras {elapsed_offset:.2f} segundos de reloj")

    start = time.monotonic()
    elapsed = lambda: elapsed_offset + (time.monotonic() - start)  # noqa: E731
    step, images_seen, epoch = 0, 0, 0
    total_ssl_loss, total_consistency_loss, total_loss = 0.0, 0.0, 0.0
    consistency_steps = 0

    while not compute.should_stop(elapsed()) and (
        cfg.max_steps == 0 or step < cfg.max_steps
    ):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            fraction = compute.fraction_done(elapsed())
            lr = schedules.lr_at(fraction, base_lr, cfg.warmup_fraction)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg.precision, device):
                if cfg.method in config_mod.SIMSIAM_METHODS:
                    v1, v2 = batch[0].to(device, non_blocking=True), batch[1].to(
                        device, non_blocking=True
                    )
                    p1, p2, z1, z2 = model(v1, v2)
                    ssl_loss = torch_simsiam.simsiam_loss(p1, p2, z1, z2)
                    if cfg.adaptation_mode == "original_transformed_consistency":
                        consistency_loss = (
                            torch_simsiam.original_transformed_consistency_loss(
                                p1, p2, z1, z2
                            )
                        )
                        loss = ssl_loss + cfg.consistency_weight * consistency_loss
                    else:
                        consistency_loss = None
                        loss = ssl_loss
                    health = torch_simsiam.collapse_indicator(z1)
                    batch_size, guided_rate = v1.shape[0], -1.0
                else:
                    crops, guided = batch
                    crops = [c.to(device, non_blocking=True) for c in crops]
                    raw = model.module if isinstance(model, DDP) else model
                    student_out = raw.student_forward(crops)
                    teacher_out = raw.teacher_forward(crops[:2])
                    loss = loss_fn(student_out, teacher_out)
                    ssl_loss = loss
                    consistency_loss = None  # DINO arm has no transformed-pair term (SimSiam only).
                    health = float(teacher_out[0].std(dim=0).mean().item())
                    batch_size, guided_rate = crops[0].shape[0], float(guided.mean())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 3.0
            )
            scaler.step(optimizer)
            scaler.update()

            if cfg.method in config_mod.DINO_METHODS:
                raw = model.module if isinstance(model, DDP) else model
                raw.ema_update(schedules.ema_momentum_at(fraction, cfg.momentum_teacher))

            step += 1
            images_seen += batch_size * world_size
            total_ssl_loss += float(ssl_loss.detach().item())
            total_loss += float(loss.detach().item())
            if consistency_loss is not None:
                total_consistency_loss += float(consistency_loss.detach().item())
                consistency_steps += 1
            if rank == 0 and step % LOG_EVERY_STEPS == 0:
                elapsed_seconds = elapsed()
                tracker.log_metric("ssl_loss", float(ssl_loss.item()), step=step)
                tracker.log_metric("total_loss", float(loss.item()), step=step)
                if consistency_loss is not None:
                    tracker.log_metric(
                        "consistency_loss", float(consistency_loss.item()), step=step
                    )
                tracker.log_metric("lr", lr, step=step)
                tracker.log_metric("embedding_std", health, step=step)
                if gpu_count:
                    gpu_hours = compute.gpu_hours_elapsed(elapsed_seconds)
                    tracker.log_metric("gpu_hours", gpu_hours, step=step)
                    tracker.log_metric(
                        "images_per_gpu_hour",
                        images_seen / max(gpu_hours, 1e-9),
                        step=step,
                    )
                else:
                    tracker.log_metric("wall_seconds", elapsed_seconds, step=step)
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
                    print(f"[{tag}] guardado en {compute.gpu_hours_elapsed(elapsed()):.2f} horas-GPU")

            if compute.should_stop(elapsed()) or (
                cfg.max_steps and step >= cfg.max_steps
            ):
                break
        epoch += 1

    final_path = out_dir / f"{cfg.method}_backbone_final.pt"
    if rank == 0:
        _export_backbone(final_path, model, cfg.method)
        tracker.log_artifact(str(final_path))
        if gpu_count:
            tracker.log_metric("total_gpu_hours", compute.gpu_hours_elapsed(elapsed()))
        else:
            tracker.log_metric("total_wall_seconds", elapsed())
        tracker.log_metric("total_images_seen", float(images_seen))
        run_contract.write_run_manifest(
            run_contract.record_execution_evidence(
                manifest,
                device=str(device),
                steps_executed=step,
                image_count=len(dataset),
                pair_count=(
                    images_seen
                    if cfg.adaptation_mode == "original_transformed_consistency"
                    else 0
                ),
                ssl_loss=total_ssl_loss / step if step else None,
                consistency_loss=(
                    total_consistency_loss / consistency_steps
                    if consistency_steps
                    else None
                ),
                dataset_image_count=len(dataset),
                dataset_pair_count=(
                    len(dataset)
                    if cfg.adaptation_mode == "original_transformed_consistency"
                    else 0
                ),
                samples_processed=images_seen,
                total_loss=total_loss / step if step else None,
            ),
            out_dir / "run.json",
        )
        tracker.end()
    if world_size > 1:
        dist.destroy_process_group()
    return final_path
