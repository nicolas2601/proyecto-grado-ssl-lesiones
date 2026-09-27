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
    torch_adaptation as adaptation,
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
    """(model, loss_fn, optimizer, dataset, collate, extras) for the configured method.

    ``extras`` carries what the loop needs beyond the v6 tuple: the levers,
    the anchors built from the STARTING weights (before any resume is loaded)
    and how many images the id exclusion removed.
    """
    excluded = (
        config_mod.load_excluded_ids(cfg.exclude_ids_file) if cfg.exclude_ids_file else frozenset()
    )
    levers = config_mod.levers_of(cfg)
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
        paths, exclusion = _exclude(paths, excluded)
        extras = {"levers": levers, "exclusion": exclusion}
        pretrained = cfg.method in ("simsiam_dinov2", "simsiam_imagenet")
        model = torch_simsiam.SimSiam(cfg.backbone, pretrained=pretrained).to(device)
        extras["anchors"] = adaptation.build_anchors(model.backbone, levers)
        adaptation.freeze_blocks(model.backbone, levers.freeze_blocks)
        if cfg.grad_checkpoint:
            adaptation.enable_grad_checkpoint(model.backbone)
        dataset = torch_data.SimSiamPairDataset(
            paths,
            cfg.resolution,
            cfg.color_constancy_probs,
            cfg.seed,
            cfg.hue_jitter,
            tone_view=levers.tone_consistency_weight > 0,
        )
        if pretrained:
            # Converged ViT weights: AdamW, like the continual DINO arm.
            optimizer = torch.optim.AdamW(
                adaptation.param_groups(model, levers), lr=0.0, weight_decay=cfg.weight_decay
            )
        else:
            # Chen & He 2021 from-scratch recipe: SGD with momentum.
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.0, momentum=0.9, weight_decay=cfg.weight_decay
            )
        return model, None, optimizer, dataset, None, extras

    paths, exclusion = _exclude(paths, excluded)
    extras = {"levers": levers, "exclusion": exclusion, "anchors": adaptation.Anchors()}
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
    return model, loss_fn, optimizer, dataset, torch_data.dino_collate, extras


def _exclude(paths: list, excluded: frozenset[str]) -> tuple[list, dict | None]:
    if not excluded:
        return paths, None
    return config_mod.apply_exclusion(paths, excluded)


def _apply_head_warmup(model, levers, step: int) -> None:
    """During the first ``head_warmup_steps`` drop every backbone gradient.

    ``grad = None`` makes AdamW skip the parameter entirely: no update, no
    weight decay, no optimizer state. The graph is unchanged, so DDP with a
    static graph keeps working; only the step is withheld (LP-FT, docs/28).
    """
    if step >= levers.head_warmup_steps:
        return
    raw = model.module if isinstance(model, DDP) else model
    for param in raw.backbone.parameters():
        param.grad = None


def _lever_objective(base_loss, levers, anchors, model, outputs, original_view):
    """``base_loss`` plus the docs/28 levers; with every lever off, ``base_loss`` itself."""
    if levers.is_default():
        return base_loss, {}
    raw = model.module if isinstance(model, DDP) else model
    h_original = outputs[4] if len(outputs) > 4 else None
    return adaptation.compose_objective(
        base_loss,
        levers,
        tone=(
            adaptation.tone_consistency(h_original, outputs[6])
            if levers.tone_consistency_weight > 0 else None
        ),
        l2sp=adaptation.l2sp_penalty(raw.backbone, anchors) if levers.anchor_l2sp > 0 else None,
        distill=(
            adaptation.distill_loss(h_original, anchors, original_view)
            if levers.anchor_distill > 0 else None
        ),
    )


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    elapsed: float,
    saved_fractions: set[float],
    step: int = 0,
    images_seen: int = 0,
    loss_totals: dict | None = None,
) -> None:
    raw = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "elapsed_wall_seconds": elapsed,
            "saved_fractions": sorted(saved_fractions),
            "step": int(step),
            "images_seen": int(images_seen),
            "loss_totals": loss_totals,
        },
        path,
    )


def _clip_gradients(parameters, max_norm: float) -> float:
    """Clip in place and return the pre-clip norm, so it can be logged.

    ``clip_grad_norm_`` already computes the norm; dropping its return value
    is what left us unable to say how much of a loss rescaling survived the
    clip. Measuring it costs nothing and turns that argument into a number.
    """
    return float(torch.nn.utils.clip_grad_norm_(parameters, max_norm))


def _counters_from(state: dict) -> tuple[int, int]:
    """``(step, images_seen)`` recorded in a resume checkpoint.

    Checkpoints written before 2026-09-22 carry neither, and a resumed run
    that restarts them at zero reports only the tail of the budget: the arm
    that died at 80 percent looked like it had seen one epoch instead of
    five. Missing keys default to zero so an old checkpoint still loads.
    """
    return int(state.get("step", 0)), int(state.get("images_seen", 0))


def _wrap_ddp(model, local_rank: int, device):
    """DDP that tolerates parameters the forward never touches.

    DINOv2 registers ``mask_token``, which only participates when a mask is
    passed to the forward - never in this loop. DDP's reducer treats an
    unused parameter as a bug ("Expected to have finished reduction in the
    prior iteration") unless it is told the graph is identical every step,
    which here it is: the same two views through the same modules. A static
    graph buys that tolerance without the per-iteration traversal that
    ``find_unused_parameters`` costs, and it is what makes the two-GPU arms
    of the thesis run at all.
    """
    return DDP(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
        static_graph=True,
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


def train(
    cfg: config_mod.PretrainConfig,
    mlflow_module=None,
    wandb_module=None,
    wandb_project: str = "",
    wandb_group: str = "",
) -> Path:
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

    model, loss_fn, optimizer, dataset, collate, extras = _build(
        cfg, device, transformed_evidence
    )
    levers, anchors = extras["levers"], extras["anchors"]
    # Written BEFORE training: a run that dies must still show what it excluded.
    manifest["exclusion"] = extras["exclusion"]
    if rank == 0:
        run_contract.write_run_manifest(manifest, out_dir / "run.json")
    if world_size > 1:
        model = _wrap_ddp(model, local_rank, device)
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
        run=tracking.run_name(
            cfg.method, cfg.backbone, cfg.seed, arm=Path(cfg.output_dir).name
        ),
        rank=rank,
        wandb_module=wandb_module,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_run_id=tracking.attempt_run_id(cfg.output_dir) if rank == 0 else "",
    )
    if rank == 0 and isinstance(tracker, tracking.NullTracker):
        print("ADVERTENCIA: ni MLflow ni Weights & Biases están disponibles; esta ejecución no tendrá seguimiento.")
    params = config_mod.mlflow_params(cfg)
    params["world_size"] = world_size
    params["effective_lr"] = base_lr
    tracker.start(params)

    elapsed_offset, saved_fractions = 0.0, set()
    resumed_counters = (0, 0)
    totals = adaptation.LossTotals()
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
        resumed_counters = _counters_from(state)
        totals = adaptation.LossTotals.from_state(state.get("loss_totals"))
        if rank == 0:
            if gpu_count:
                print(
                    f"Reanudado en {compute.gpu_hours_elapsed(elapsed_offset):.2f} horas-GPU"
                )
            else:
                print(f"Reanudado tras {elapsed_offset:.2f} segundos de reloj")

    start = time.monotonic()
    elapsed = lambda: elapsed_offset + (time.monotonic() - start)  # noqa: E731
    step, images_seen = resumed_counters
    epoch = 0
    need_features = levers.tone_consistency_weight > 0 or levers.anchor_distill > 0

    while not compute.should_stop(elapsed()) and (
        cfg.max_steps == 0 or step < cfg.max_steps
    ):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            fraction = schedules.progress(
                step=step,
                max_steps=cfg.max_steps,
                compute_fraction=compute.fraction_done(elapsed()),
            )
            lr = schedules.lr_at(fraction, base_lr, cfg.warmup_fraction)
            adaptation.apply_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)
            with _autocast(cfg.precision, device):
                if cfg.method in config_mod.SIMSIAM_METHODS:
                    v1, v2 = batch[0].to(device, non_blocking=True), batch[1].to(
                        device, non_blocking=True
                    )
                    if levers.tone_consistency_weight > 0:
                        v_tone = batch[2].to(device, non_blocking=True)
                        outputs = model(v1, v2, return_features=True, tone_view=v_tone)
                    elif need_features:
                        outputs = model(v1, v2, return_features=True)
                    else:
                        outputs = model(v1, v2)
                    p1, p2, z1, z2 = outputs[:4]
                    ssl_loss = torch_simsiam.simsiam_loss(p1, p2, z1, z2)
                    if cfg.adaptation_mode == "original_transformed_consistency":
                        base_loss, consistency_loss = torch_simsiam.paired_objective(
                            ssl_loss, cfg.consistency_weight
                        )
                    else:
                        consistency_loss = None
                        base_loss = ssl_loss
                    loss, lever_parts = _lever_objective(
                        base_loss, levers, anchors, model, outputs, v1
                    )
                    health = torch_simsiam.collapse_indicator(z1)
                    if need_features:
                        # The tone term has no predictor: watch the backbone itself.
                        lever_parts["backbone_std"] = torch.tensor(
                            torch_simsiam.collapse_indicator(outputs[-1])
                        )
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
                    lever_parts = {}
                    health = float(teacher_out[0].std(dim=0).mean().item())
                    batch_size, guided_rate = crops[0].shape[0], float(guided.mean())

            scaler.scale(loss).backward()
            if cfg.method in config_mod.SIMSIAM_METHODS:
                _apply_head_warmup(model, levers, step)
            scaler.unscale_(optimizer)
            grad_norm = _clip_gradients(
                [p for g in optimizer.param_groups for p in g["params"]],
                config_mod.GRAD_CLIP_NORM,
            )
            scaler.step(optimizer)
            scaler.update()

            if cfg.method in config_mod.DINO_METHODS:
                raw = model.module if isinstance(model, DDP) else model
                raw.ema_update(schedules.ema_momentum_at(fraction, cfg.momentum_teacher))

            step += 1
            images_seen += batch_size * world_size
            totals.add(
                ssl=float(ssl_loss.detach().item()),
                total=float(loss.detach().item()),
                consistency=(
                    None if consistency_loss is None else float(consistency_loss.detach().item())
                ),
            )
            if rank == 0 and step % LOG_EVERY_STEPS == 0:
                elapsed_seconds = elapsed()
                tracker.log_metric("ssl_loss", float(ssl_loss.item()), step=step)
                tracker.log_metric("total_loss", float(loss.item()), step=step)
                if consistency_loss is not None:
                    tracker.log_metric(
                        "consistency_loss", float(consistency_loss.item()), step=step
                    )
                for name, value in lever_parts.items():
                    tracker.log_metric(f"lever_{name}", float(value.item()), step=step)
                tracker.log_metric("lr", lr, step=step)
                tracker.log_metric("embedding_std", health, step=step)
                tracker.log_metric("grad_norm", grad_norm, step=step)
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
                    _save_checkpoint(
                        resume_path, model, optimizer, elapsed(), saved_fractions,
                        step=step, images_seen=images_seen,
                        loss_totals=totals.state(),
                    )
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
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device) / 2**30
            tracker.log_metric("max_memory_allocated_gib", peak)
            print(f"max_memory_allocated_gib {peak:.2f}", flush=True)
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
                ssl_loss=totals.mean("ssl"),
                consistency_loss=totals.mean("consistency"),
                dataset_image_count=len(dataset),
                dataset_pair_count=(
                    len(dataset)
                    if cfg.adaptation_mode == "original_transformed_consistency"
                    else 0
                ),
                samples_processed=images_seen,
                total_loss=totals.mean("total"),
            ),
            out_dir / "run.json",
        )
        tracker.end()
    if world_size > 1:
        dist.destroy_process_group()
    return final_path
