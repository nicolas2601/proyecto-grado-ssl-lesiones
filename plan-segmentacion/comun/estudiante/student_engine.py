"""Student U-Net trained on pseudo-masks: loss, training loop and prediction.

Label-source agnostic on purpose: it consumes a ``pseudo_masks.npy`` (0 =
background, 1 = lesion, 255 = ignore) from ANY path (CNN saliency or DINOv2
teacher), so the pseudo-label source is the only variable between paths.

Nothing here opens a manual mask or a clinical label. The checkpoint is
selected by the loss on confident pseudo-label pixels of the held-out pool
split (or the last epoch), never by ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import dist_utils as du
import gpu_augment as ga
from cache_io import IGNORE, open_cache, read_manifest, split_indices, create_cache
from ckpt_io import load_checkpoint, save_checkpoint
from dermssl.torch_unet import UNet
from run_logging import JsonlLogger, ThroughputMeter, timing_summary
from sampling import shard_batches
from schedules import cosine_lr
from selection import CheckpointSelector
from student_data import indices_with_confident_pixels, require_training_images

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"
PREDICTIONS_FILE = "predictions.npy"


# --------------------------------------------------------------------- loss
def per_sample_losses(
    logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0, smooth: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE + soft Dice over confident pixels, per sample.

    ``target`` holds 0 (background), 1 (lesion) and 255 (ignore). Returns the
    per-sample loss and a boolean mask of samples that have confident pixels.
    """
    confident = (target != IGNORE).float()
    positive = (target == 1).float()
    bce = F.binary_cross_entropy_with_logits(logits, positive, reduction="none")
    counts = confident.sum(dim=(1, 2, 3))
    bce_loss = (bce * confident).sum(dim=(1, 2, 3)) / counts.clamp(min=1.0)

    probability = torch.sigmoid(logits) * confident
    intersection = (probability * positive).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + positive.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

    has_confident = counts > 0
    return bce_loss + dice_weight * dice_loss, has_confident


def summed_masked_loss(
    logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of per-sample losses over valid samples and the valid-sample count."""
    losses, valid = per_sample_losses(logits, target, dice_weight)
    return (losses * valid).sum(), valid.sum()


def masked_bce_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, dice_weight: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean loss over samples with confident pixels, and how many there were."""
    total, valid = summed_masked_loss(logits, target, dice_weight)
    return total / valid.clamp(min=1), valid


# -------------------------------------------------------------------- model
def build_student(base_channels: int) -> UNet:
    return UNet(in_channels=3, base_channels=base_channels)


def _load_batch(images, masks, indices, device):
    raw = torch.from_numpy(images[indices]).to(device, non_blocking=True)
    target = torch.from_numpy(masks[indices]).to(device, non_blocking=True).float().unsqueeze(1)
    return ga.to_float_batch(raw), target


@torch.no_grad()
def validation_loss(model, images, masks, indices, context, args) -> float:
    """Pseudo-validation loss on confident pixels, reduced over ranks."""
    model.eval()
    total = torch.zeros((), device=context.device)
    valid = torch.zeros((), device=context.device)
    batches = shard_batches(
        indices, args.batch_size, context.rank, context.world_size, 0, args.seed,
        shuffle=False, drop_last=False,
    )
    for batch_indices in batches:
        batch, target = _load_batch(images, masks, batch_indices, context.device)
        with du.autocast_context(context.device, args.precision):
            logits = model(ga.normalize_batch(batch))
        batch_total, batch_valid = summed_masked_loss(logits.float(), target, args.dice_weight)
        total += batch_total
        valid += batch_valid.float()
    total = du.all_reduce_sum(total, context)
    valid = du.all_reduce_sum(valid, context)
    return float(total / valid.clamp(min=1.0))


# ----------------------------------------------------------------- training
def _save_best(path: Path, model, args, resolution: int, epoch: int, val_loss: float) -> None:
    save_checkpoint(
        path,
        {
            "model": model.state_dict(),
            "config": {"base_channels": args.base_channels, "resolution": resolution},
            "epoch": epoch,
            "val_loss": val_loss,
        },
    )


def _resolve_selection_mode(args, val_count: int, context) -> str:
    if args.selection == "best-val" and val_count == 0:
        if context.is_main:
            print("aviso: la particion val no tiene pixeles confiables; se usa la ultima epoca")
        return "last"
    return args.selection


def train(args, context) -> dict:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = context.device
    du.set_seed(args.seed)

    images = open_cache(Path(args.cache_dir) / "images.npy", preload=args.preload)
    masks = open_cache(Path(args.pseudo_masks))
    if masks.shape != images.shape[:3]:
        raise SystemExit(f"error: pseudo-mascaras {masks.shape} no coinciden con la cache {images.shape}")
    resolution = int(images.shape[1])
    rows = read_manifest(Path(args.cache_dir) / "manifest.csv")
    train_indices = require_training_images(
        indices_with_confident_pixels(masks, split_indices(rows, "train"))
    )
    val_indices = indices_with_confident_pixels(masks, split_indices(rows, "val"))

    model = build_student(args.base_channels).to(device)
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainable = du.wrap_ddp(model, context)
    selector = CheckpointSelector(_resolve_selection_mode(args, len(val_indices), context))

    history: list[dict] = []
    start_epoch = 0
    last_path = output_dir / LAST_CHECKPOINT
    if last_path.is_file() and not args.no_resume:
        state = load_checkpoint(last_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        selector = CheckpointSelector.from_state(state["selector"])
        history, start_epoch = state["history"], state["epoch"] + 1
        if context.is_main:
            print(f"reanudando desde la epoca {start_epoch}")

    steps_per_epoch = len(
        shard_batches(train_indices, args.batch_size, 0, context.world_size, 0, args.seed)
    )
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    logger = JsonlLogger(output_dir / "metrics.jsonl") if context.is_main else None
    meter = ThroughputMeter()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        generator = du.make_generator(context, args.seed, epoch)
        batches = shard_batches(
            train_indices, args.batch_size, context.rank, context.world_size, epoch, args.seed
        )[:steps_per_epoch]
        loss_sum = torch.zeros((), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        meter.start()
        for step, batch_indices in enumerate(batches):
            learning_rate = cosine_lr(epoch * steps_per_epoch + step, total_steps, args.lr, warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch, target = _load_batch(images, masks, batch_indices, device)
            batch, target = ga.student_augment(batch, target, generator)
            with du.autocast_context(device, args.precision):
                logits = trainable(ga.normalize_batch(batch))
            loss, _ = masked_bce_dice_loss(logits.float(), target, args.dice_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            meter.add(args.batch_size * context.world_size)
            if context.is_main and (step + 1) % args.log_every == 0:
                print(f"estudiante epoca {epoch} paso {step + 1}/{len(batches)} perdida {loss.item():.4f}", flush=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        snapshot = meter.snapshot()
        train_loss = float(du.all_reduce_sum(loss_sum, context)) / (len(batches) * context.world_size)
        val_loss = validation_loss(model, images, masks, val_indices, context, args)
        is_selected = selector.update(epoch, val_loss)
        history.append({"seconds": snapshot["seconds"], "images": snapshot["images"]})
        if context.is_main:
            logger.log(
                {"stage": "student", "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 "selected": is_selected, "lr": learning_rate, **snapshot}
            )
            if is_selected:
                _save_best(output_dir / BEST_CHECKPOINT, model, args, resolution, epoch, val_loss)
            save_checkpoint(
                last_path,
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "selector": selector.state_dict(), "history": history},
            )
        du.barrier(context)

    summary = {**timing_summary(history), "selection": selector.state_dict()}
    if context.is_main:
        (output_dir / "timing.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return summary


# --------------------------------------------------------------- prediction
@torch.no_grad()
def write_predictions(args, context, images, output_path: Path) -> Path:
    """Sigmoid probabilities of the selected checkpoint for every cached image."""
    if context.is_main and output_path.exists() and not args.overwrite_predictions:
        raise FileExistsError(
            f"prediction output already exists: {output_path}; use --overwrite-predictions"
        )
    checkpoint = load_checkpoint(Path(args.output_dir) / BEST_CHECKPOINT)
    model = build_student(checkpoint["config"]["base_channels"]).to(context.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    total, resolution = int(images.shape[0]), int(images.shape[1])
    if context.is_main:
        create_cache(output_path, (total, resolution, resolution), np.float16).flush()
    du.barrier(context)
    out = open_cache(output_path, "r+")
    batches = shard_batches(
        np.arange(total), args.batch_size, context.rank, context.world_size, 0, args.seed,
        shuffle=False, drop_last=False,
    )
    for batch_indices in batches:
        raw = torch.from_numpy(images[batch_indices]).to(context.device)
        with du.autocast_context(context.device, args.precision):
            logits = model(ga.normalize_batch(ga.to_float_batch(raw)))
        out[batch_indices] = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy().astype(np.float16)
    out.flush()
    du.barrier(context)
    return output_path
