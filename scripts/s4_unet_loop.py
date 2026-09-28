"""S4-UNet training loop: the student's protocol with a frozen DINOv2 encoder.

Everything that defines the student's protocol is imported, not copied: the
confident-pixel loss (``student_engine``), the exclusion of samples the
uncertainty module ignored (``student_data``), the pseudo-validation split
(``cache_io``), the augmentation (``gpu_augment.student_augment``), batching,
cosine schedule and checkpoint selection. Only the model differs.

Resume is exact at epoch boundaries: batches and augmentation are pure
functions of (seed, epoch), and the decoder, optimizer and selector state are
restored from ``last.pt``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_PLAN = REPO / "plan-segmentacion" / "comun"
for _extra in (REPO / "src", _PLAN / "nucleo", _PLAN / "estudiante"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import torch  # noqa: E402

import dist_utils as du  # noqa: E402
import gpu_augment as ga  # noqa: E402
from cache_io import open_cache, read_manifest, split_indices  # noqa: E402
from ckpt_io import load_checkpoint, save_checkpoint  # noqa: E402
from dermssl import s4_unet as s4  # noqa: E402
from run_logging import JsonlLogger, ThroughputMeter, timing_summary  # noqa: E402
from sampling import shard_batches  # noqa: E402
from schedules import cosine_lr  # noqa: E402
from selection import CheckpointSelector  # noqa: E402
from student_data import indices_with_confident_pixels, require_training_images  # noqa: E402
from student_engine import masked_bce_dice_loss, summed_masked_loss  # noqa: E402
from wandb_logging import sink_from_args  # noqa: E402

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"
SMOKE_WARMUP_STEPS = 3
SMOKE_VAL_BATCHES = 5


def load_backbone(arch: str, source: str):
    """The frozen DINOv2 (tests replace it with a fake)."""
    return s4.hub_backbone(arch, source)


def after_epoch(epoch: int) -> None:
    """Hook after each epoch's checkpoint; tests use it to simulate a kill."""


class Run:
    """Data, model and optimizer of one S4-UNet training run."""

    def __init__(self, args, context):
        self.args, self.context = args, context
        cache = Path(args.cache_dir)
        self.images = open_cache(cache / "images.npy", preload=args.preload)
        self.masks = open_cache(Path(args.pseudo_masks))
        if self.masks.shape != self.images.shape[:3]:
            raise SystemExit(f"error: pseudo-masks {self.masks.shape} != cache {self.images.shape}")
        rows = read_manifest(cache / "manifest.csv")
        self.train_indices = require_training_images(
            indices_with_confident_pixels(self.masks, split_indices(rows, "train")))
        self.val_indices = indices_with_confident_pixels(self.masks, split_indices(rows, "val"))
        du.set_seed(args.seed)
        arch, source = args.backbone
        self.model = s4.S4UNet(load_backbone(arch, source), features=args.features)
        self.model.to(context.device)
        optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
        self.optimizer = optimizer_class(self.model.decoder.parameters(), lr=args.lr,
                                         weight_decay=args.weight_decay)
        self.decoder = du.wrap_ddp(self.model.decoder, context)

    @property
    def resolution(self) -> int:
        return int(self.images.shape[1])

    def steps_per_epoch(self) -> int:
        steps = len(shard_batches(self.train_indices, self.args.batch_size, 0,
                                  self.context.world_size, 0, self.args.seed))
        cap = self.args.max_steps_per_epoch
        return min(steps, cap) if cap > 0 else steps

    def batch(self, indices):
        device = self.context.device
        raw = torch.from_numpy(self.images[indices]).to(device, non_blocking=True)
        target = torch.from_numpy(self.masks[indices]).to(device).float().unsqueeze(1)
        return ga.to_float_batch(raw), target

    def logits(self, batch: torch.Tensor, decoder=None) -> torch.Tensor:
        with du.autocast_context(self.context.device, self.args.precision):
            normalized = ga.normalize_batch(batch)
            return (decoder or self.decoder)(self.model.encode(normalized), normalized.shape[-2:])

    def train_step(self, indices, generator, learning_rate: float) -> torch.Tensor:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        batch, target = self.batch(indices)
        batch, target = ga.student_augment(batch, target, generator)
        loss, _ = masked_bce_dice_loss(self.logits(batch).float(), target, self.args.dice_weight)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def validation_loss(self, max_batches: int = 0) -> float:
        self.model.eval()
        total = torch.zeros((), device=self.context.device)
        valid = torch.zeros((), device=self.context.device)
        batches = shard_batches(self.val_indices, self.args.batch_size, self.context.rank,
                                self.context.world_size, 0, self.args.seed,
                                shuffle=False, drop_last=False)
        for indices in batches[:max_batches] if max_batches else batches:
            batch, target = self.batch(indices)
            logits = self.logits(batch, self.model.decoder).float()
            batch_total, batch_valid = summed_masked_loss(logits, target, self.args.dice_weight)
            total += batch_total
            valid += batch_valid.float()
        total, valid = du.all_reduce_sum(total, self.context), du.all_reduce_sum(valid, self.context)
        return float(total / valid.clamp(min=1.0))

    def val_batches(self) -> int:
        per_rank = -(-len(self.val_indices) // self.context.world_size)
        return -(-per_rank // self.args.batch_size)


def _sync(context) -> None:
    if context.device.type == "cuda":
        torch.cuda.synchronize()


def _checkpoint(run: Run, config: dict, epoch: int, selector, history) -> dict:
    return {"model": s4.decoder_state(run.model), "optimizer": run.optimizer.state_dict(),
            "epoch": epoch, "selector": selector.state_dict(), "history": history,
            "config": config}


def train(run: Run, config: dict) -> dict:
    args, context = run.args, run.context
    output_dir = Path(args.output_dir)
    selector = CheckpointSelector("last" if args.selection == "best-val" and
                                  len(run.val_indices) == 0 else args.selection)
    history, start_epoch = [], 0
    last_path = output_dir / LAST_CHECKPOINT
    if last_path.is_file() and not args.no_resume:
        state = load_checkpoint(last_path)
        run.model.decoder.load_state_dict({k[len("decoder."):]: v for k, v in state["model"].items()})
        run.optimizer.load_state_dict(state["optimizer"])
        selector = CheckpointSelector.from_state(state["selector"])
        history, start_epoch = state["history"], state["epoch"] + 1
    steps = run.steps_per_epoch()
    total_steps, warmup = steps * args.epochs, steps * args.warmup_epochs
    sink = sink_from_args(args, "s4_unet", is_main=context.is_main)
    logger = JsonlLogger(output_dir / "metrics.jsonl", sinks=(sink,)) if context.is_main else None
    meter = ThroughputMeter()
    for epoch in range(start_epoch, args.epochs):
        run.model.train()
        generator = du.make_generator(context, args.seed, epoch)
        batches = shard_batches(run.train_indices, args.batch_size, context.rank,
                                context.world_size, epoch, args.seed)[:steps]
        loss_sum = torch.zeros((), device=context.device)
        _sync(context)
        meter.start()
        for step, indices in enumerate(batches):
            rate = cosine_lr(epoch * steps + step, total_steps, args.lr, warmup)
            loss_sum += run.train_step(indices, generator, rate)
            meter.add(args.batch_size * context.world_size)
            if context.is_main and (step + 1) % args.log_every == 0:
                print(f"s4 epoca {epoch} paso {step + 1}/{len(batches)}", flush=True)
        _sync(context)
        snapshot = meter.snapshot()
        train_loss = float(du.all_reduce_sum(loss_sum, context)) / (len(batches) * context.world_size)
        val_loss = run.validation_loss()
        selected = selector.update(epoch, val_loss)
        history.append({"seconds": snapshot["seconds"], "images": snapshot["images"]})
        if context.is_main:
            logger.log({"stage": "s4_unet", "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "selected": selected, "lr": rate, **snapshot})
            if selected:
                save_checkpoint(output_dir / BEST_CHECKPOINT,
                                {"model": s4.decoder_state(run.model), "config": config,
                                 "epoch": epoch, "val_loss": val_loss})
            save_checkpoint(last_path, _checkpoint(run, config, epoch, selector, history))
        du.barrier(context)
        after_epoch(epoch)
    sink.finish()
    return {**timing_summary(history), "selection": selector.state_dict()}


def timing_smoke(run: Run, steps_timed: int) -> dict:
    """s/step on real batches; projects the registered schedule in GPU-hours."""
    args, context = run.args, run.context
    run.model.train()
    generator = du.make_generator(context, args.seed, 0)
    batches = shard_batches(run.train_indices, args.batch_size, context.rank,
                            context.world_size, 0, args.seed)
    needed = SMOKE_WARMUP_STEPS + steps_timed
    batches = (batches * (needed // len(batches) + 1))[:needed]
    for indices in batches[:SMOKE_WARMUP_STEPS]:
        run.train_step(indices, generator, args.lr)
    _sync(context)
    began = time.perf_counter()
    for indices in batches[SMOKE_WARMUP_STEPS:]:
        run.train_step(indices, generator, args.lr)
    _sync(context)
    per_step = (time.perf_counter() - began) / steps_timed
    val_batches = min(SMOKE_VAL_BATCHES, run.val_batches())
    began = time.perf_counter()
    run.validation_loss(max_batches=val_batches)
    _sync(context)
    per_val = (time.perf_counter() - began) / max(1, val_batches)
    steps = run.steps_per_epoch()
    epoch_seconds = per_step * steps + per_val * run.val_batches()
    return {"steps_timed": steps_timed, "seconds_per_step": per_step,
            "seconds_per_val_batch": per_val, "steps_per_epoch": steps,
            "val_batches_per_epoch": run.val_batches(), "epochs": args.epochs,
            "world_size": context.world_size, "batch_size_per_gpu": args.batch_size,
            "projected_wall_hours": epoch_seconds * args.epochs / 3600.0,
            "projected_gpu_hours": epoch_seconds * args.epochs * context.world_size / 3600.0,
            "n_train_confident": int(len(run.train_indices)), "device": str(context.device)}


def confident_counts(run: Run) -> dict:
    return {"n_train_confident": int(len(run.train_indices)),
            "n_val_confident": int(len(run.val_indices)),
            "train_indices_sha256": _indices_digest(run.train_indices)}


def _indices_digest(indices: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
