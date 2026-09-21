"""Entrena la cabeza CCAM sobre el backbone contrastivo congelado y guarda la saliencia.

Camino B, etapa 2. El backbone (`backbone.pt` de train_moco.py) se usa congelado
con el ultimo stride en 1. Solo se entrena la cabeza (conv + BN + sigmoide) con
la perdida contrastiva de CCAM. Luego escribe `saliency.npy` (N, R, R, float16)
para TODAS las imagenes de la cache; `make_pseudo_masks.py` la convierte en
pseudo-mascaras. No lee mascaras ni etiquetas.

    python train_ccam.py --cache-dir salidas/cache224 \\
        --backbone salidas/moco/backbone.pt --output-dir salidas/ccam
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (
    _PLAN_ROOT.parent / "src",
    _PLAN_ROOT / "comun" / "nucleo",
    _PLAN_ROOT / "camino-b-cnn" / "preentrenamiento-contrastivo",
    Path(__file__).resolve().parent,
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dermssl.cli import build_spanish_parser  # noqa: E402
from wandb_logging import add_tracking_arguments, sink_from_args  # noqa: E402

HEAD_CHECKPOINT = "last.pt"
SALIENCY_FILE = "saliency.npy"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Cache creada por prepare_cache.py")
    parser.add_argument("--backbone", required=True, help="backbone.pt del preentrenamiento contrastivo")
    parser.add_argument("--output-dir", required=True, help="Carpeta de salida (cabeza y saliencia)")
    parser.add_argument("--epochs", type=int, default=3, help="Epocas de la cabeza CCAM")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamano de lote POR GPU (>= 2)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje de Adam")
    parser.add_argument("--alpha", type=float, default=0.25, help="Suavidad de los pesos por rango de CCAM")
    parser.add_argument("--head-kernel", type=int, default=3, choices=(1, 3), help="Kernel de la cabeza")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"), help="Precision numerica")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="Limita pasos por epoca (0 = completa)")
    parser.add_argument("--predict-batch-size", type=int, default=128, help="Lote al generar la saliencia")
    parser.add_argument("--log-every", type=int, default=50, help="Pasos entre mensajes en pantalla")
    parser.add_argument("--no-resume", action="store_true", help="Ignora last.pt y empieza de cero")
    preload = parser.add_mutually_exclusive_group()
    preload.add_argument("--preload", action="store_true", help="Precarga images.npy en RAM")
    preload.add_argument("--no-preload", action="store_false", dest="preload", help="Conserva images.npy como memmap")
    parser.set_defaults(preload=False)
    parser.add_argument("--force-cpu", action="store_true", help="Fuerza CPU (diagnostico)")
    add_tracking_arguments(parser)
    return parser.parse_args(argv)


def _augment(base, generator):
    import gpu_augment as ga

    out, _ = ga.random_resized_crop(base, generator, scale=(0.5, 1.0), ratio=(0.8, 1.25))
    return ga.normalize_batch(ga.color_jitter(out, generator, 0.2, 0.2, 0.2, 0.0, probability=0.5))


def train_head(args, context, provider, head, images, train_indices, output_dir, sink=None) -> list[dict]:
    import torch

    import dist_utils as du
    import gpu_augment as ga
    from ccam_engine import ccam_loss
    from ckpt_io import load_checkpoint, save_checkpoint
    from run_logging import JsonlLogger, ThroughputMeter
    from sampling import shard_batches

    device = context.device
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    history: list[dict] = []
    start_epoch = 0
    last_path = output_dir / HEAD_CHECKPOINT
    if last_path.is_file() and not args.no_resume:
        state = load_checkpoint(last_path)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        history, start_epoch = state["history"], state["epoch"] + 1

    trainable = du.wrap_ddp(head, context)
    sinks = (sink,) if sink is not None else ()
    logger = JsonlLogger(output_dir / "metrics.jsonl", echo=True, sinks=sinks) if context.is_main else None
    meter = ThroughputMeter()
    for epoch in range(start_epoch, args.epochs):
        head.train()
        generator = du.make_generator(context, args.seed, epoch)
        batches = shard_batches(
            train_indices, args.batch_size, context.rank, context.world_size, epoch, args.seed
        )
        if args.max_steps_per_epoch > 0:
            batches = batches[: args.max_steps_per_epoch]
        loss_sum = torch.zeros((), device=device)
        meter.start()
        for step, batch_indices in enumerate(batches):
            raw = torch.from_numpy(images[batch_indices]).to(device, non_blocking=True)
            view = _augment(ga.to_float_batch(raw), generator)
            with du.autocast_context(device, args.precision):
                features = provider.extract(view).float()
            loss = ccam_loss(features, trainable(features), args.alpha)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            meter.add(args.batch_size * context.world_size)
            if context.is_main and (step + 1) % args.log_every == 0:
                print(f"ccam epoca {epoch} paso {step + 1}/{len(batches)} perdida {loss.item():.4f}", flush=True)
        snapshot = meter.snapshot()
        mean_loss = float(du.all_reduce_sum(loss_sum, context)) / (len(batches) * context.world_size)
        history.append({"seconds": snapshot["seconds"], "images": snapshot["images"]})
        if context.is_main:
            logger.log({"stage": "ccam", "epoch": epoch, "loss": mean_loss, **snapshot})
            save_checkpoint(
                last_path,
                {"head": head.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "history": history},
            )
    return history


def write_saliency(args, context, provider, head, images, output_dir) -> Path:
    import numpy as np
    import torch
    import torch.nn.functional as F

    import dist_utils as du
    import gpu_augment as ga
    from cache_io import create_cache, open_cache
    from sampling import shard_batches

    path = output_dir / SALIENCY_FILE
    total, resolution = int(images.shape[0]), int(images.shape[1])
    if context.is_main:
        create_cache(path, (total, resolution, resolution), np.float16).flush()
    du.barrier(context)
    out = open_cache(path, "r+")

    head.eval()
    batches = shard_batches(
        np.arange(total), args.predict_batch_size, context.rank, context.world_size,
        0, args.seed, shuffle=False, drop_last=False,
    )
    with torch.no_grad():
        for batch_indices in batches:
            raw = torch.from_numpy(images[batch_indices]).to(context.device)
            batch = ga.normalize_batch(ga.to_float_batch(raw))
            with du.autocast_context(context.device, args.precision):
                features = provider.extract(batch).float()
            activation = head(features)
            activation = F.interpolate(
                activation, size=(resolution, resolution), mode="bilinear", align_corners=False
            )
            out[batch_indices] = activation.squeeze(1).cpu().numpy().astype(np.float16)
    out.flush()
    du.barrier(context)
    return path


def run(args: argparse.Namespace) -> dict:
    import dist_utils as du
    from cache_io import open_cache, read_manifest, split_indices
    from ccam_engine import CCAMHead
    from contrastive_backbone import ResNet50MoCoFeatures
    from run_logging import timing_summary

    context = du.init_distributed(force_cpu=args.force_cpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    du.set_seed(args.seed)

    images = open_cache(Path(args.cache_dir) / "images.npy", preload=args.preload)
    train_indices = split_indices(read_manifest(Path(args.cache_dir) / "manifest.csv"), "train")
    provider = ResNet50MoCoFeatures.from_checkpoint(Path(args.backbone)).to(context.device)
    head = CCAMHead(provider.out_channels, args.head_kernel).to(context.device)

    sink = sink_from_args(args, "ccam", is_main=context.is_main)
    history = train_head(args, context, provider, head, images, train_indices, output_dir, sink)
    saliency_path = write_saliency(args, context, provider, head, images, output_dir)

    summary = timing_summary(history)
    if context.is_main:
        (output_dir / "timing.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"saliencia escrita en {saliency_path}")
    sink.finish()
    du.cleanup(context)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
