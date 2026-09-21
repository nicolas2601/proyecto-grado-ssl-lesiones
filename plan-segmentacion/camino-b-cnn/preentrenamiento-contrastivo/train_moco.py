"""Preentrenamiento contrastivo MoCo v2 (ResNet50) sobre el pool sin etiquetas.

Camino B, etapa 1. Parte de pesos aleatorios (sin ImageNet) y entrena con las
imagenes de la cache `train`; la particion `val` no se usa. No lee mascaras ni
etiquetas. Aumentos en GPU por lote, sin procesos de carga. bf16, DDP con
torchrun y reanudacion automatica desde `last.pt`.

Una GPU:
    python train_moco.py --cache-dir salidas/cache224 --output-dir salidas/moco
Dos GPU:
    torchrun --nproc_per_node=2 train_moco.py --cache-dir ... --output-dir ...

Pesos externos (opcional, DESVIACION del escenario puro):
    --init-weights ruta/al/resnet50_ssl.pt
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
    Path(__file__).resolve().parent,
):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dermssl.cli import build_spanish_parser  # noqa: E402

LAST_CHECKPOINT = "last.pt"
BACKBONE_FILE = "backbone.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Cache creada por prepare_cache.py")
    parser.add_argument("--output-dir", required=True, help="Carpeta de checkpoints y metricas")
    parser.add_argument("--epochs", type=int, default=10, help="Epocas de preentrenamiento")
    parser.add_argument("--batch-size", type=int, default=128, help="Tamano de lote POR GPU")
    parser.add_argument("--lr", type=float, default=5e-4, help="Tasa de aprendizaje maxima de AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Decaimiento de pesos de AdamW")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Epocas de calentamiento lineal")
    parser.add_argument("--embed-dim", type=int, default=128, help="Dimension del embedding contrastivo")
    parser.add_argument("--queue-size", type=int, default=16384, help="Negativos en la cola (>= lote global)")
    parser.add_argument("--momentum", type=float, default=0.99, help="Momento del codificador de claves")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperatura de InfoNCE")
    parser.add_argument("--layer4-stride", type=int, default=2, choices=(1, 2), help="Stride de la ultima etapa (CCAM la usa en 1)")
    parser.add_argument("--bn-splits", type=int, default=4, help="Sub-lotes de BatchNorm (emula shuffle BN)")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"), help="Precision numerica")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="Limita pasos por epoca (0 = epoca completa)")
    parser.add_argument("--log-every", type=int, default=50, help="Pasos entre mensajes en pantalla")
    parser.add_argument("--no-resume", action="store_true", help="Ignora last.pt y empieza de cero")
    preload = parser.add_mutually_exclusive_group()
    preload.add_argument("--preload", action="store_true", help="Precarga images.npy en RAM")
    preload.add_argument("--no-preload", action="store_false", dest="preload", help="Conserva images.npy como memmap")
    parser.set_defaults(preload=False)
    parser.add_argument("--init-weights", default="", help="Pesos SSL externos (desvia el escenario puro)")
    parser.add_argument("--force-cpu", action="store_true", help="Fuerza CPU (diagnostico)")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    import torch

    import dist_utils as du
    import gpu_augment as ga
    from cache_io import open_cache, read_manifest, split_indices
    from ckpt_io import load_checkpoint, save_checkpoint
    from contrastive_backbone import MoCoV2, load_external_weights, save_backbone
    from run_logging import JsonlLogger, ThroughputMeter, timing_summary
    from sampling import shard_batches
    from schedules import cosine_lr

    context = du.init_distributed(force_cpu=args.force_cpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = context.device
    du.set_seed(args.seed)  # identical initial weights on every rank

    images = open_cache(Path(args.cache_dir) / "images.npy", preload=args.preload)
    train_indices = split_indices(read_manifest(Path(args.cache_dir) / "manifest.csv"), "train")
    resolution = int(images.shape[1])

    model = MoCoV2(
        embed_dim=args.embed_dim,
        queue_size=args.queue_size,
        momentum=args.momentum,
        temperature=args.temperature,
        layer4_stride=args.layer4_stride,
        bn_splits=args.bn_splits,
    ).to(device)
    if args.init_weights:
        missing, unexpected = load_external_weights(model.encoder_q.trunk, Path(args.init_weights))
        model.encoder_k.load_state_dict(model.encoder_q.state_dict())
        print(f"init-weights externos: faltan {len(missing)}, sobran {len(unexpected)}")
    query_encoder = du.wrap_ddp(model.encoder_q, context)
    optimizer = torch.optim.AdamW(
        model.encoder_q.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    history: list[dict] = []
    start_epoch = 0
    last_path = output_dir / LAST_CHECKPOINT
    if last_path.is_file() and not args.no_resume:
        state = load_checkpoint(last_path)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
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
    logger = JsonlLogger(output_dir / "metrics.jsonl", echo=context.is_main) if context.is_main else None
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
            raw = torch.from_numpy(images[batch_indices]).to(device, non_blocking=True)
            base = ga.to_float_batch(raw)
            view_q = ga.normalize_batch(ga.moco_augment(base, generator))
            view_k = ga.normalize_batch(ga.moco_augment(base, generator))

            with du.autocast_context(device, args.precision):
                queries = query_encoder(view_q)
                model.momentum_update()
                keys = model.encode_key(view_k)
            loss = model.loss(queries, keys)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.enqueue(du.all_gather_cat(keys, context))

            loss_sum += loss.detach()
            meter.add(args.batch_size * context.world_size)
            if context.is_main and (step + 1) % args.log_every == 0:
                print(f"epoca {epoch} paso {step + 1}/{len(batches)} perdida {loss.item():.4f}", flush=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        snapshot = meter.snapshot()
        mean_loss = float(du.all_reduce_sum(loss_sum, context)) / (len(batches) * context.world_size)
        record = {"epoch": epoch, "loss": mean_loss, "lr": learning_rate, **snapshot}
        history.append({"seconds": snapshot["seconds"], "images": snapshot["images"]})
        if context.is_main:
            logger.log(record)
            save_checkpoint(
                last_path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "history": history,
                },
            )
            save_backbone(
                output_dir / BACKBONE_FILE,
                model.encoder_q.trunk,
                {"resolution": resolution, "epochs_done": epoch + 1, "init_weights": args.init_weights},
            )

    summary = timing_summary(history)
    if context.is_main:
        (output_dir / "timing.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output_dir / "run_config.json").write_text(
            json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    du.barrier(context)
    du.cleanup(context)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
