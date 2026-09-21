"""Entrena la U-Net estudiante sobre pseudo-mascaras (sin mascaras reales ni etiquetas).

Pierde solo en los pixeles confiables (0 = fondo, 1 = lesion); el valor 255 de
la banda incierta se ignora. La mejor epoca se elige con la perdida de
pseudo-validacion (particion `val` de la cache) o, si se pide, la ultima epoca.
El umbral de inferencia es fijo en 0,5 y vive en la evaluacion.

Una GPU:
    python train_student.py --cache-dir salidas/cache224 \\
        --pseudo-masks salidas/pseudo_r0/pseudo_masks.npy --output-dir salidas/student_r1
Dos GPU:
    torchrun --nproc_per_node=2 train_student.py ...

Al terminar escribe `predictions.npy` (probabilidades) para re-etiquetar en la
siguiente ronda; use --no-predict para omitirlo. Con --predict-only se puede
inferir desde un `best.pt` existente sin reentrenar.
"""

from __future__ import annotations

import argparse
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
from student_data import SELECTION_CHOICES  # noqa: E402
from wandb_logging import add_tracking_arguments  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Cache creada por prepare_cache.py")
    parser.add_argument("--pseudo-masks", help="pseudo_masks.npy (0, 1, 255)")
    parser.add_argument("--output-dir", required=True, help="Carpeta de checkpoints y metricas")
    parser.add_argument("--epochs", type=int, default=10, help="Epocas de esta ronda")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano de lote POR GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Tasa de aprendizaje maxima")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Decaimiento de pesos")
    parser.add_argument("--optimizer", default="adam", choices=("adam", "adamw"), help="Optimizador")
    parser.add_argument("--warmup-epochs", type=int, default=1, help="Epocas de calentamiento lineal")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Peso del termino Dice suave")
    parser.add_argument("--base-channels", type=int, default=32, help="Canales base de la U-Net (32 = 7,77 M parametros)")
    parser.add_argument("--selection", default="best-val", choices=SELECTION_CHOICES, help="Checkpoint final: menor perdida de pseudo-validacion o ultima epoca")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"), help="Precision numerica")
    parser.add_argument("--seed", type=int, default=42, help="Semilla de aleatoriedad")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="Limita pasos por epoca (0 = completa)")
    parser.add_argument("--log-every", type=int, default=50, help="Pasos entre mensajes en pantalla")
    parser.add_argument("--no-resume", action="store_true", help="Ignora last.pt y empieza de cero")
    preload = parser.add_mutually_exclusive_group()
    preload.add_argument("--preload", action="store_true", help="Precarga images.npy en RAM")
    preload.add_argument("--no-preload", action="store_false", dest="preload", help="Conserva images.npy como memmap")
    parser.set_defaults(preload=False)
    parser.add_argument("--no-predict", action="store_true", help="No escribe predictions.npy al terminar")
    parser.add_argument(
        "--predict-only", action="store_true",
        help="Solo infiere desde output-dir/best.pt; no entrena ni modifica checkpoints",
    )
    parser.add_argument(
        "--overwrite-predictions", action="store_true",
        help="Permite reemplazar predictions.npy existente",
    )
    parser.add_argument("--force-cpu", action="store_true", help="Fuerza CPU (diagnostico)")
    add_tracking_arguments(parser)
    args = parser.parse_args(argv)
    if args.predict_only and args.no_predict:
        parser.error("--predict-only no se puede combinar con --no-predict")
    if not args.predict_only and args.pseudo_masks is None:
        parser.error("--pseudo-masks es obligatorio salvo con --predict-only")
    return args


def _validate_predict_only(args: argparse.Namespace) -> Path:
    """Validate the immutable checkpoint and safe prediction destination."""
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise SystemExit(f"error: la carpeta de salida no existe: {output_dir}")
    checkpoint = output_dir / "best.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"error: falta el checkpoint requerido: {checkpoint}")
    prediction_path = output_dir / "predictions.npy"
    if prediction_path.exists() and not args.overwrite_predictions:
        raise SystemExit(
            f"error: ya existe la salida {prediction_path}; use --overwrite-predictions para reemplazarla"
        )
    return prediction_path


def run(args: argparse.Namespace) -> dict:
    output_path = _validate_predict_only(args) if args.predict_only else None
    import dist_utils as du
    import student_engine as engine
    from cache_io import open_cache

    if args.predict_only:
        context = du.init_distributed(force_cpu=args.force_cpu)
        try:
            images = open_cache(Path(args.cache_dir) / "images.npy", preload=args.preload)
            path = engine.write_predictions(args, context, images, output_path)
            if context.is_main:
                print(f"predicciones escritas en {path}")
            return {"mode": "predict-only", "predictions": str(path)}
        finally:
            du.cleanup(context)

    context = du.init_distributed(force_cpu=args.force_cpu)
    summary = engine.train(args, context)
    if not args.no_predict:
        images = open_cache(Path(args.cache_dir) / "images.npy", preload=args.preload)
        path = engine.write_predictions(args, context, images, Path(args.output_dir) / engine.PREDICTIONS_FILE)
        if context.is_main:
            print(f"predicciones escritas en {path}")
    du.cleanup(context)
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
