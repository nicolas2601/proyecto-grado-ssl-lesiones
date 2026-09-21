"""Evalua predicciones contra mascaras de referencia (solo para medir).

Junto con ``eval_inputs`` es el UNICO lugar del pipeline que abre mascaras
manuales (ISIC 2018, PH2, Dermofit). Se ejecuta despues del entrenamiento: no
selecciona checkpoints ni umbrales. El umbral de binarizacion es fijo en 0,5.

Hay dos fuentes de prediccion, mutuamente excluyentes:

* ``--checkpoint``: se corre el estudiante sobre las imagenes;
* ``--pred-mask-dir``: se leen mascaras PNG ya calculadas por otro metodo
  (baselines publicados, otro pipeline). El reporte es identico.

Ejemplo:
    python plan-segmentacion/comun/evaluacion/evaluate_segmentation.py \\
        --checkpoint salidas/piloto/student_round1/best.pt \\
        --image-dir /datos/isic2018/images --mask-dir /datos/isic2018/masks \\
        --output-dir salidas/piloto/evaluacion_isic2018
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_PLAN_ROOT = Path(__file__).resolve().parents[2]
for _extra in (_PLAN_ROOT.parent / "src", _PLAN_ROOT / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import eval_inputs  # noqa: E402
import eval_report  # noqa: E402
import eval_student  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402
from eval_inputs import load_binary_mask as load_mask  # noqa: E402
from eval_inputs import load_image, pair_images_and_masks  # noqa: E402
from eval_report import summarize, write_per_image_csv, write_summary_json  # noqa: E402
from eval_tracking import track_evaluation  # noqa: E402

# Fixed by the thesis protocol: never tuned, never selected with ground truth.
DEFAULT_THRESHOLD = 0.5
BASE_METRICS = ("dice", "iou", "accuracy", "sensitivity", "specificity")
COVERAGE_METRICS = ("pred_coverage", "ref_coverage")
OVERLAY_SEED = 0
# External masks carry no training resolution, so the comparison grid is fixed here.
EXTERNAL_RESOLUTION = 224


# ------------------------------------------------------------------ metrics
def binarize(probabilities: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Strictly greater than the threshold is lesion."""
    return np.asarray(probabilities) > threshold


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.nan


def compute_binary_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict:
    """Dice, IoU, pixel accuracy, sensitivity and specificity of two boolean masks.

    When both masks are empty, Dice and IoU are 1 (perfect agreement) and
    sensitivity is undefined (NaN), so summaries can skip it.
    """
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if prediction.shape != truth.shape:
        raise ValueError(f"shape mismatch: {prediction.shape} vs {truth.shape}")
    true_positive = float(np.logical_and(prediction, truth).sum())
    false_positive = float(np.logical_and(prediction, ~truth).sum())
    false_negative = float(np.logical_and(~prediction, truth).sum())
    true_negative = float(np.logical_and(~prediction, ~truth).sum())

    union = true_positive + false_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    both_empty = union == 0
    return {
        "dice": 1.0 if both_empty else 2 * true_positive / dice_denominator,
        "iou": 1.0 if both_empty else true_positive / union,
        "accuracy": (true_positive + true_negative) / prediction.size,
        "sensitivity": _safe_ratio(true_positive, true_positive + false_negative),
        "specificity": _safe_ratio(true_negative, true_negative + false_positive),
        "pred_coverage": (true_positive + false_positive) / prediction.size,
        "ref_coverage": (true_positive + false_negative) / prediction.size,
    }


def hausdorff95(prediction: np.ndarray, truth: np.ndarray) -> float:
    """95th percentile symmetric surface distance in pixels; NaN if a mask is empty."""
    from scipy import ndimage  # guarded: optional metric

    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if not prediction.any() or not truth.any():
        return math.nan

    def surface(mask: np.ndarray) -> np.ndarray:
        return mask & ~ndimage.binary_erosion(mask, border_value=0)

    distance_to_truth = ndimage.distance_transform_edt(~truth)
    distance_to_prediction = ndimage.distance_transform_edt(~prediction)
    distances = np.concatenate(
        [distance_to_truth[surface(prediction)], distance_to_prediction[surface(truth)]]
    )
    return float(np.percentile(distances, 95))


def metric_columns(hausdorff: bool) -> list[str]:
    """Per-image CSV columns, in report order."""
    return [*BASE_METRICS, *COVERAGE_METRICS] + (["hd95"] if hausdorff else [])


def _row(image_id: str, prediction: np.ndarray, truth: np.ndarray, hausdorff: bool) -> dict:
    """One per-image row; reference masks are compared here and nowhere else."""
    row = {"image_id": image_id, **compute_binary_metrics(prediction, truth)}
    if hausdorff:
        row["hd95"] = hausdorff95(prediction, truth)
    return row


# ------------------------------------------------------------------ report
def _overlay_records(sources, identifiers, resolution, dice_by_id, predict_for) -> list[dict]:
    """Rebuild the few selected predictions to paint the contact sheets."""
    records = []
    for image_id in identifiers:
        image_path, mask_path = sources[image_id]
        image = load_image(image_path, resolution)
        records.append(
            {
                "image_id": image_id,
                "dice": dice_by_id[image_id],
                "image": image,
                "reference": load_mask(mask_path, resolution),
                "prediction": predict_for(image_id, image),
            }
        )
    return records


def _write_overlays(args, pairs, rows, resolution, predict_for) -> list[Path]:
    if not args.overlay_dir or not rows:
        return []
    overlay_dir = Path(args.overlay_dir)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    sources = {image_id: (image_path, mask_path) for image_id, image_path, mask_path in pairs}
    dice_by_id = {row["image_id"]: row["dice"] for row in rows}
    groups = eval_report.select_overlay_groups(rows, args.overlay_count, seed=OVERLAY_SEED)
    paths = []
    for name, identifiers in groups.items():
        if not identifiers:
            continue
        records = _overlay_records(sources, identifiers, resolution, dice_by_id, predict_for)
        paths.append(eval_report.write_overlay_grid(overlay_dir / f"{name}.png", records))
    return paths


def _write_report(args, pairs, rows, resolution: int, meta_extra: dict, predict_for) -> dict:
    """CSV, JSON, overlays and tracking; identical for both prediction sources."""
    columns = metric_columns(args.hausdorff)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_per_image_csv(output_dir / "per_image.csv", rows, columns)
    summary = summarize(rows, columns)
    strata = eval_report.strata_from_args(args, rows)
    meta = {
        "n_images": len(rows),
        "eval_name": args.eval_name,
        "threshold": DEFAULT_THRESHOLD,
        "resolution": resolution,
        "image_dir": str(args.image_dir),
        "note": "reference masks were read only to compute these metrics",
        **meta_extra,
    }
    write_summary_json(output_dir / "summary.json", summary, meta, strata)
    overlays = _write_overlays(args, pairs, rows, resolution, predict_for)
    track_evaluation(args, summary, strata, rows, overlays)
    return summary


def _reference_pairs(args) -> list[tuple[str, Path, Path]]:
    return pair_images_and_masks(
        Path(args.image_dir), Path(args.mask_dir), args.mask_suffix, args.mask_extension
    )


def evaluate_external(args: argparse.Namespace) -> dict:
    """Score masks produced elsewhere: no model, no torch, same report."""
    resolution = args.eval_resolution or EXTERNAL_RESOLUTION
    pairs = _reference_pairs(args)
    predictions = eval_inputs.pair_predictions(
        pairs, Path(args.pred_mask_dir), args.pred_mask_suffix, args.pred_mask_extension
    )

    def predict_for(image_id, _image):
        return load_mask(predictions[image_id], resolution)

    rows = [
        _row(
            image_id,
            load_mask(predictions[image_id], resolution),
            load_mask(mask_path, resolution),
            args.hausdorff,
        )
        for image_id, _, mask_path in pairs
    ]
    meta = {"predictions": "external", "pred_mask_dir": str(args.pred_mask_dir)}
    return _write_report(args, pairs, rows, resolution, meta, predict_for)


def evaluate_checkpoint(args: argparse.Namespace) -> dict:
    """Run the student over the evaluation set and report its masks."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_resolution = eval_student.load_student(Path(args.checkpoint), device)
    resolution = args.eval_resolution or checkpoint_resolution
    pairs = _reference_pairs(args)

    def make_row(image_id, probability, mask_path):
        truth = load_mask(mask_path, resolution)
        return _row(image_id, binarize(probability), truth, args.hausdorff)

    def predict_for(_image_id, image):
        return binarize(eval_student.predict_masks(model, image[None], device, 1, tta=args.tta)[0])

    rows = eval_student.score_pairs(
        model, device, pairs, resolution, args.batch_size, make_row, tta=args.tta
    )
    meta = {"predictions": "checkpoint", "checkpoint": str(args.checkpoint), "tta": bool(args.tta)}
    return _write_report(args, pairs, rows, resolution, meta, predict_for)


def evaluate(args: argparse.Namespace) -> dict:
    """Dispatch on the prediction source chosen at the command line."""
    if args.pred_mask_dir:
        return evaluate_external(args)
    return evaluate_checkpoint(args)


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", default="", help="Checkpoint del estudiante (best.pt)")
    source.add_argument("--pred-mask-dir", default="", help="Carpeta con mascaras PNG ya predichas por otro metodo")
    parser.add_argument("--pred-mask-suffix", default="", help="Sufijo del nombre de la mascara predicha")
    parser.add_argument("--pred-mask-extension", default=".png", help="Extension de las mascaras predichas")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    _add_source_arguments(parser)
    parser.add_argument("--image-dir", required=True, help="Carpeta con las imagenes de prueba")
    parser.add_argument("--mask-dir", required=True, help="Carpeta con las mascaras de referencia")
    parser.add_argument("--output-dir", required=True, help="Carpeta de salida (CSV y JSON)")
    parser.add_argument("--mask-suffix", default="_segmentation", help="Sufijo del nombre de la mascara")
    parser.add_argument("--mask-extension", default=".png", help="Extension de las mascaras")
    parser.add_argument("--eval-resolution", type=int, default=0, help="Resolucion de evaluacion; 0 usa la del entrenamiento (o 224 con --pred-mask-dir)")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano de lote para la inferencia")
    parser.add_argument("--hausdorff", action="store_true", help="Calcula tambien Hausdorff-95 (mas lento)")
    parser.add_argument("--tta", action="store_true", help="Promedia la prediccion sobre volteos horizontal y vertical (solo inferencia)")
    parser.add_argument("--eval-name", default="referencia", help="Nombre del conjunto; prefijo de las metricas")
    parser.add_argument("--fst-manifest", default="", help="Manifiesto FST (CSV image_id,fst_bucket o el JSON de fst_pseudo_manifest)")
    parser.add_argument("--fst-min-n", type=int, default=30, help="Minimo de imagenes por bucket; debajo se marca SKIPPED")
    parser.add_argument("--overlay-dir", default="", help="Carpeta para las rejillas PNG de superposicion")
    parser.add_argument("--overlay-count", type=int, default=12, help="Imagenes por grupo (peores, mejores, aleatorias)")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""), help="Proyecto de Weights & Biases (vacio = sin seguimiento)")
    parser.add_argument("--wandb-run-name", default="", help="Nombre de la corrida en Weights & Biases")
    args = parser.parse_args(argv)
    if args.pred_mask_dir and not args.eval_resolution:
        args.eval_resolution = EXTERNAL_RESOLUTION
    if args.fst_manifest:
        try:
            eval_report.validate_fst_manifest(Path(args.fst_manifest))
        except ValueError as error:
            parser.error(str(error))
    return args


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
