"""Evalua un estudiante contra mascaras de referencia (solo para medir).

Este es el UNICO modulo del pipeline que abre mascaras manuales (ISIC 2018,
PH2, Dermofit). Se ejecuta despues del entrenamiento: no selecciona
checkpoints ni umbrales. El umbral de binarizacion es fijo en 0,5.

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

import eval_report  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402
from eval_report import summarize, write_per_image_csv, write_summary_json  # noqa: E402
from eval_tracking import track_evaluation  # noqa: E402

# Fixed by the thesis protocol: never tuned, never selected with ground truth.
DEFAULT_THRESHOLD = 0.5
BASE_METRICS = ("dice", "iou", "accuracy", "sensitivity", "specificity")
COVERAGE_METRICS = ("pred_coverage", "ref_coverage")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
OVERLAY_SEED = 0


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


# --------------------------------------------------------------------- I/O
def pair_images_and_masks(
    image_dir: Path,
    mask_dir: Path,
    mask_suffix: str = "_segmentation",
    mask_extension: str = ".png",
) -> list[tuple[str, Path, Path]]:
    """Match every image with ``<id><suffix><ext>`` in the mask folder."""
    images = sorted(
        p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    pairs, missing = [], []
    for image_path in images:
        mask_path = Path(mask_dir) / f"{image_path.stem}{mask_suffix}{mask_extension}"
        if mask_path.is_file():
            pairs.append((image_path.stem, image_path, mask_path))
        else:
            missing.append(image_path.stem)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images have no mask (first: {missing[:3]}); check --mask-suffix"
        )
    eval_report.reject_duplicate_ids([image_id for image_id, _, _ in pairs])
    return pairs


def load_mask(path: Path, resolution: int) -> np.ndarray:
    """Read a reference mask, resize with nearest neighbour and binarize."""
    from PIL import Image

    with Image.open(path) as image:
        resized = image.convert("L").resize((resolution, resolution), Image.NEAREST)
        return np.asarray(resized) > 127


def load_image(path: Path, resolution: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


# --------------------------------------------------------------- inference
def _load_student(checkpoint_path: Path, device):
    from ckpt_io import load_checkpoint
    from dermssl.torch_unet import UNet

    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint.get("config", {})
    model = UNet(base_channels=int(config.get("base_channels", 32)))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), int(config.get("resolution", 224))


def predict_masks(model, images: np.ndarray, device, batch_size: int) -> np.ndarray:
    """Probabilities ``(N, H, W)`` float32 for a uint8 image stack ``(N, H, W, 3)``."""
    import torch
    from gpu_augment import normalize_batch, to_float_batch

    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = torch.from_numpy(images[start : start + batch_size]).to(device)
            batch = normalize_batch(to_float_batch(chunk))
            outputs.append(torch.sigmoid(model(batch)).squeeze(1).float().cpu().numpy())
    return np.concatenate(outputs)


def _score_pairs(model, device, pairs, resolution: int, args) -> list[dict]:
    """One row per image; reference masks are read here and nowhere else."""
    rows = []
    for start in range(0, len(pairs), args.batch_size):
        chunk = pairs[start : start + args.batch_size]
        images = np.stack([load_image(image_path, resolution) for _, image_path, _ in chunk])
        probabilities = predict_masks(model, images, device, len(chunk))
        for (image_id, _, mask_path), probability in zip(chunk, probabilities, strict=True):
            prediction = binarize(probability)
            truth = load_mask(mask_path, resolution)
            row = {"image_id": image_id, **compute_binary_metrics(prediction, truth)}
            if args.hausdorff:
                row["hd95"] = hausdorff95(prediction, truth)
            rows.append(row)
    return rows


def _overlay_records(model, device, sources, identifiers, resolution, dice_by_id) -> list[dict]:
    """Re-run the model on the few selected images to paint the contact sheets."""
    records = []
    for image_id in identifiers:
        image_path, mask_path = sources[image_id]
        image = load_image(image_path, resolution)
        probability = predict_masks(model, image[None], device, 1)[0]
        records.append(
            {
                "image_id": image_id,
                "dice": dice_by_id[image_id],
                "image": image,
                "reference": load_mask(mask_path, resolution),
                "prediction": binarize(probability),
            }
        )
    return records


def _write_overlays(args, model, device, pairs, rows, resolution) -> list[Path]:
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
        records = _overlay_records(model, device, sources, identifiers, resolution, dice_by_id)
        paths.append(eval_report.write_overlay_grid(overlay_dir / f"{name}.png", records))
    return paths


def evaluate(args: argparse.Namespace) -> dict:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint_resolution = _load_student(Path(args.checkpoint), device)
    resolution = args.eval_resolution or checkpoint_resolution
    pairs = pair_images_and_masks(
        Path(args.image_dir), Path(args.mask_dir), args.mask_suffix, args.mask_extension
    )
    columns = metric_columns(args.hausdorff)
    rows = _score_pairs(model, device, pairs, resolution, args)

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
        "checkpoint": str(args.checkpoint),
        "image_dir": str(args.image_dir),
        "note": "reference masks were read only to compute these metrics",
    }
    write_summary_json(output_dir / "summary.json", summary, meta, strata)
    overlays = _write_overlays(args, model, device, pairs, rows, resolution)
    track_evaluation(args, summary, strata, rows, overlays)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_spanish_parser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint del estudiante (best.pt)")
    parser.add_argument("--image-dir", required=True, help="Carpeta con las imagenes de prueba")
    parser.add_argument("--mask-dir", required=True, help="Carpeta con las mascaras de referencia")
    parser.add_argument("--output-dir", required=True, help="Carpeta de salida (CSV y JSON)")
    parser.add_argument("--mask-suffix", default="_segmentation", help="Sufijo del nombre de la mascara")
    parser.add_argument("--mask-extension", default=".png", help="Extension de las mascaras")
    parser.add_argument("--eval-resolution", type=int, default=0, help="Resolucion de evaluacion; 0 usa la del entrenamiento")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamano de lote para la inferencia")
    parser.add_argument("--hausdorff", action="store_true", help="Calcula tambien Hausdorff-95 (mas lento)")
    parser.add_argument("--eval-name", default="referencia", help="Nombre del conjunto; prefijo de las metricas")
    parser.add_argument("--fst-manifest", default="", help="Manifiesto FST (CSV image_id,fst_bucket o el JSON de fst_pseudo_manifest)")
    parser.add_argument("--fst-min-n", type=int, default=30, help="Minimo de imagenes por bucket; debajo se marca SKIPPED")
    parser.add_argument("--overlay-dir", default="", help="Carpeta para las rejillas PNG de superposicion")
    parser.add_argument("--overlay-count", type=int, default=12, help="Imagenes por grupo (peores, mejores, aleatorias)")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", ""), help="Proyecto de Weights & Biases (vacio = sin seguimiento)")
    parser.add_argument("--wandb-run-name", default="", help="Nombre de la corrida en Weights & Biases")
    args = parser.parse_args(argv)
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
