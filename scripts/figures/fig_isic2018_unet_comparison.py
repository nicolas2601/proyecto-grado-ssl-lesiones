"""Qualitative ISIC 2018: real U-Net predictions versus matching reference masks.

Source arrays are reinferred from the final student_r3 checkpoint in original
32-image batches. The extractor checks every case against committed per-image
Dice and stores checkpoint/script hashes in the NPZ provenance sidecar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beautiful as bf  # noqa: E402
import fig3_masks as masks  # noqa: E402
from fig3_extract_cases import DICE_TOLERANCE, EVAL_BATCH_SIZE, file_sha256  # noqa: E402

STEM = "fig_isic2018_unet_vs_reference"
NPZ = REPO / "resultados/figuras/fuentes/fig3_casos.npz"
REFERENCE_COLOR, PREDICTION_COLOR = "#6F8D73", "#C25747"
BACKGROUND, PANEL, INK, MUTED = "#FBFBFA", "#FAF9F7", "#302C29", "#645D58"
COLUMN_TITLES = ("Imagen original", "Máscara de referencia", "Predicción U-Net", "Contornos sobre imagen")


def verified_case(case: dict, arrays: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if case["dataset"] != "isic2018":
        raise ValueError(f"Not an ISIC 2018 case: {case['dataset']}")
    prefix = f"isic2018__{case['image_id']}"
    keys = [f"{prefix}__{suffix}" for suffix in ("image", "reference", "student_r3")]
    for key in keys:
        if key not in arrays:
            raise ValueError(f"Missing case array: {key}")
    image, truth, prediction = (np.asarray(arrays[key]) for key in keys)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Shape mismatch for {case['image_id']}")
    if truth.shape != image.shape[:2] or prediction.shape != image.shape[:2]:
        raise ValueError(f"Shape mismatch for {case['image_id']}")
    if not np.isin(truth, (0, 1)).all() or not np.isin(prediction, (0, 1)).all():
        raise ValueError(f"Masks must be binary: {case['image_id']}")
    overlap = np.count_nonzero((truth > 0) & (prediction > 0))
    denominator = np.count_nonzero(truth) + np.count_nonzero(prediction)
    dice = 1.0 if denominator == 0 else 2.0 * overlap / denominator
    if abs(dice - case["dice"]["student_r3"]) > DICE_TOLERANCE:
        raise ValueError(f"Dice mismatch for {case['image_id']}: {dice:.6f}")
    return image, truth, prediction


def _load_verified(repo: Path, npz_path: Path) -> tuple[list[dict], dict]:
    provenance_path = npz_path.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance["max_abs_dice_difference_vs_csv"] > DICE_TOLERANCE:
        raise ValueError("Extractor Dice gate failed")
    if provenance["evaluation_batch_size"] != EVAL_BATCH_SIZE:
        raise ValueError("Extractor did not reuse the original evaluation batch")
    checkpoint = provenance["checkpoints"]["student_r3"]
    if file_sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
        raise ValueError("student_r3 checkpoint hash mismatch")
    cases = masks.select_cases(repo)
    if provenance["cases"] != cases:
        raise ValueError("Cases differ from the committed evaluation CSV")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    selected = [case for case in cases if case["dataset"] == "isic2018"]
    for case in selected:
        verified_case(case, arrays)
    return selected, arrays


def _draw_case(row_axes, case: dict, arrays: dict) -> None:
    image, truth, prediction = verified_case(case, arrays)
    for ax, panel in zip(row_axes[:3], (image, truth, prediction), strict=True):
        ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None, vmin=0 if panel.ndim == 2 else None,
                  vmax=1 if panel.ndim == 2 else None)
    row_axes[3].imshow(image)
    for contour, color in ((truth, REFERENCE_COLOR), (prediction, PREDICTION_COLOR)):
        row_axes[3].contour(contour, levels=[0.5], colors=[color], linewidths=1.7)
    label = f"p{case['percentile']} · {case['image_id']}\nDice U-Net = {bf.fmt_es(case['dice']['student_r3'], 3)}"
    row_axes[0].set_ylabel(label, fontsize=10, color=INK, rotation=0, ha="right", va="center", labelpad=16)
    for ax in row_axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_color("#AAA39E")
            spine.set_linewidth(0.7)


def build(repo: Path = REPO, npz_path: Path = NPZ, outdir: Path | None = None) -> list[Path]:
    cases, arrays = _load_verified(Path(repo), Path(npz_path))
    bf.apply_style(width=15)
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none"})
    fig, axes = plt.subplots(len(cases), len(COLUMN_TITLES), figsize=(15, 8.3), facecolor=BACKGROUND)
    for ax, title in zip(axes[0], COLUMN_TITLES, strict=True):
        ax.set_title(title, fontsize=12, pad=12, color=INK)
    for row_axes, case in zip(axes, cases, strict=True):
        _draw_case(row_axes, case, arrays)
    handles = [Line2D([0], [0], color=color, lw=2, label=label) for color, label in
               ((REFERENCE_COLOR, "Referencia ISIC 2018"), (PREDICTION_COLOR, "U-Net student_r3"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.075), ncol=2,
               frameon=False, fontsize=11)
    fig.text(0.08, 0.018, "Mismo ID y preprocesamiento (224 px); Dice U-Net verificado contra CSV. "
             "p90/p50/p10: percentiles del Dice medio U-Net/A2, no clínicos.\n"
             "Solo se muestra U-Net student_r3; tres ejemplos de 2.594 imágenes ISIC 2018.",
             fontsize=9.5, color=MUTED)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.88, bottom=0.15, wspace=0.07, hspace=0.30)
    target = Path(outdir) if outdir else Path(repo) / "resultados/figuras"
    paths = bf.save_all(fig, STEM, target)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    for path in build():
        print(path)
