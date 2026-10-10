"""S1: verified qualitative segmentation cases from a pod-side manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402
from qualitative_figure_inputs import read_binary_mask, read_manifest, read_rgb, resolve_case_path, sha256_file  # noqa: E402

STEM = "fig_s1_segmentation_cases"
DEFAULT_MANIFEST = REPO / "resultados/insumos_pod/S1/manifest.json"
DEFAULT_OUTDIR = REPO / "resultados/figuras/thesis_s1_t5_qualitative_20261009/S1"
LEVEL_ORDER = ("alto", "medio", "bajo")
FILES = ("original.png", "reference_mask.png", "student_r3_mask.png", "s4_a2_mask.png")
COLUMNS = ("Original", "Referencia", "student_r3", "S4-A2", "Contornos")
CONTOURS = (
    ("Referencia", style.COLORS["accent"], "solid"),
    ("student_r3", style.COLORS["text"], "dashed"),
    ("S4-A2", style.COLORS["frame"], "dotted"),
)
CONTOUR_MASK_FILES = {
    "Referencia": "reference_mask.png",
    "student_r3": "student_r3_mask.png",
    "S4-A2": "s4_a2_mask.png",
}
DICE_TOLERANCE = 1e-4
CAPTION = "Casos de segmentación en niveles alto, medio y bajo, con IDs y resolución compartidos entre columnas."
NOTE = (
    "Los criterios percentiles se transcriben del manifiesto; no se eligieron casos por inspección visual. "
    "Máscaras binarias: negro = fondo, blanco = lesión. El panel Contornos superpone referencia, student_r3 y S4-A2. "
    "Dice se recalcula desde las máscaras y debe coincidir con el manifiesto dentro de 1e-4."
)


def _valid_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"S1 case requires non-empty {field}")
    return value.strip()


def _dice(reference, prediction) -> float:
    denominator = int(reference.sum() + prediction.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int((reference.astype(bool) & prediction.astype(bool)).sum()) / denominator


def _metric(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"S1 {field} must be a finite Dice value")
    if not 0 <= value <= 1:
        raise ValueError(f"S1 {field} must lie in [0, 1]")
    return float(value)


def _relative_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def load_cases(manifest_path: str | Path = DEFAULT_MANIFEST) -> list[dict]:
    root, entries, manifest = read_manifest(manifest_path)
    cases = []
    seen = set()
    resolutions = set()
    for entry in entries:
        image_id = _valid_text(entry.get("image_id"), "image_id")
        dataset = _valid_text(entry.get("dataset"), "dataset")
        level = _valid_text(entry.get("nivel"), "nivel")
        criterion = _valid_text(entry.get("criterio"), "criterio")
        if level not in LEVEL_ORDER:
            raise ValueError(f"S1 nivel must be one of {LEVEL_ORDER}: {level}")
        identity = (dataset, image_id)
        if identity in seen:
            raise ValueError(f"duplicate S1 case ID: {dataset}/{image_id}")
        seen.add(identity)

        paths = {filename: resolve_case_path(root, entry, filename) for filename in FILES}
        if any(path.suffix.casefold() != ".png" for path in paths.values()):
            raise ValueError(f"S1 case files must be PNG: {image_id}")
        image = read_rgb(paths["original.png"])
        masks = {name: read_binary_mask(paths[name]) for name in FILES[1:]}
        resolution = image.shape[:2]
        if any(mask.shape != resolution for mask in masks.values()):
            raise ValueError(f"S1 image/mask resolution mismatch: {image_id}")
        resolutions.add(resolution)
        dice_values = {
            "student_r3": _metric(entry.get("dice_student_r3"), "dice_student_r3"),
            "S4-A2": _metric(entry.get("dice_s4a2"), "dice_s4a2"),
        }
        expected = {
            "student_r3": _dice(masks["reference_mask.png"], masks["student_r3_mask.png"]),
            "S4-A2": _dice(masks["reference_mask.png"], masks["s4_a2_mask.png"]),
        }
        for model, value in dice_values.items():
            if abs(value - expected[model]) > DICE_TOLERANCE:
                raise ValueError(f"Dice mismatch for {dataset}/{image_id} {model}: {expected[model]:.6f} != {value:.6f}")
        cases.append({
            "image_id": image_id,
            "dataset": dataset,
            "nivel": level,
            "criterio": criterion,
            "dice": dice_values,
            "resolution": resolution,
            "image": image,
            "masks": masks,
            "files": {name: {"path": _relative_path(root, path), "sha256": sha256_file(path)}
                      for name, path in paths.items()},
        })

    levels = {case["nivel"] for case in cases}
    if levels != set(LEVEL_ORDER):
        raise ValueError(f"S1 manifest must include all levels alto, medio, bajo; got {sorted(levels)}")
    if len(resolutions) != 1:
        raise ValueError(f"S1 cases must share one image resolution: {sorted(resolutions)}")
    cases.sort(key=lambda case: (LEVEL_ORDER.index(case["nivel"]), case["dataset"], case["image_id"]))
    return cases


def _draw_contour(axis, mask, color: str, linestyle: str) -> None:
    if mask.any() and not mask.all():
        axis.contour(mask.astype(float), levels=[0.5], colors=[color], linestyles=[linestyle], linewidths=1.1)


def draw(cases: list[dict]):
    height = max(4.8, 2.2 + 1.5 * len(cases))
    with style.context():
        fig, axes = plt.subplots(len(cases), len(COLUMNS), squeeze=False,
                                 figsize=(style.FIGURE_WIDTH_IN, height),
                                 facecolor=style.COLORS["background"])
        row_labels = []
        for column, title in enumerate(COLUMNS):
            axes[0, column].set_title(title, fontsize=style.LABEL_PT, pad=8, color=style.COLORS["text"])
        for row, case in enumerate(cases):
            views = [case["image"], case["masks"]["reference_mask.png"],
                     case["masks"]["student_r3_mask.png"], case["masks"]["s4_a2_mask.png"]]
            for axis, view in zip(axes[row, :4], views, strict=True):
                if view.ndim == 2:
                    axis.imshow(view, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                else:
                    axis.imshow(view, interpolation="nearest")
            contour_axis = axes[row, 4]
            contour_axis.imshow(case["image"], interpolation="nearest")
            for mask_name, color, linestyle in CONTOURS:
                mask = case["masks"][CONTOUR_MASK_FILES[mask_name]]
                _draw_contour(contour_axis, mask, color, linestyle)
            row_label = (
                f"{case['nivel']} · {case['criterio']}\n{case['dataset']}\n{case['image_id']}\n"
                f"r3 Dice {style.fmt_es(case['dice']['student_r3'], 3)}\n"
                f"S4-A2 Dice {style.fmt_es(case['dice']['S4-A2'], 3)}"
            )
            row_labels.append((row, row_label))
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_facecolor(style.COLORS["background"])
                for spine in axis.spines.values():
                    spine.set_color(style.COLORS["frame"])
                    spine.set_linewidth(style.LINE_WIDTHS["frame"])
        handles = [Line2D([], [], color=color, linestyle=linestyle, linewidth=1.2, label=label)
                   for label, color, linestyle in CONTOURS]
        fig.text(0.25, 0.97, "Segmentación cualitativa · casos alto/medio/bajo",
                 ha="left", va="top", fontsize=style.PANEL_TITLE_PT, color=style.COLORS["text"])
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.67, 0.035),
                   frameon=False, ncol=3, fontsize=8.5)
        fig.subplots_adjust(left=0.24, right=0.99, top=0.90, bottom=0.13, wspace=0.025, hspace=0.22)
        for row, label in row_labels:
            position = axes[row, 0].get_position()
            fig.text(0.012, position.y0 + position.height / 2, label,
                     ha="left", va="center", fontsize=8, color=style.COLORS["text"])
    return fig


def _traceability(manifest_path: Path, cases: list[dict]) -> dict:
    return {
        "figure_id": "S1",
        "manifest": {"path": manifest_path.name, "sha256": sha256_file(manifest_path)},
        "resolution_px": list(cases[0]["resolution"]),
        "levels": list(LEVEL_ORDER),
        "dice_tolerance": DICE_TOLERANCE,
        "checks": ["all three levels present", "same ID and resolution per case", "masks binary", "Dice recomputed from masks"],
        "contours": [{"label": label, "color": color, "linestyle": linestyle}
                     for label, color, linestyle in CONTOURS],
        "script": {"path": "scripts/figures/fig_s1_cases.py",
                   "sha256": sha256_file(REPO / "scripts/figures/fig_s1_cases.py")},
        "cases": [{key: case[key] for key in ("image_id", "dataset", "nivel", "criterio", "dice", "files")}
                  for case in cases],
    }


def build(manifest_path: str | Path | None = None, outdir: str | Path | None = None) -> list[Path]:
    manifest = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST
    cases = load_cases(manifest)
    target = Path(outdir) if outdir is not None else DEFAULT_OUTDIR
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"{STEM}{suffix}" for suffix in suffixes if (target / f"{STEM}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)
    fig = draw(cases)
    try:
        paths = style.save_figure(fig, STEM, target)
    finally:
        plt.close(fig)
    (target / f"{STEM}.caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    (target / f"{STEM}.note.txt").write_text(NOTE + "\n", encoding="utf-8")
    (target / f"{STEM}.traceability.json").write_text(
        json.dumps(_traceability(manifest.resolve(), cases), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    for path in build(args.manifest, args.outdir):
        print(path)
