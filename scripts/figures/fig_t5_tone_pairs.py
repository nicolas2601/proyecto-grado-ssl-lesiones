"""T5: verified original/transformed tone pairs with protected lesion masks."""

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
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402
from dermssl.circle import FST_SHIFT_L_STEP, MAX_FST_SHIFT  # noqa: E402
from dermssl.mask_manifest import mask_sha256 as canonical_mask_sha256  # noqa: E402
from dermssl.fitzpatrick import MAX_MASK_AREA_FRACTION, MIN_MASK_AREA_FRACTION  # noqa: E402
from qualitative_figure_inputs import (  # noqa: E402
    read_binary_mask,
    read_manifest,
    read_rgb,
    resolve_case_path,
    sha256_file,
    verify_sha256,
)

STEM = "fig_t5_tone_pairs"
DEFAULT_MANIFEST = REPO / "resultados/insumos_pod/T5/manifest.json"
DEFAULT_OUTDIR = REPO / "resultados/figuras/thesis_s1_t5_qualitative_20261009/T5"
FILES = ("original.png", "transformed.png", "mask.png")
COLUMNS = ("Original", "Transformada", "Máscara")
CAPTION = "Pares original/transformada CIRCLe con la misma máscara de lesión y el tercil de luminancia registrado."
NOTE = (
    "El campo shift_buckets es s. El cambio L_shift = −8s se deriva con FST_SHIFT_L_STEP=8 L* por nivel "
    "y se muestra antes del clipping L*′=clip(L*−8s,0,100); no es un delta medido de píxeles. "
    "La máscara usa semántica true=lesion: la transformación actúa fuera de la lesión, que queda protegida. "
    "Los hashes source_sha256, transformed_sha256 y mask_sha256 se verifican contra los archivos. "
    "El tercil se calculó con la media de L del original sobre una muestra de 605 imágenes candidatas del servidor; "
    "no es el tercil de luminancia del manifiesto de tono del conjunto completo."
)


def derive_l_shift(shift_buckets: int) -> float:
    if not isinstance(shift_buckets, int) or isinstance(shift_buckets, bool):
        raise ValueError("shift_buckets must be an integer s")
    if not -MAX_FST_SHIFT <= shift_buckets <= MAX_FST_SHIFT:
        raise ValueError(f"shift_buckets s must lie in [-{MAX_FST_SHIFT}, {MAX_FST_SHIFT}]")
    return -FST_SHIFT_L_STEP * shift_buckets


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"T5 case requires non-empty {field}")
    return value.strip()


def _verify_canonical_mask(mask: np.ndarray, expected: str) -> str:
    """The manifest carries the project's canonical mask hash (shape + true=lesion + boolean bytes), not the PNG file hash."""
    actual = canonical_mask_sha256(mask == 1)
    if expected != actual:
        raise ValueError(f"mask canonical SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _load_pair(root: Path, entry: dict) -> dict:
    image_id = _text(entry.get("image_id"), "image_id")
    tercil = _text(entry.get("tercil"), "tercil")
    semantics = entry.get("mask_semantics", "true=lesion")
    if semantics != "true=lesion":
        raise ValueError(f"T5 mask_semantics must be true=lesion for {image_id}")
    shift = entry.get("shift_buckets", entry.get("s"))
    if "s" in entry and entry.get("s") != shift:
        raise ValueError(f"T5 s and shift_buckets disagree for {image_id}")
    l_shift = derive_l_shift(shift)
    if "L_shift" in entry:
        supplied = entry["L_shift"]
        if isinstance(supplied, bool) or not isinstance(supplied, (int, float)) or not math.isclose(
            float(supplied), l_shift, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(f"T5 L_shift mismatch for {image_id}: expected derived value {l_shift:g}")

    paths = {filename: resolve_case_path(root, entry, filename) for filename in FILES}
    if any(path.suffix.casefold() != ".png" for path in paths.values()):
        raise ValueError(f"T5 case files must be PNG: {image_id}")
    original = read_rgb(paths["original.png"])
    transformed = read_rgb(paths["transformed.png"])
    mask = read_binary_mask(paths["mask.png"])
    if original.shape != transformed.shape or mask.shape != original.shape[:2]:
        raise ValueError(f"T5 image/mask resolution mismatch: {image_id}")
    mask_fraction = float(mask.mean())
    if not MIN_MASK_AREA_FRACTION <= mask_fraction <= MAX_MASK_AREA_FRACTION:
        raise ValueError(f"T5 lesion mask is degenerate for {image_id}: area={mask_fraction:.4f}")
    if not np.array_equal(original[mask == 1], transformed[mask == 1]):
        raise ValueError(f"T5 lesion pixels are not preserved for {image_id}")

    hashes = {
        "source_sha256": verify_sha256(paths["original.png"], entry.get("source_sha256"), "source"),
        "transformed_sha256": verify_sha256(paths["transformed.png"], entry.get("transformed_sha256"), "transformed"),
        "mask_sha256": _verify_canonical_mask(mask, entry.get("mask_sha256")),
    }
    return {
        "image_id": image_id,
        "dataset": entry.get("dataset"),
        "tercil": tercil,
        "shift_buckets": shift,
        "L_shift": l_shift,
        "mask_semantics": semantics,
        "resolution": original.shape[:2],
        "original": original,
        "transformed": transformed,
        "mask": mask,
        "files": {filename: {"path": paths[filename].relative_to(root).as_posix(),
                             "sha256": hashes[field]}
                  for filename, field in zip(FILES, hashes, strict=True)},
    }


def load_pairs(manifest_path: str | Path = DEFAULT_MANIFEST) -> list[dict]:
    root, entries, _ = read_manifest(manifest_path)
    pairs = [_load_pair(root, entry) for entry in entries]
    image_ids = [pair["image_id"] for pair in pairs]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("T5 manifest contains duplicate image_id values")
    resolutions = {pair["resolution"] for pair in pairs}
    if len(resolutions) != 1:
        raise ValueError(f"T5 pairs must share one resolution: {sorted(resolutions)}")
    return sorted(pairs, key=lambda pair: (pair["tercil"], pair["image_id"]))


def _draw_contour(axis, mask: np.ndarray) -> None:
    if mask.any() and not mask.all():
        axis.contour(mask.astype(float), levels=[0.5], colors=[style.COLORS["accent"]], linewidths=1.2)


def draw(pairs: list[dict]):
    height = max(4.8, 2.2 + 1.65 * len(pairs))
    with style.context():
        fig, axes = plt.subplots(len(pairs), 3, squeeze=False,
                                 figsize=(style.FIGURE_WIDTH_IN, height),
                                 facecolor=style.COLORS["background"])
        for column, title in enumerate(COLUMNS):
            axes[0, column].set_title(title, fontsize=style.LABEL_PT, pad=8, color=style.COLORS["text"])
        row_labels = []
        for row, pair in enumerate(pairs):
            original_axis, transformed_axis, mask_axis = axes[row]
            original_axis.imshow(pair["original"], interpolation="nearest")
            transformed_axis.imshow(pair["transformed"], interpolation="nearest")
            _draw_contour(original_axis, pair["mask"])
            _draw_contour(transformed_axis, pair["mask"])
            mask_axis.imshow(pair["mask"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            dataset = f"{pair['dataset']} · " if pair["dataset"] else ""
            row_labels.append((row,
                f"{dataset}{pair['image_id']}\n{pair['tercil'].replace('_extra', ' (adicional)')}\n"
                f"s = {pair['shift_buckets']:+d}\nΔL* = {style.fmt_es(pair['L_shift'], 0, signed=True)} (derivado)"
            ))
            for axis in axes[row]:
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_facecolor(style.COLORS["background"])
                for spine in axis.spines.values():
                    spine.set_color(style.COLORS["frame"])
                    spine.set_linewidth(style.LINE_WIDTHS["frame"])
        handles = [Line2D([], [], color=style.COLORS["accent"], linewidth=1.2, label="Lesión protegida")]
        fig.text(0.3, 0.97, "Pares de tono CIRCLe",
                 ha="left", va="top", fontsize=style.PANEL_TITLE_PT, color=style.COLORS["text"])
        fig.text(0.3, 0.925, "L*′ = clip(L* − 8s, 0, 100)",
                 ha="left", va="top", fontsize=style.LABEL_PT, color=style.COLORS["text"])
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.67, 0.035),
                   frameon=False, fontsize=style.LABEL_PT)
        fig.subplots_adjust(left=0.3, right=0.99, top=0.88, bottom=0.13, wspace=0.04, hspace=0.22)
        for row, label in row_labels:
            position = axes[row, 0].get_position()
            fig.text(0.012, position.y0 + position.height / 2, label,
                     ha="left", va="center", fontsize=style.LABEL_PT, color=style.COLORS["text"])
    return fig


def _traceability(manifest_path: Path, pairs: list[dict]) -> dict:
    return {
        "figure_id": "T5",
        "question": "Cómo se ve la transformación de tono (s = +2) sobre casos reales, con la lesión protegida.",
        "status": "Cálculo adicional (transformación aplicada a imágenes del servidor)",
        "manifest": {"path": manifest_path.name, "sha256": sha256_file(manifest_path)},
        "formula": "L*' = clip(L* - 8s, 0, 100)",
        "L_shift_semantics": "derived signed request before clipping: -FST_SHIFT_L_STEP * shift_buckets",
        "FST_SHIFT_L_STEP": FST_SHIFT_L_STEP,
        "mask_semantics": "true=lesion; lesion pixels are preserved",
        "code_source": {"path": "src/dermssl/circle.py",
                        "sha256": sha256_file(REPO / "src/dermssl/circle.py")},
        "figure_script": {"path": "scripts/figures/fig_t5_tone_pairs.py",
                          "sha256": sha256_file(REPO / "scripts/figures/fig_t5_tone_pairs.py")},
        "pairs": [{key: pair[key] for key in (
            "image_id", "dataset", "tercil", "shift_buckets", "L_shift", "mask_semantics", "resolution", "files"
        )} for pair in pairs],
    }


def build(manifest_path: str | Path | None = None, outdir: str | Path | None = None) -> list[Path]:
    manifest = Path(manifest_path) if manifest_path is not None else DEFAULT_MANIFEST
    pairs = load_pairs(manifest)
    target = Path(outdir) if outdir is not None else DEFAULT_OUTDIR
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"{STEM}{suffix}" for suffix in suffixes if (target / f"{STEM}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)
    fig = draw(pairs)
    try:
        paths = style.save_figure(fig, STEM, target)
    finally:
        plt.close(fig)
    (target / f"{STEM}.caption.txt").write_text(CAPTION + "\n", encoding="utf-8")
    (target / f"{STEM}.note.txt").write_text(NOTE + "\n", encoding="utf-8")
    (target / f"{STEM}.traceability.json").write_text(
        json.dumps(_traceability(manifest.resolve(), pairs), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    for path in build(args.manifest, args.outdir):
        print(path)
