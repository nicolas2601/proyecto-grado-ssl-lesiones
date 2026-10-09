"""Objective 1: CIELAB pixel distributions for an explicitly chosen image population.

Both panels use exactly the same pixels. The caller must choose full images or
skin outside matched lesion masks; no dataset or ROI is silently inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.colors as colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beautiful as bf  # noqa: E402
from dermssl.fitzpatrick import BORDER_LUMINANCE_THRESHOLD, srgb_to_lab  # noqa: E402

STEM = "fig_obj1_cielab_pixels"
L_BINS = np.arange(0, 102, 2)
AB_BINS = np.arange(-128, 129, 4)
LOG_RANGE = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BACKGROUND, PANEL, INK, MUTED = "#FBFBFA", "#FAF9F7", "#302C29", "#645D58"
BORDER, GRID = "#AAA39E", "#E5DED8"


def _image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")
    paths = [path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not paths or any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("Image directory is empty or contains unsafe image links")
    if len({path.stem for path in paths}) != len(paths):
        raise ValueError("Duplicate image IDs in image directory")
    return sorted(paths, key=lambda path: path.name)


def _read_pixels(path: Path, mask_dir: Path | None, resolution: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB").resize((resolution, resolution), Image.Resampling.BILINEAR),
                         dtype=np.float64) / 255.0
    eligible = np.ones(rgb.shape[:2], dtype=bool)
    if mask_dir is not None:
        mask_path = mask_dir / f"{path.stem}_segmentation.png"
        if not mask_path.is_file() or mask_path.is_symlink():
            raise FileNotFoundError(f"Missing/unsafe lesion mask for {path.stem}: {mask_path}")
        with Image.open(mask_path) as mask:
            lesion = np.asarray(mask.convert("L").resize((resolution, resolution), Image.Resampling.NEAREST)) > 127
        eligible = ~lesion & (rgb.mean(axis=-1) >= BORDER_LUMINANCE_THRESHOLD)
    return srgb_to_lab(rgb)[eligible]


def collect(image_dir: Path, population: str, resolution: int, mask_dir: Path | None,
            sample_per_image: int, seed: int) -> dict:
    """Stream deterministic per-image samples into two histograms, never pooling raw images."""
    if population not in ("full-image", "outside-lesion"):
        raise ValueError("population must be full-image or outside-lesion")
    if population == "outside-lesion" and (mask_dir is None or not Path(mask_dir).is_dir()):
        raise ValueError("outside-lesion population requires an existing mask directory")
    if population == "full-image" and mask_dir is not None:
        raise ValueError("full-image population must not have a mask directory")
    if resolution <= 0 or sample_per_image < 0:
        raise ValueError("resolution must be positive and sample_per_image non-negative")
    paths = _image_paths(Path(image_dir))
    rng = np.random.default_rng(seed)
    l_hist = np.zeros(len(L_BINS) - 1, dtype=np.int64)
    ab_hist = np.zeros((len(AB_BINS) - 1, len(AB_BINS) - 1), dtype=np.int64)
    n_pixels, n_without_pixels = 0, 0
    for path in paths:
        pixels = _read_pixels(path, Path(mask_dir) if mask_dir else None, resolution)
        if not len(pixels):
            n_without_pixels += 1
            continue
        if sample_per_image and len(pixels) > sample_per_image:
            pixels = pixels[rng.choice(len(pixels), sample_per_image, replace=False)]
        if not np.isfinite(pixels).all() or (np.abs(pixels[:, 1:]) > 128).any():
            raise ValueError(f"CIELAB values outside plotting range for {path.stem}")
        l_hist += np.histogram(pixels[:, 0], bins=L_BINS)[0]
        ab_hist += np.histogram2d(pixels[:, 1], pixels[:, 2], bins=(AB_BINS, AB_BINS))[0].astype(np.int64)
        n_pixels += len(pixels)
    if not n_pixels or l_hist.sum() != n_pixels or ab_hist.sum() != n_pixels:
        raise ValueError("Selected pixels absent or histogram totals disagree")
    digest = hashlib.sha256("\n".join(path.stem for path in paths).encode()).hexdigest()
    return {"l_hist": l_hist, "ab_hist": ab_hist, "n_pixels": n_pixels, "n_images": len(paths),
            "n_without_pixels": n_without_pixels, "image_ids_sha256": digest,
            "population": population, "resolution": resolution, "sample_per_image": sample_per_image,
            "seed": seed}


def _axes_style(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=11)
    ax.set_axisbelow(True)


def _draw(data: dict):
    bf.apply_style(width=16)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BACKGROUND)
    left.bar(L_BINS[:-1], data["l_hist"], width=np.diff(L_BINS), align="edge", color="#F2C598",
             edgecolor="#725043", linewidth=0.4)
    left.set(xlabel="Luminosidad L*", ylabel="Frecuencia de píxeles", xlim=(0, 100))
    left.set_title("(a) Distribución de luminosidad", loc="left", color=INK, fontsize=17, pad=17)
    _axes_style(left)
    left.grid(axis="y", color=GRID, linewidth=0.7)
    histogram = np.ma.masked_equal(data["ab_hist"].T, 0)
    occupied = histogram.compressed()
    norm = colors.LogNorm(vmin=1, vmax=int(occupied.max())) if occupied.max() / occupied.min() >= LOG_RANGE else None
    palette = colors.LinearSegmentedColormap.from_list("skin_density", ("#FEF3E0", "#CE8B51", "#53311B"))
    density = right.pcolormesh(AB_BINS, AB_BINS, histogram, cmap=palette, norm=norm,
                               vmin=None if norm else 0, shading="flat", rasterized=True)
    right.set(xlabel="a* (crominancia)", ylabel="b* (crominancia)")
    right.set_title("(b) Crominancia de los mismos píxeles", loc="left", color=INK, fontsize=17, pad=17)
    _axes_style(right)
    bar = fig.colorbar(density, ax=right, pad=0.02, fraction=0.044)
    bar.set_label("Densidad de píxeles", color=INK, fontsize=12)
    bar.ax.tick_params(colors=INK)
    count = bf.fmt_es(data["n_pixels"], 0)
    scope = "piel fuera de la lesión" if data["population"] == "outside-lesion" else "imagen completa"
    sampling = (f"máx. {bf.fmt_es(data['sample_per_image'], 0)} por imagen"
                if data["sample_per_image"] else "todos los píxeles elegibles")
    fig.text(0.055, 0.045, f"Población: {scope}; N = {bf.fmt_es(data['n_images'], 0)} imágenes; "
             f"{count} píxeles ({sampling}); imágenes sin píxeles: {data['n_without_pixels']}.",
             color=MUTED, fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.19, top=0.88, wspace=0.27)
    return fig


def build(image_dir: Path, population: str, resolution: int, mask_dir: Path | None = None,
          sample_per_image: int = 0, seed: int = 20260923, outdir: Path | None = None) -> list[Path]:
    data = collect(image_dir, population, resolution, mask_dir, sample_per_image, seed)
    target = Path(outdir) if outdir else REPO / "resultados/figuras"
    fig = _draw(data)
    paths = bf.save_all(fig, STEM, target)
    bf.close(fig)
    summary = {key: value for key, value in data.items() if key not in ("l_hist", "ab_hist")}
    summary.update({"l_bins": L_BINS.tolist(), "ab_bins": AB_BINS.tolist(),
                    "l_hist": data["l_hist"].tolist(), "ab_hist": data["ab_hist"].tolist()})
    (target / f"{STEM}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--population", choices=("full-image", "outside-lesion"), required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--sample-per-image", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    for output in build(args.image_dir, args.population, args.resolution, args.mask_dir,
                        args.sample_per_image, args.seed, args.outdir):
        print(output)
