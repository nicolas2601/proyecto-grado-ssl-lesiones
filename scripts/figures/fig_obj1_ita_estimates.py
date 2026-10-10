"""Separate, descriptive ROI-ITA estimates for HAM10000, ISIC 2019 and ISIC 2020.

The six groups use Groh's ITA thresholds, not clinically recorded Fitzpatrick types.
Missing estimates stay in the denominator and are reported explicitly.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import beautiful as bf  # noqa: E402
import fig1_ita as original  # noqa: E402
from fig_obj1_diagnosis_tone import SKIN_FILLS  # noqa: E402
from dermssl.fitzpatrick import FST_THRESHOLDS  # noqa: E402

DATASETS = original.DATASETS
BUCKETS = original.BUCKETS
ITA_BINS = np.arange(-90, 91, 10)
BACKGROUND, PANEL, INK, MUTED = "#FBFBFA", "#FAF9F7", "#302C29", "#645D58"
BORDER, GRID = "#AAA39E", "#E5DED8"
STEM = "fig_obj1_ita"


def load_dataset(repo: Path, dataset: str) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    raw, rows = original._load(Path(repo) / original.E5_DIR / f"e5_fitzpatrick_{dataset}.json")
    thresholds = raw["fst_thresholds"]
    if thresholds != list(FST_THRESHOLDS) or not rows:
        raise ValueError(f"Invalid ITA thresholds or empty dataset: {dataset}")
    counts = dict.fromkeys(BUCKETS, 0)
    values: dict[str, list[float]] = {bucket: [] for bucket in BUCKETS}
    seen: set[str] = set()
    n_abstention = 0
    for row in rows:
        image_id, status, bucket = row["image_id"], row["roi_status"], row["roi_bucket"]
        if not image_id or image_id in seen:
            raise ValueError(f"Missing or duplicate image ID in {dataset}")
        seen.add(image_id)
        if status != "ok":
            if bucket is not None:
                raise ValueError(f"Abstaining image received an ITA group: {dataset} {image_id}")
            n_abstention += 1
            continue
        angle = row["roi_ita"]
        if (bucket not in BUCKETS or not isinstance(angle, (int, float)) or not math.isfinite(angle)
                or not -90 <= angle <= 90 or original.bucket_of(angle, thresholds) != bucket):
            raise ValueError(f"Invalid ITA or group: {dataset} {image_id}")
        counts[bucket] += 1
        values[bucket].append(float(angle))
    return {"dataset": dataset, "n_total": len(rows), "n_valid": len(rows) - n_abstention,
            "n_abstention": n_abstention, "counts": counts, "values": values}


def _style_axis(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _draw_counts(ax, data: dict) -> None:
    numbers = [data["counts"][bucket] for bucket in BUCKETS]
    bars = ax.bar(range(len(BUCKETS)), numbers, width=0.74, color=SKIN_FILLS,
                  edgecolor=[original._mix(color, "black", 0.7) for color in SKIN_FILLS], linewidth=0.8)
    for bar, number in zip(bars, numbers, strict=True):
        ax.annotate(bf.fmt_es(number, 0), (bar.get_x() + bar.get_width() / 2, number),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(BUCKETS)), BUCKETS)
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(numbers) * 3)
    ax.set(xlabel="Grupo I–VI estimado por ITA", ylabel="Número de imágenes · escala logarítmica")
    _style_axis(ax)


def _draw_histogram(ax, data: dict) -> None:
    base = np.zeros(len(ITA_BINS) - 1, dtype=int)
    for bucket, color in zip(BUCKETS, SKIN_FILLS, strict=True):
        hist, _ = np.histogram(data["values"][bucket], bins=ITA_BINS)
        ax.bar(ITA_BINS[:-1], hist, width=np.diff(ITA_BINS), align="edge", bottom=base,
               color=color, edgecolor=original._mix(color, "black", 0.7), linewidth=0.35)
        base += hist
    ax.set(xlabel="ITA de la ROI perilesional (°)", ylabel="Número de imágenes", xlim=(-90, 90))
    ax.set_ylim(0, max(1, base.max()) * 1.1)
    _style_axis(ax)
    handles = [Patch(facecolor=color, edgecolor=original._mix(color, "black", 0.7),
                     label=f"{bucket} · n = {bf.fmt_es(data['counts'][bucket], 0)}")
               for bucket, color in zip(BUCKETS, SKIN_FILLS, strict=True)]
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=9)


def build(repo: Path, dataset: str, outdir: Path) -> list[Path]:
    data = load_dataset(repo, dataset)
    bf.apply_style(width=16)
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "svg.fonttype": "none"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BACKGROUND)
    _draw_counts(left, data)
    _draw_histogram(right, data)
    name = original.DATASET_LABELS[dataset]
    left.set_title(f"(a) {name} · estimación I–VI", loc="left", fontsize=16, color=INK, pad=17)
    right.set_title("(b) Distribución ITA de la misma ROI", loc="left", fontsize=16, color=INK, pad=17)
    fig.text(0.055, 0.045, f"N = {bf.fmt_es(data['n_total'], 0)}; ITA válido: "
             f"{bf.fmt_es(data['n_valid'], 0)}; abstención: {bf.fmt_es(data['n_abstention'], 0)}. "
             "ITA estimado, no validado como fototipo Fitzpatrick clínico.", color=MUTED, fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.20, wspace=0.24)
    paths = bf.save_all(fig, f"{STEM}_{dataset}", outdir)
    bf.close(fig)
    return paths


def build_all(repo: Path = REPO, outdir: Path | None = None) -> dict[str, list[Path]]:
    target = Path(outdir) if outdir else Path(repo) / "resultados/figuras"
    return {dataset: build(repo, dataset, target) for dataset in DATASETS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    for dataset, paths in build_all(outdir=args.outdir).items():
        print(f"{dataset}: {', '.join(str(path) for path in paths)}")
