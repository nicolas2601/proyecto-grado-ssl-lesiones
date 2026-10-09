"""Figures 2–5: reconstructed CIELAB L*–b* distributions over the E5 ITA bands.

Each dataset is exported on its own and all three are exported together. b* is
reconstructed from the committed, four-decimal L* and ITA values because E5 does
not serialize roi_b; rows without an invertible ITA are counted and omitted.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beautiful as bf  # noqa: E402
import fig1_ita as f1  # noqa: E402

DATASETS = f1.DATASETS
DATASET_COLORS = {"ham10000": "#B77E71", "isic2019": "#9B8060", "isic2020": "#79563F"}
STEM = "fig2_ita_lab"
OUTDIR = Path("resultados") / "figuras"
SINGULAR_EPSILON = 1e-3
B_STAR_MAX = 60.0
L_STAR_MIN, L_STAR_MAX = 20.0, 85.0
WIDTH_IN, HEIGHT_IN = 13.0, 8.0
TEXT_PT = 13
BAND_ALPHA = 0.34
POINT_SIZE = 7


def reconstruct_b_star(l_star: float | None, ita: float | None) -> float | None:
    """Invert ITA = atan((L* - 50) / b*) for finite, non-singular inputs."""
    if l_star is None or ita is None:
        return None
    if not math.isfinite(l_star) or not math.isfinite(ita):
        return None
    if abs(l_star - 50.0) < SINGULAR_EPSILON or abs(ita) < SINGULAR_EPSILON:
        return None
    tangent = math.tan(math.radians(ita))
    if abs(tangent) < SINGULAR_EPSILON:
        return None
    b_star = (l_star - 50.0) / tangent
    if not math.isfinite(b_star) or b_star <= 0:
        return None
    return b_star


def _source_path(repo: Path, dataset: str) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return Path(repo) / f1.E5_DIR / f"e5_fitzpatrick_{dataset}.json"


def thresholds_for(repo: Path, dataset: str) -> list[float]:
    raw = json.loads(_source_path(repo, dataset).read_text(encoding="utf-8"))
    return [float(value) for value in raw["fst_thresholds"]]


def load_dataset_points(repo: Path, dataset: str) -> dict:
    """Load points from valid ROI ITA rows and report omissions explicitly."""
    raw = json.loads(_source_path(repo, dataset).read_text(encoding="utf-8"))
    columns = raw["per_image"]["columns"]
    points = []
    n_ita_valid = 0
    n_singular = 0
    for values in raw["per_image"]["rows"]:
        row = dict(zip(columns, values, strict=True))
        if row["roi_status"] != "ok":
            continue
        n_ita_valid += 1
        b_star = reconstruct_b_star(row["roi_L"], row["roi_ita"])
        if b_star is None:
            n_singular += 1
            continue
        points.append({"image_id": row["image_id"], "l_star": float(row["roi_L"]),
                       "b_star": b_star, "ita": float(row["roi_ita"])})
    return {"dataset": dataset, "n_total": len(raw["per_image"]["rows"]),
            "n_ita_valid": n_ita_valid, "n_singular": n_singular, "n_points": len(points), "points": points,
            "thresholds": [float(value) for value in raw["fst_thresholds"]]}


def _band_labels(thresholds: list[float]) -> list[str]:
    labels = [f"I · ITA > {bf.fmt_es(thresholds[0], 0)}°"]
    for index in range(1, len(f1.BUCKETS) - 1):
        labels.append(f"{f1.BUCKETS[index]} · {bf.fmt_es(thresholds[index - 1], 0)}° ≥ ITA > "
                      f"{bf.fmt_es(thresholds[index], 0)}°")
    labels.append(f"VI · ITA ≤ {bf.fmt_es(thresholds[-1], 0)}°")
    return labels


def _draw_bands(ax, thresholds: list[float], fills: list) -> None:
    b_grid = np.linspace(0, B_STAR_MAX, 400)
    boundary_lines = [50 + b_grid * math.tan(math.radians(value)) for value in thresholds]
    lower_lines = [*boundary_lines, np.full_like(b_grid, L_STAR_MIN)]
    upper_lines = [np.full_like(b_grid, L_STAR_MAX), *boundary_lines]
    labels = _band_labels(thresholds)
    for lower, upper, fill, label in zip(lower_lines, upper_lines, fills, labels, strict=True):
        ax.fill_between(b_grid, lower, upper, color=fill, alpha=BAND_ALPHA, linewidth=0,
                        label=label, zorder=0)
    for boundary in boundary_lines:
        ax.plot(b_grid, boundary, color="#6b4c36", linewidth=0.8, alpha=0.8, zorder=1)


def _draw_dataset_points(ax, dataset_data: dict, *, combined: bool) -> None:
    names = DATASETS if combined else (dataset_data["dataset"],)
    smallest_dataset = min(len(dataset_data[name]["points"]) for name in names) if combined else 0
    for dataset in names:
        data = dataset_data[dataset] if combined else dataset_data
        points = data["points"]
        alpha = 0.28 * smallest_dataset / len(points) if combined and points else 0.32
        ax.scatter([point["b_star"] for point in points], [point["l_star"] for point in points],
                   s=POINT_SIZE, marker="o", color=DATASET_COLORS[dataset], edgecolors="none",
                   alpha=alpha, rasterized=True, zorder=3,
                   label=f"{f1.DATASET_LABELS[dataset]} (n = {bf.fmt_es(len(points), 0)})")


def _legends(ax, thresholds: list[float], *, combined: bool) -> None:
    fills, _ = f1.bucket_palette()
    band_handles = [mpatches.Patch(facecolor=fill, edgecolor="none", alpha=BAND_ALPHA, label=label)
                    for fill, label in zip(fills, _band_labels(thresholds), strict=True)]
    if combined:
        dataset_handles = [mpatches.Patch(facecolor=DATASET_COLORS[dataset], edgecolor="none",
                                          label=f1.DATASET_LABELS[dataset]) for dataset in DATASETS]
        dataset_legend = ax.legend(handles=dataset_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                                   frameon=False, fontsize=TEXT_PT - 2, title="Dataset",
                                   title_fontsize=TEXT_PT - 1)
        ax.add_artist(dataset_legend)
    ax.legend(handles=band_handles, loc="upper left", bbox_to_anchor=(1.02, 0.65), frameon=False,
              fontsize=TEXT_PT - 3, title="Convención ITA I–VI", title_fontsize=TEXT_PT - 1,
              handlelength=1.1, labelspacing=0.7)


def _render(repo: Path, outdir: Path, datasets: tuple[str, ...], loaded: dict) -> list[Path]:
    combined = len(datasets) > 1
    data = loaded if combined else loaded[datasets[0]]
    thresholds = loaded[datasets[0]]["thresholds"]
    for dataset in datasets[1:]:
        if loaded[dataset]["thresholds"] != thresholds:
            raise ValueError(f"ITA thresholds differ for {datasets[0]} and {dataset}")

    bf.apply_style(width=WIDTH_IN)
    plt.rcParams.update({key: TEXT_PT for key in ("font.size", "axes.labelsize", "xtick.labelsize",
                                                   "ytick.labelsize", "legend.fontsize")})
    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    _draw_bands(ax, thresholds, f1.bucket_palette()[0])
    _draw_dataset_points(ax, data, combined=combined)
    ax.set(xlim=(0, B_STAR_MAX), ylim=(L_STAR_MIN, L_STAR_MAX),
           xlabel="b* (croma amarillo)", ylabel="L* (luminosidad)")
    panel_name = "comparación de los tres datasets" if combined else f1.DATASET_LABELS[datasets[0]]
    ax.set_title(f"Plano CIELAB L*–b* · {panel_name}", loc="left", pad=18, fontweight="bold")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(lambda value, _: bf.fmt_es(value, 0))
    ax.yaxis.set_major_formatter(lambda value, _: bf.fmt_es(value, 0))
    bf.style_axes(ax)
    ax.grid(False, which="minor")
    _legends(ax, thresholds, combined=combined)

    names = " + ".join(f1.DATASET_LABELS[dataset] for dataset in datasets)
    counts = " · ".join(
        f"{f1.DATASET_LABELS[dataset]}: n = {bf.fmt_es(len(loaded[dataset]['points']), 0)}"
        f" ({loaded[dataset]['n_singular']} ITA no recuperable)"
        for dataset in datasets
    )
    fig.text(0.08, 0.035,
             f"{names} · {counts}\n"
             "b* reconstruido desde L* e ITA redondeados; opacidad balanceada en comparación; "
             "I–VI son umbrales del estimador, no fototipos clínicos.",
             ha="left", va="bottom", fontsize=TEXT_PT - 3, style="italic")
    fig.subplots_adjust(left=0.09, right=0.72, top=0.85, bottom=0.18)
    stem = f"{STEM}_{'combined' if combined else datasets[0]}"
    paths = bf.save_all(fig, stem, outdir)
    svg = next(path for path in paths if path.suffix == ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    bf.close(fig)
    return paths


def build_all(repo: Path = REPO, outdir: Path | None = None) -> dict[str, list[Path]]:
    """Export three per-dataset views plus a combined view using the same E5 thresholds."""
    repo = Path(repo)
    target = Path(outdir) if outdir else repo / OUTDIR
    loaded = {dataset: load_dataset_points(repo, dataset) for dataset in DATASETS}
    outputs = {dataset: _render(repo, target, (dataset,), loaded) for dataset in DATASETS}
    outputs["combined"] = _render(repo, target, DATASETS, loaded)
    return outputs


if __name__ == "__main__":
    for dataset, paths in build_all().items():
        print(f"{dataset}: {', '.join(str(path) for path in paths)}")
