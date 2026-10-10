"""S2 and S3: Dice of student_r3 and S4-A2 (point + 95 % CI, and per-image distribution), from committed evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402

RESULTS = "plan-segmentacion/camino-b-cnn/resultados"
SOURCES = {
    "student_r3": f"{RESULTS}/2026-09-21-final",
    "S4-A2": f"{RESULTS}/2026-09-28-s4-a2",
}
DATASETS = (("isic2018", "ISIC 2018", 2594), ("ham10000", "HAM10000", 10015))
PAIRED = f"{RESULTS}/2026-09-28-s4-a2/paired_vs_student_{{dataset}}.json"
MODELS = tuple(SOURCES)
MARKERS = {"student_r3": ("o", "none"), "S4-A2": ("s", "accent")}  # open circle = baseline, filled square = alternative
XLIM_S2 = (0.70, 0.82)
NOTE_S2 = ("Puntos: Dice medio por imagen; barras: IC 95 % bootstrap individual de cada modelo, no de la diferencia. "
           "El eje horizontal está truncado (0,70–0,82) para hacer visibles los intervalos. ISIC 2018 es parcialmente "
           "transductivo: 722 de sus imágenes estuvieron en el conjunto de preentrenamiento sin etiquetas.")
NOTE_S3 = ("Cada violín resume el Dice por imagen (marca interna: mediana; líneas: cuartiles) calculado sobre todas las "
           "imágenes de la evaluación; no se reconstruye desde medias.")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_summary(repo: Path, model: str, dataset: str, n: int) -> dict:
    path = repo / SOURCES[model] / f"evaluacion_{dataset}" / "summary.json"
    dice = json.loads(path.read_text(encoding="utf-8"))["metrics"]["dice"]
    if int(dice["n"]) != n:
        raise ValueError(f"{path}: n={dice['n']} but expected {n}")
    return {"mean": float(dice["mean"]), "lo": float(dice["ci95_low"]), "hi": float(dice["ci95_high"]),
            "n": n, "path": path}


def load_dice(repo: Path, model: str, dataset: str, n: int) -> np.ndarray:
    path = repo / SOURCES[model] / f"evaluacion_{dataset}" / "per_image.csv"
    with path.open(encoding="utf-8") as handle:
        values = np.array([float(row["dice"]) for row in csv.DictReader(handle)])
    if len(values) != n or not np.isfinite(values).all():
        raise ValueError(f"{path}: expected {n} finite Dice values, got {len(values)}")
    return values


def _style_axes(ax) -> None:
    style.style_axes(ax, grid_axis="x")


def build_s2(repo: Path, outdir: Path) -> list[Path]:
    summaries = {(m, d): load_summary(repo, m, d, n) for m in MODELS for d, _, n in DATASETS}
    with style.context():
        fig, axes = plt.subplots(1, 2, figsize=(style.FIGURE_WIDTH_IN, 2.9), sharex=True)
        for ax, (dataset, label, n) in zip(axes, DATASETS, strict=True):
            for y, model in enumerate(MODELS):
                row = summaries[(model, dataset)]
                marker, fill = MARKERS[model]
                ax.errorbar(row["mean"], y, xerr=[[row["mean"] - row["lo"]], [row["hi"] - row["mean"]]],
                            fmt=marker, markersize=7, color=style.COLORS["accent"], elinewidth=style.LINE_WIDTHS["curve"],
                            capsize=3, capthick=style.LINE_WIDTHS["curve"], markeredgewidth=style.LINE_WIDTHS["bar"],
                            markerfacecolor=style.COLORS["accent"] if fill == "accent" else style.COLORS["background"])
                ax.annotate(style.fmt_es(row["mean"], 3), (row["mean"], y), xytext=(0, 10), textcoords="offset points",
                            ha="center", va="bottom")
            ax.set_ylim(len(MODELS) - 0.4, -0.6)
            ax.set_yticks(range(len(MODELS)), MODELS if ax is axes[0] else [""] * len(MODELS))
            ax.set_xlim(*XLIM_S2)
            ticks = (0.72, 0.76, 0.80)
            ax.set_xticks(ticks, [style.fmt_es(v, 2) for v in ticks])
            ax.set_xlabel("Dice medio")
            ax.set_title(f"{label} (n = {style.fmt_es(n, 0)})", loc="left")
            _style_axes(ax)
        fig.subplots_adjust(left=0.2, right=0.96, top=0.86, bottom=0.2, wspace=0.12)
        stem = "fig_s2_dice_ci"
        paths = style.save_figure(fig, stem, outdir)
        plt.close(fig)
    _write_sidecars(repo, outdir, stem, "Dice medio con IC 95 % de student_r3 y S4-A2 en ISIC 2018 y HAM10000.",
                    NOTE_S2, [summaries[(m, d)]["path"] for m in MODELS for d, _, _ in DATASETS],
                    "Qué tan cerca están student_r3 y S4-A2 en Dice medio, con su incertidumbre individual.",
                    paired=[PAIRED.format(dataset=d) for d, _, _ in DATASETS])
    return paths


def build_s3(repo: Path, outdir: Path) -> list[Path]:
    data = {(m, d): load_dice(repo, m, d, n) for m in MODELS for d, _, n in DATASETS}
    with style.context():
        fig, axes = plt.subplots(1, 2, figsize=(style.FIGURE_WIDTH_IN, 3.6), sharey=True)
        for ax, (dataset, label, n) in zip(axes, DATASETS, strict=True):
            for x, model in enumerate(MODELS):
                values = data[(model, dataset)]
                marker, fill = MARKERS[model]
                parts = ax.violinplot(values, positions=[x], widths=0.8, showextrema=False)
                body = parts["bodies"][0]
                body.set_facecolor(style.COLORS["accent"] if fill == "accent" else style.COLORS["bar_light"])
                body.set_edgecolor(style.COLORS["accent"])
                body.set_linewidth(style.LINE_WIDTHS["bar"])
                body.set_alpha(1.0)
                q1, med, q3 = np.percentile(values, [25, 50, 75])
                line = style.COLORS["background"] if fill == "accent" else style.COLORS["text"]
                ax.hlines([q1, q3], x - 0.12, x + 0.12, color=line, linewidth=style.LINE_WIDTHS["curve"])
                ax.plot([x], [med], marker=marker, markersize=6, markeredgecolor=style.COLORS["text"],
                        markerfacecolor=style.COLORS["background"], markeredgewidth=style.LINE_WIDTHS["bar"], linestyle="none")
            ax.set_xticks(range(len(MODELS)), MODELS)
            ax.set_xlim(-0.6, len(MODELS) - 0.4)
            ax.set_ylim(0, 1)
            ax.set_yticks(np.linspace(0, 1, 6), [style.fmt_es(v, 1) for v in np.linspace(0, 1, 6)])
            ax.set_title(f"{label} (n = {style.fmt_es(n, 0)})", loc="left")
            style.style_axes(ax, grid_axis="y")
        axes[0].set_ylabel("Dice por imagen")
        fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12, wspace=0.08)
        stem = "fig_s3_dice_distribution"
        paths = style.save_figure(fig, stem, outdir)
        plt.close(fig)
    sources = [repo / SOURCES[m] / f"evaluacion_{d}" / "per_image.csv" for m in MODELS for d, _, _ in DATASETS]
    _write_sidecars(repo, outdir, stem, "Distribución del Dice por imagen de student_r3 y S4-A2 en ISIC 2018 y HAM10000.",
                    NOTE_S3, sources, "Cómo se reparte el Dice entre imágenes, más allá de la media.")
    return paths


def _write_sidecars(repo, outdir, stem, caption, note, sources, question, paired=None) -> None:
    out = Path(outdir)
    trace = {"figure": stem, "question": question, "status": "Artefactos verificados",
             "resampling_unit": "imagen", "n_resamples_per_ci": 1000,
             "sources": [{"path": str(Path(p).relative_to(repo)), "sha256": _sha(Path(p))} for p in sources],
             "paired_contrast_sources": paired, "script": "scripts/figures/fig_s2_s3_dice.py",
             "limits": ["ISIC 2018 parcialmente transductivo", "student_r3 y S4-A2 evaluados a 224 px, umbral 0,5"]}
    (out / f"{stem}.traceability.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{stem}.caption.txt").write_text(caption + "\n", encoding="utf-8")
    (out / f"{stem}.note.txt").write_text(note + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados" / "figuras")
    args = parser.parse_args()
    for path in (*build_s2(REPO, args.outdir), *build_s3(REPO, args.outdir)):
        print(path)
