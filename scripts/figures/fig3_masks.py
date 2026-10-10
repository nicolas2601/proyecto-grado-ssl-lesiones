"""Figure 3: segmentation masks. (a) Dice distribution per segmenter and dataset, (b) a qualitative panel.

The sources are the committed ``per_image.csv`` of the two segmenters: ``student_r3`` (U-Net, 2026-09-21-final)
and ``A2`` (frozen DINOv2 + DPT decoder, 2026-09-28-s4-a2), on HAM10000 and ISIC 2018. The qualitative cases are
picked by Dice percentile of the MEAN Dice of both segmenters (p90 good, p50 median, p10 bad), never by eye; the
images and masks of those cases come from ``fig3_extract_cases.py`` (run once on the pod).

    PYTHONPATH=$HOME/work/tmp/mpl python scripts/figures/fig3_masks.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beautiful as bf  # noqa: E402

RESULTS = "plan-segmentacion/camino-b-cnn/resultados"
SOURCES = {
    ("student_r3", "ham10000"): f"{RESULTS}/2026-09-21-final/evaluacion_ham10000/per_image.csv",
    ("student_r3", "isic2018"): f"{RESULTS}/2026-09-21-final/evaluacion_isic2018/per_image.csv",
    ("A2", "ham10000"): f"{RESULTS}/2026-09-28-s4-a2/evaluacion_ham10000/per_image.csv",
    ("A2", "isic2018"): f"{RESULTS}/2026-09-28-s4-a2/evaluacion_isic2018/per_image.csv",
}
SEGMENTERS = ("student_r3", "A2")
DATASETS = ("ham10000", "isic2018")
DATASET_LABEL = {"ham10000": "HAM10000", "isic2018": "ISIC 2018"}
CASE_ROLES = (("bueno", 90), ("mediano", 50), ("malo", 10))
CASES_JSON = "resultados/figuras/fig3_cases.json"
CASES_NPZ = "resultados/figuras/fuentes/fig3_casos.npz"
STEM = "fig3_masks"
WIDTH = 16.0
SEGMENTER_STYLE = {"student_r3": (bf.FILLS[0], bf.EDGES[0], "student_r3 (U-Net)"),
                   "A2": (bf.FILLS[2], bf.EDGES[2], "A2 (DINOv2 congelado + DPT)")}
REFERENCE_COLOR = bf.EDGES[1]
COLUMNS = ("imagen", "referencia", "student_r3", "A2")
TEXT_PT = 17  # ticks, axis labels and legend: the same size and width as figure 1, so the page typography matches
NOTE_PT = 14  # annotations (medians, tile captions): one step below


class CasesMismatchError(RuntimeError):
    """The pod-side cases file does not hold the image and the three masks of every selected case."""


def load_dice(repo: Path = REPO) -> dict[tuple[str, str], dict[str, float]]:
    """Dice per image id for every (segmenter, dataset), read from the committed CSV files."""
    out = {}
    for key, rel in SOURCES.items():
        with open(repo / rel, newline="", encoding="utf-8") as handle:
            out[key] = {row["image_id"]: float(row["dice"]) for row in csv.DictReader(handle)}
    return out


def select_cases(repo: Path = REPO) -> list[dict]:
    """The image nearest to p90 / p50 / p10 of the mean Dice of both segmenters, per dataset (ties: smallest id)."""
    dice = load_dice(repo)
    cases = []
    for dataset in DATASETS:
        first, second = (dice[(name, dataset)] for name in SEGMENTERS)
        ids = sorted(set(first) & set(second))
        mean = np.array([(first[i] + second[i]) / 2 for i in ids])
        for role, percentile in CASE_ROLES:
            image_id = ids[int(np.argmin(np.abs(mean - np.percentile(mean, percentile))))]
            cases.append({"dataset": dataset, "role": role, "percentile": percentile, "image_id": image_id,
                          "dice": {name: dice[(name, dataset)][image_id] for name in SEGMENTERS}})
    return cases


def figure_data(repo: Path = REPO) -> dict:
    """Numbers drawn by panel (a), plus the qualitative cases of panel (b)."""
    dice = load_dice(repo)
    summary: dict = {dataset: {} for dataset in DATASETS}
    for (segmenter, dataset), by_id in dice.items():
        values = np.array(list(by_id.values()))
        summary[dataset][segmenter] = {
            "n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
        }
    return {"dice": summary, "cases": select_cases(repo)}


def write_cases(repo: Path = REPO) -> Path:
    path = repo / CASES_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(select_cases(repo), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _violins(ax, repo: Path) -> None:
    """Panel (a): Dice per image, one violin per segmenter inside each dataset group, with box and median."""
    dice = load_dice(repo)
    offsets = {"student_r3": -0.2, "A2": 0.2}
    for group, dataset in enumerate(DATASETS):
        for name in SEGMENTERS:
            values = np.array(list(dice[(name, dataset)].values()))
            fill, edge, _ = SEGMENTER_STYLE[name]
            at = group + offsets[name]
            body = ax.violinplot([values], positions=[at], widths=0.38, showextrema=False)["bodies"][0]
            body.set(facecolor=fill, edgecolor=edge, alpha=0.75, linewidth=1.5)
            ax.boxplot([values], positions=[at], widths=0.07, showfliers=False, patch_artist=True,
                       boxprops={"facecolor": edge, "edgecolor": edge}, whiskerprops={"color": edge},
                       capprops={"color": edge}, medianprops={"color": "white", "linewidth": 2})
            ax.text(at, 1.02, bf.fmt_es(float(np.median(values)), 3), ha="center", va="bottom",
                    fontsize=NOTE_PT, color=edge)
    n = {d: len(dice[("student_r3", d)]) for d in DATASETS}
    ax.set_xticks(range(len(DATASETS)), [f"{DATASET_LABEL[d]}\nn = {bf.fmt_es(n[d], 0)}" for d in DATASETS])
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: bf.fmt_es(value, 1)))
    ax.set_ylabel("Dice por imagen")
    bf.style_axes(ax)
    handles = [Patch(facecolor=SEGMENTER_STYLE[n][0], edgecolor=SEGMENTER_STYLE[n][1], label=SEGMENTER_STYLE[n][2])
               for n in SEGMENTERS]
    bf.top_legend(ax, handles=handles, labels=[h.get_label() for h in handles], fontsize=TEXT_PT)
    ax.text(0.0, -0.2, "Número sobre cada violín: mediana del Dice.", transform=ax.transAxes, fontsize=NOTE_PT,
            style="italic", va="top")


def _check_cases(arrays, cases: list[dict]) -> None:
    missing = [f"{c['dataset']}__{c['image_id']}__{part}" for c in cases
               for part in ("image", "reference", *SEGMENTERS) if f"{c['dataset']}__{c['image_id']}__{part}" not in arrays]
    if missing:
        raise CasesMismatchError(f"cases file lacks {len(missing)} arrays, e.g. {missing[0]}")


def _frameless(ax, label: str = "") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label:
        ax.set_xlabel(label, fontsize=NOTE_PT, labelpad=3)


def _reference_tile(ax, image, reference) -> None:
    ax.imshow(image)
    ax.imshow(np.ma.masked_equal(reference, 0), cmap="gray", alpha=0.35, vmin=0, vmax=1)
    ax.contour(reference.astype(float), levels=[0.5], colors=[REFERENCE_COLOR], linewidths=3)


def _prediction_tile(ax, image, reference, prediction, color, label: str) -> None:
    """The predicted contour in the segmenter's colour over the image, the reference as a dashed contour."""
    ax.imshow(image)
    ax.contour(reference.astype(float), levels=[0.5], colors=[REFERENCE_COLOR], linewidths=2, linestyles="dashed")
    ax.contour(prediction.astype(float), levels=[0.5], colors=[color], linewidths=3)
    _frameless(ax, label)


def _case_row(fig, grid, row: int, block: int, case: dict, arrays) -> None:
    base = f"{case['dataset']}__{case['image_id']}"
    image, reference = arrays[f"{base}__image"], arrays[f"{base}__reference"]
    axes = [fig.add_subplot(grid[1 + row, block * 4 + col]) for col in range(len(COLUMNS))]
    axes[0].imshow(image)
    _reference_tile(axes[1], image, reference)
    for ax, name in zip(axes[2:], SEGMENTERS):
        _prediction_tile(ax, image, reference, arrays[f"{base}__{name}"], SEGMENT_COLOR[name],
                         f"Dice {bf.fmt_es(case['dice'][name], 3)}")
    for ax in axes[:2]:
        _frameless(ax)
    if row == 0:
        for ax, name in zip(axes, COLUMNS):
            ax.set_title(name, fontsize=TEXT_PT, pad=6)
    if block == 0:
        axes[0].set_ylabel(f"{case['role']}\n(p{case['percentile']})", fontsize=TEXT_PT)


def _cases_panel(fig, cell, arrays, cases: list[dict]) -> None:
    """Panel (b): 3 cases (rows) x 4 columns (image, reference, student_r3, A2) per dataset, side by side."""
    grid = cell.subgridspec(1 + len(CASE_ROLES), 2 * len(COLUMNS), hspace=0.18, wspace=0.04,
                            height_ratios=[0.25] + [1] * len(CASE_ROLES))
    for block, dataset in enumerate(DATASETS):
        header = fig.add_subplot(grid[0, block * 4:(block + 1) * 4])
        header.axis("off")
        header.text(0.5, 0.2, DATASET_LABEL[dataset], ha="center", fontsize=TEXT_PT + 3, fontweight="bold")
        for row, case in enumerate(c for c in cases if c["dataset"] == dataset):
            _case_row(fig, grid, row, block, case, arrays)


SEGMENT_COLOR = {name: style[1] for name, style in SEGMENTER_STYLE.items()}


def build(repo: Path = REPO, outdir: Path | None = None, cases_npz: Path | None = None,
          *, panel_a_only: bool = False) -> dict:
    """Draw figure 3; the qualitative panel is mandatory unless panel_a_only is explicitly selected."""
    outdir = outdir or repo / "resultados" / "figuras"
    npz_path = cases_npz if cases_npz is not None else repo / CASES_NPZ
    cases = select_cases(repo)
    arrays = None
    if npz_path.is_file():
        arrays = dict(np.load(npz_path))
        _check_cases(arrays, cases)
    elif not panel_a_only:
        raise FileNotFoundError(f"panel b requires verified pod cases: {npz_path}; pass panel_a_only=True only for an explicitly partial figure")
    bf.apply_style(WIDTH)
    plt.rcParams.update({k: TEXT_PT for k in ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    fig = plt.figure(figsize=(WIDTH, 15.5 if arrays is not None else 7.5))
    grid = GridSpec(2 if arrays is not None else 1, 1, figure=fig, height_ratios=[1, 1.7] if arrays is not None else [1],
                    hspace=0.45)
    _violins(fig.add_subplot(grid[0]), repo)
    fig.text(0.01, 0.99, "(a)", fontsize=TEXT_PT + 3, fontweight="bold", va="top")
    if arrays is not None:
        _cases_panel(fig, grid[1], arrays, cases)
        fig.text(0.01, 0.58, "(b)", fontsize=TEXT_PT + 3, fontweight="bold", va="top")
    paths = bf.save_all(fig, STEM, outdir)
    plt.close(fig)
    return {"paths": paths, "panel_b": arrays is not None}


    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel-a-only", action="store_true", help="emit an explicitly partial figure without the pod-side panel (b)")
    args = parser.parse_args()
    if args.panel_a_only:
        print(build(panel_a_only=True)["paths"])
    else:
        print(write_cases())
        print(build()["paths"])
