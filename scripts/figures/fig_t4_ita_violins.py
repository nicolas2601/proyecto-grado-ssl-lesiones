"""T4: per-image ITA distributions from verified E5 ROI estimates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402
from dermssl.fitzpatrick import FST_THRESHOLDS, MIN_B_FOR_ITA, fst_bucket  # noqa: E402

E5_DIR = "resultados/banco_completo/e5"
STEM = "fig_t4_ita_violins"
DATASETS = {
    "ham10000": "HAM10000",
    "isic2019": "ISIC 2019",
    "isic2020": "ISIC 2020",
}
BUCKETS = ("I", "II", "III", "IV", "V", "VI")
ITA_MIN = -90.0
ITA_MAX = 90.0
BIN_WIDTH_DEG = 1.0
SMOOTH_SIGMA_DEG = 2.0
MAX_HALF_WIDTH = 0.32
NOTE = (
    "Cada violín representa ITA por imagen con `roi_status=ok` del estimador ROI de E5; el anillo perilesional "
    "se calcula alrededor de la lesión y sus ITA se expresan en grados. I–VI son categorías del estimador ITA; "
    "no equivalen a fototipos clínicos validados. Las abstenciones no entran a la distribución y se reportan con n. "
    "ISIC incluye y reporta por separado los valores válidos del fallback `no_mask_border`. PAD-UFES-20 usa Otsu "
    "y fototipo registrado; queda fuera de este violín y se cubre en `fig_char_ita_distribution` y T3."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(repo: Path, relative: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", relative],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _load_dataset(repo: Path, dataset: str, thresholds: tuple[float, ...]) -> dict:
    relative = f"{E5_DIR}/e5_fitzpatrick_{dataset}.json"
    path = repo / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    if tuple(document.get("fst_thresholds", ())) != thresholds:
        raise ValueError(f"{path}: ITA thresholds differ from the verified estimator thresholds")
    if float(document.get("min_b_for_ita", -1)) != MIN_B_FOR_ITA:
        raise ValueError(f"{path}: minimum b* guard differs from the estimator")

    per_image = document["per_image"]
    columns = per_image["columns"]
    rows = per_image["rows"]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError(f"{path}: per-image row width differs from column list")
    index = {name: columns.index(name) for name in (
        "image_id", "roi_bucket", "roi_estimator", "roi_image", "roi_ita", "roi_status"
    )}
    image_ids = [row[index["image_id"]] for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{path}: duplicate image IDs in per-image records")

    values = []
    fallback_count = 0
    fallback_origins: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row[index["roi_status"]]
        if not isinstance(status, str) or not status:
            raise ValueError(f"{path}: missing ROI status")
        status_counts[status] = status_counts.get(status, 0) + 1
        ita = row[index["roi_ita"]]
        if status != "ok":
            if _is_finite_number(ita):
                raise ValueError(f"{path}: non-ok ROI row has a finite ITA value")
            continue
        if not _is_finite_number(ita) or not ITA_MIN <= float(ita) <= ITA_MAX:
            raise ValueError(f"{path}: ok ROI row has an invalid ITA value")
        bucket = row[index["roi_bucket"]]
        expected_bucket = BUCKETS[fst_bucket(float(ita)) - 1]
        if bucket != expected_bucket:
            raise ValueError(f"{path}: ROI bucket {bucket!r} does not match ITA {ita}")
        estimator = row[index["roi_estimator"]]
        if estimator not in ("perilesional_roi", "no_mask_border"):
            raise ValueError(f"{path}: unexpected estimator for valid ROI ITA: {estimator!r}")
        origin = row[index["roi_image"]]
        if not isinstance(origin, str) or not origin:
            raise ValueError(f"{path}: valid ROI row has no image origin")
        values.append(float(ita))
        if estimator == "no_mask_border":
            fallback_count += 1
            fallback_origins[origin] = fallback_origins.get(origin, 0) + 1

    n_total = len(rows)
    n_valid = len(values)
    n_abstention = n_total - n_valid
    if sum(status_counts.values()) != n_total or not n_valid:
        raise ValueError(f"{path}: ROI row counts do not reconcile or no valid values exist")
    return {
        "dataset": dataset,
        "label": DATASETS[dataset],
        "procedure": "E5 perilesional ROI estimator",
        "n_total": n_total,
        "n_valid": n_valid,
        "n_abstention": n_abstention,
        "abstention_by_status": {key: count for key, count in status_counts.items() if key != "ok"},
        "n_fallback_no_mask_border": fallback_count,
        "fallback_origins": fallback_origins,
        "ita_values": values,
        "source": {"path": relative, "sha256": _sha256(path),
                   "last_changed_commit": _git_revision(repo, relative),
                   "code_revision": document.get("code_revision"),
                   "roi_definition": document["estimators"]["roi"]},
    }


def figure_data(repo: Path = REPO) -> dict:
    repo = Path(repo)
    thresholds = tuple(float(value) for value in FST_THRESHOLDS)
    datasets = {key: _load_dataset(repo, key, thresholds) for key in DATASETS}
    return {
        "datasets": datasets,
        "thresholds": list(thresholds),
        "reporting_guard": {"median_b_star_minimum": MIN_B_FOR_ITA, "comparison": ">="},
    }


def density_profile(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if not values or any(not math.isfinite(value) or not ITA_MIN <= value <= ITA_MAX for value in values):
        raise ValueError("violin profile requires finite ITA values in [-90, 90]")
    edges = np.arange(ITA_MIN - BIN_WIDTH_DEG / 2,
                      ITA_MAX + BIN_WIDTH_DEG / 2 + BIN_WIDTH_DEG, BIN_WIDTH_DEG)
    counts, _ = np.histogram(values, bins=edges)
    radius = int(math.ceil(3 * SMOOTH_SIGMA_DEG / BIN_WIDTH_DEG))
    offsets = np.arange(-radius, radius + 1) * BIN_WIDTH_DEG
    kernel = np.exp(-0.5 * (offsets / SMOOTH_SIGMA_DEG) ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(counts, kernel, mode="same")
    maximum = float(smoothed.max())
    if maximum <= 0:
        raise ValueError("ITA density profile has no observations")
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, smoothed / maximum


def _group_centers(thresholds: tuple[float, ...]) -> list[tuple[str, float]]:
    edges = [ITA_MIN, *reversed(thresholds), ITA_MAX]
    groups = list(reversed(BUCKETS))
    return [(group, (lower + upper) / 2)
            for group, lower, upper in zip(groups, edges[:-1], edges[1:], strict=True)]


def draw(data: dict):
    thresholds = tuple(data["thresholds"])
    positions = {dataset: index + 1 for index, dataset in enumerate(DATASETS)}
    with style.context():
        fig, axis = plt.subplots(figsize=(style.FIGURE_WIDTH_IN, 6.2),
                                 facecolor=style.COLORS["background"])
        axis.set_xlim(0.25, len(DATASETS) + 0.55)
        axis.set_ylim(ITA_MIN, ITA_MAX)
        for threshold in thresholds:
            axis.axhline(threshold, color=style.COLORS["frame"],
                         linewidth=style.LINE_WIDTHS["frame"], zorder=1)
        for group, center in _group_centers(thresholds):
            axis.text(0.42, center, group, ha="center", va="center",
                      fontsize=style.LABEL_PT, color=style.COLORS["text"])
        for dataset, panel in data["datasets"].items():
            position = positions[dataset]
            centers, density = density_profile(panel["ita_values"])
            half_width = density * MAX_HALF_WIDTH
            axis.fill_betweenx(centers, position - half_width, position + half_width,
                               where=density > 0, interpolate=True,
                               facecolor=style.COLORS["bar_light"], edgecolor=style.COLORS["accent"],
                               linewidth=style.LINE_WIDTHS["bar"], zorder=2)
            median = float(np.median(panel["ita_values"]))
            axis.plot([position - 0.12, position + 0.12], [median, median],
                      color=style.COLORS["text"], linewidth=style.LINE_WIDTHS["curve"], zorder=3)
        tick_labels = [
            f"{panel['label']}\nn ITA = {style.fmt_es(panel['n_valid'], 0)}"
            for panel in data["datasets"].values()
        ]
        axis.set_xticks(list(positions.values()), tick_labels)
        y_ticks = [ITA_MIN, *reversed(thresholds), ITA_MAX]
        axis.set_yticks(y_ticks, [style.fmt_es(value, 0) for value in y_ticks])
        axis.set_xlabel("Dataset; n de imágenes con ITA válido")
        axis.set_ylabel("ITA (grados)")
        style.style_axes(axis, "y")
        fig.text(0.06, 0.97, "Distribución de ITA por imagen · ROI perilesional E5",
                 ha="left", va="top", fontsize=style.PANEL_TITLE_PT, color=style.COLORS["text"])
        fig.subplots_adjust(left=0.13, right=0.97, top=0.91, bottom=0.17)
    return fig, axis


def _note(data: dict) -> str:
    summaries = []
    for panel in data["datasets"].values():
        summaries.append(
            f"{panel['label']}: n ITA válido={style.fmt_es(panel['n_valid'], 0)}, "
            f"n total={style.fmt_es(panel['n_total'], 0)}, abstenciones={style.fmt_es(panel['n_abstention'], 0)}, "
            f"fallback no_mask_border válido={style.fmt_es(panel['n_fallback_no_mask_border'], 0)}"
        )
    return NOTE + " " + "; ".join(summaries) + "."


def _traceability(repo: Path, data: dict) -> dict:
    script = repo / "scripts/figures/fig_t4_ita_violins.py"
    dataset_summaries = {
        dataset: {key: value for key, value in panel.items() if key != "ita_values"}
        for dataset, panel in data["datasets"].items()
    }
    return {
        "figure_id": "T4",
        "metric": "ita_degrees",
        "unit": "degrees per image",
        "procedure": "E5 perilesional ROI estimator; valid fallback no_mask_border values included and counted separately",
        "thresholds": data["thresholds"],
        "reporting_guard": data["reporting_guard"],
        "group_semantics": "ITA estimator categories I-VI; not clinically validated Fitzpatrick phototypes",
        "density_profile": {
            "method": "all valid values histogrammed at 1-degree bins and convolved with a normalized Gaussian kernel",
            "bin_width_degrees": BIN_WIDTH_DEG,
            "gaussian_sigma_degrees": SMOOTH_SIGMA_DEG,
            "observation_subsampling": "none",
        },
        "figure_script": {"path": "scripts/figures/fig_t4_ita_violins.py", "sha256": _sha256(script)},
        "datasets": dataset_summaries,
        "excluded": "PAD-UFES-20 uses Otsu and registered FST; see fig_char_ita_distribution and T3",
    }


def build(repo: Path = REPO, outdir: Path | None = None) -> list[Path]:
    repo = Path(repo)
    target = Path(outdir) if outdir is not None else (
        repo / "resultados/figuras/thesis_c1_c2_20261009_final/T4"
    )
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"{STEM}{suffix}" for suffix in suffixes if (target / f"{STEM}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    data = figure_data(repo)
    target.mkdir(parents=True, exist_ok=True)
    fig, _ = draw(data)
    try:
        paths = style.save_figure(fig, STEM, target)
    finally:
        plt.close(fig)
    (target / f"{STEM}.caption.txt").write_text(
        "Distribución continua de ITA por imagen para HAM10000, ISIC 2019 e ISIC 2020, con el ROI perilesional de E5.\n",
        encoding="utf-8")
    (target / f"{STEM}.note.txt").write_text(_note(data) + "\n", encoding="utf-8")
    (target / f"{STEM}.traceability.json").write_text(
        json.dumps(_traceability(repo, data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    for path in build(outdir=args.outdir):
        print(path)
