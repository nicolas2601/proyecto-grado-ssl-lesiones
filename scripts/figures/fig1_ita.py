"""Figure 1: ITA (individual typology angle) of the perilesional ROI per dataset, with the abstention shown.

Panel (a): distribution of ``roi_ita`` (images with an ITA only) over the Fitzpatrick threshold bands.
Panel (b): share of ALL images per bucket I-VI plus abstention. Every number is read from the committed
``e5_fitzpatrick_*.json`` files; nothing is typed by hand.

    PYTHONPATH=$HOME/work/tmp/mpl ~/.cache/tvenv/bin/python scripts/figures/fig1_ita.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import beautiful as bf  # noqa: E402

DATASETS = ("ham10000", "isic2019", "isic2020")
DATASET_LABELS = {"ham10000": "HAM10000", "isic2019": "ISIC 2019", "isic2020": "ISIC 2020"}
BUCKETS = ("I", "II", "III", "IV", "V", "VI")
E5_DIR = Path("resultados") / "banco_completo" / "e5"
STEM = "fig1_ita"
OUTDIR = Path("resultados") / "figuras"
NOTE = "ITA: estimación no validada de fototipo; colores: convención visual, no diagnóstico clínico"
ABSTENTION = "abstención"
ABSTENTION_KEY = "abstention"

WIDTH_IN, HEIGHT_IN = 16, 8
TEXT_PT = 17  # apply_style(16) gives 32 pt, too large for a 2-panel 16 x 8 in figure; sizes are lowered after it
MIN_LABEL_PCT = 11.0
GAP_PCT = 0.35  # white gap between stacked segments, about 2 px at 300 dpi on this layout
# Visual convention for ordinal ITA estimates only, NOT validated Fitzpatrick phenotypes.
# Drawn without any reference photograph or racial descriptor. The violins stay dataset-coloured.
BUCKET_FILLS = ("#c2a68e", "#b88a64", "#ad713b", "#89562f", "#664126", "#3b2819")
ABSTENTION_FILL, ABSTENTION_EDGE = "#7e7e7e", "#4e4e4e"
ABSTENTION_HATCH = "///"  # print/achromatopsia channel; gray alone can match the ITA-III ramp step
NEUTRAL_VIOLIN_FILL = "#d8d8d8"  # the mark is a dataset distribution, not a Fitzpatrick class
NEUTRAL_VIOLIN_EDGE = "#666666"
NEUTRAL_BOX_FILL = "#777777"
EDGE_DARKEN = 0.72
BAND_ALPHA = 0.14


def source_files(repo: Path) -> list[Path]:
    return [Path(repo) / E5_DIR / f"e5_fitzpatrick_{d}.json" for d in DATASETS]


def bucket_of(ita: float, thresholds: list[float]) -> str:
    """Groh buckets: I if ITA > t0, II if > t1, ...; a value on a threshold goes to the darker bucket."""
    for bucket, threshold in zip(BUCKETS, thresholds):
        if ita > threshold:
            return bucket
    return BUCKETS[-1]


def _load(path: Path) -> tuple[dict, list[dict]]:
    raw = json.loads(path.read_text())
    cols = raw["per_image"]["columns"]
    return raw, [dict(zip(cols, row)) for row in raw["per_image"]["rows"]]


def _dataset_data(rows: list[dict]) -> dict:
    itas = sorted(r["roi_ita"] for r in rows if r["roi_status"] == "ok")
    q25, median, q75 = statistics.quantiles(itas, n=4, method="inclusive")
    counts = {b: sum(r["roi_bucket"] == b for r in rows) for b in BUCKETS}
    reasons = sorted({r["roi_status"] for r in rows} - {"ok"})
    n_total = len(rows)
    n_abstention = n_total - len(itas)
    return {
        "n_total": n_total,
        "n_ita": len(itas),
        "itas": itas,
        "quantiles": {"min": itas[0], "q25": q25, "median": median, "q75": q75, "max": itas[-1]},
        "bucket_counts": counts,
        "n_abstention": n_abstention,
        "abstention_by_reason": {s: sum(r["roi_status"] == s for r in rows) for s in reasons},
        "fractions": {**{b: c / n_total for b, c in counts.items()}, "abstention": n_abstention / n_total},
    }


def figure_data(repo: Path) -> dict:
    """Counts, quantiles and fractions per dataset, plus the thresholds, all read from the JSON files."""
    data: dict = {}
    thresholds = None
    for dataset, path in zip(DATASETS, source_files(repo)):
        raw, rows = _load(path)
        if thresholds not in (None, raw["fst_thresholds"]):
            raise ValueError(f"{path.name}: fst_thresholds differ across datasets")
        thresholds = raw["fst_thresholds"]
        data[dataset] = _dataset_data(rows)
    data["thresholds"] = thresholds
    return data


def _mix(color: str, other: str, share: float) -> tuple[float, float, float]:
    a, b = mcolors.to_rgb(color), mcolors.to_rgb(other)
    return tuple(share * x + (1 - share) * y for x, y in zip(a, b))


def bucket_palette() -> tuple[list, list]:
    """Stable ordinal cream-to-dark-brown ramp for visual ITA buckets I–VI; these are not clinical skin types."""
    fills = [mcolors.to_rgb(color) for color in BUCKET_FILLS]
    edges = [_mix(color, "black", EDGE_DARKEN) for color in BUCKET_FILLS]
    return fills, edges


def _relative_luminance(color) -> float:
    """WCAG relative luminance; select ink for readable direct labels, not based on an eyeballed threshold."""
    rgb = mcolors.to_rgb(color)
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return sum(weight * value for weight, value in zip((0.2126, 0.7152, 0.0722), linear))


def _text_color(fill) -> str:
    luminance = _relative_luminance(fill)
    dark = _relative_luminance("#111111")
    white_contrast = 1.05 / (luminance + 0.05)
    dark_contrast = (luminance + 0.05) / (dark + 0.05)
    return "white" if white_contrast >= dark_contrast else "#111111"


def _band_edges(thresholds: list[float], lo: float, hi: float) -> list[tuple[str, float, float]]:
    uppers = [hi, *thresholds]
    lowers = [*thresholds, lo]
    return [(b, low, up) for b, low, up in zip(BUCKETS, lowers, uppers)]


def _panel_distribution(ax, data: dict, fills) -> None:
    thresholds = data["thresholds"]
    values = [data[d]["itas"] for d in DATASETS]
    lo = min(min(v) for v in values) - 5
    hi = max(max(v) for v in values) + 5
    for (bucket, low, up), fill in zip(_band_edges(thresholds, lo, hi), fills):
        ax.axhspan(low, up, color=fill, alpha=BAND_ALPHA, linewidth=0, zorder=0)
        ax.text(1.01, (max(low, lo) + min(up, hi)) / 2, bucket, transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=TEXT_PT, color="#333333")
    positions = range(1, len(DATASETS) + 1)
    parts = ax.violinplot(values, positions=positions, widths=0.8, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(NEUTRAL_VIOLIN_FILL)
        body.set_edgecolor(NEUTRAL_VIOLIN_EDGE)
        body.set_alpha(0.42)
        body.set_linewidth(1.0)
    box = ax.boxplot(values, positions=list(positions), widths=0.14, showfliers=False, patch_artist=True,
                     medianprops={"color": "white", "linewidth": 1.6}, whiskerprops={"color": NEUTRAL_VIOLIN_EDGE, "linewidth": 1.0},
                     capprops={"color": NEUTRAL_VIOLIN_EDGE, "linewidth": 1.0}, boxprops={"linewidth": 1.0})
    for patch in box["boxes"]:
        patch.set(facecolor=NEUTRAL_BOX_FILL, edgecolor=NEUTRAL_VIOLIN_EDGE)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{DATASET_LABELS[d].replace(' ', chr(10))}\nn = {bf.fmt_es(data[d]['n_ita'], 0)}" for d in DATASETS])
    ax.set_ylim(lo, hi)
    ax.tick_params(axis="x", labelsize=TEXT_PT - 3)
    ax.set_ylabel("ITA de la ROI perilesional (°)")
    ax.yaxis.set_major_formatter(lambda v, _: bf.fmt_es(v, 0))
    bf.style_axes(ax)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(False, axis="x")


def _panel_buckets(ax, data: dict, fills, edges) -> None:
    ypos = list(range(len(DATASETS)))[::-1]
    colors = [*zip(fills, edges), (ABSTENTION_FILL, ABSTENTION_EDGE)]
    names = [*BUCKETS, ABSTENTION_KEY]
    for y, dataset in zip(ypos, DATASETS):
        left = 0.0
        for name, (fill, edge) in zip(names, colors):
            pct = data[dataset]["fractions"][name] * 100
            ax.barh(y, max(pct - GAP_PCT, 0), left=left + GAP_PCT / 2, height=0.6, color=fill, edgecolor=edge,
                    linewidth=1.0, zorder=3, hatch=ABSTENTION_HATCH if name == ABSTENTION_KEY else None)
            if pct >= MIN_LABEL_PCT:
                ax.text(left + pct / 2, y, f"{bf.fmt_es(pct, 0)} %", ha="center", va="center", zorder=4,
                        fontsize=TEXT_PT - 2, color=_text_color(fill))
            left += pct
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{DATASET_LABELS[d]}\nN = {bf.fmt_es(data[d]['n_total'], 0)}" for d in DATASETS])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de todas las imágenes")
    ax.xaxis.set_major_formatter(lambda v, _: bf.fmt_es(v, 0))
    bf.style_axes(ax)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(False, axis="y")
    handles = [mpatches.Patch(facecolor=f, edgecolor=e, linewidth=1.0,
                              hatch=ABSTENTION_HATCH if name == ABSTENTION_KEY else None) for name, (f, e) in zip(names, colors)]
    bf.top_legend(ax, handles, [*BUCKETS, ABSTENTION], ncol=7, handlelength=1.0, handletextpad=0.4, columnspacing=0.9)
    ax.text(0.0, -0.2, NOTE, transform=ax.transAxes, fontsize=TEXT_PT - 2, style="italic", va="top")


def build(repo: Path, outdir: Path | None = None) -> list[Path]:
    repo = Path(repo)
    data = figure_data(repo)
    bf.apply_style(width=WIDTH_IN)
    plt.rcParams.update({k: TEXT_PT for k in ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    fills, edges = bucket_palette()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(WIDTH_IN, HEIGHT_IN), gridspec_kw={"width_ratios": [1.15, 1.2], "wspace": 0.5})
    _panel_distribution(ax_a, data, fills)
    _panel_buckets(ax_b, data, fills, edges)
    for ax, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax.text(-0.02, 1.10, letter, transform=ax.transAxes, fontsize=TEXT_PT + 2, fontweight="bold", va="bottom", ha="right")
    paths = bf.save_all(fig, STEM, Path(outdir) if outdir else repo / OUTDIR)
    svg = next(path for path in paths if path.suffix == ".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
                   encoding="utf-8")
    bf.close(fig)
    return paths


if __name__ == "__main__":
    for written in build(REPO):
        print(written)
