"""Objective 1: recorded PAD phototypes and image-matched, masked ITA values."""

from __future__ import annotations

import argparse
import hashlib
import json
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

import beautiful as bf  # noqa: E402
import fig1_ita as ita  # noqa: E402
from fig_obj1_diagnosis_tone import SKIN_FILLS  # noqa: E402

SOURCE = REPO / "resultados/fst/pad_ufes_t1/pad_ufes_fst_validation.json"
STEM = "fig_obj1_pad_fst_ita"
BUCKETS = ita.BUCKETS
BACKGROUND, PANEL, INK, MUTED = "#FFFFFF", "#FFFFFF", "#302C29", "#645D58"
BORDER, GRID = "#AAA39E", "#E5DED8"
ITA_BINS = np.arange(-90, 91, 10)
HISTOGRAM_HEADROOM = 1.1


def load_data(source: Path) -> dict:
    """Use only PAD rows with a recorded FST; the same row supplies masked ITA."""
    raw = json.loads(Path(source).read_text(encoding="utf-8"))
    records = raw["records"]
    n_total, n_missing = raw["n_images"], raw["n_without_fst"]
    if n_total != len(records) + n_missing or n_missing < 0:
        raise ValueError("PAD FST totals do not reconcile")
    counts = dict.fromkeys(BUCKETS, 0)
    itas: dict[str, list[float]] = {bucket: [] for bucket in BUCKETS}
    seen = set()
    for row in records:
        image_id, fst = row["img_id"], row["actual_fst"]
        if not image_id or image_id in seen or type(fst) is not int or fst not in range(1, 7):
            raise ValueError("PAD FST IDs must be unique and types must be I–VI")
        seen.add(image_id)
        bucket = BUCKETS[fst - 1]
        counts[bucket] += 1
        estimate = row["masked"]
        if estimate["status"] != "ok":
            continue
        value = estimate["ita"]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not -90 <= value <= 90:
            raise ValueError(f"Invalid masked ITA for PAD image {image_id}")
        itas[bucket].append(float(value))
    n_valid = sum(map(len, itas.values()))
    return {"n_total": n_total, "n_missing_fst": n_missing, "fst_counts": counts,
            "ita_by_fst": itas, "n_ita_valid": n_valid,
            "n_ita_unavailable": len(records) - n_valid}


def _axes_style(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _phototypes(ax, data: dict) -> None:
    counts = [data["fst_counts"][bucket] for bucket in BUCKETS]
    positions = np.arange(len(BUCKETS))
    bars = ax.bar(positions, counts, width=0.78, color=SKIN_FILLS,
                  edgecolor=[ita._mix(color, "black", 0.7) for color in SKIN_FILLS],
                  linewidth=0.8)
    for bar, count in zip(bars, counts, strict=True):
        ax.annotate(bf.fmt_es(count, 0), (bar.get_x() + bar.get_width() / 2, count),
                    xytext=(0, 5), textcoords="offset points", ha="center", color=INK, fontsize=12)
    ax.set_xticks(positions, BUCKETS)
    ax.set_ylim(0, max(counts) * 1.16)
    ax.set(xlabel="Fototipo Fitzpatrick registrado", ylabel="Número de imágenes")
    _axes_style(ax)


def _ita_histograms(ax, data: dict) -> None:
    stacked = np.zeros(len(ITA_BINS) - 1, dtype=int)
    for bucket, color in zip(BUCKETS, SKIN_FILLS, strict=True):
        values = data["ita_by_fst"][bucket]
        histogram, _ = np.histogram(values, bins=ITA_BINS)
        ax.bar(ITA_BINS[:-1], histogram, width=np.diff(ITA_BINS), align="edge", bottom=stacked,
               color=color, edgecolor=ita._mix(color, "black", 0.7), linewidth=0.35)
        stacked += histogram
    ax.set(xlabel="ITA estimado del fondo (°)", ylabel="Número de imágenes")
    ax.set_xlim(ITA_BINS[0], ITA_BINS[-1])
    ax.set_ylim(0, max(1, stacked.max()) * HISTOGRAM_HEADROOM)
    _axes_style(ax)
    handles = [Patch(facecolor=color, edgecolor=ita._mix(color, "black", 0.7),
                     label=f"{bucket} · n = {bf.fmt_es(len(data['ita_by_fst'][bucket]), 0)}")
               for bucket, color in zip(BUCKETS, SKIN_FILLS, strict=True)]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=2, fontsize=10)


def build(source: Path = SOURCE, outdir: Path | None = None) -> list[Path]:
    source = Path(source)
    data = load_data(source)
    target = Path(outdir) if outdir else REPO / "resultados/figuras"
    bf.apply_style(width=16)
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "svg.fonttype": "none"})
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BACKGROUND)
    _phototypes(ax_a, data)
    _ita_histograms(ax_b, data)
    ax_a.set_title("(a) Fototipo registrado en PAD-UFES-20", loc="left", color=INK, fontsize=17, pad=17)
    ax_b.set_title("(b) ITA por fototipo registrado", loc="left", color=INK, fontsize=17, pad=17)
    note = (f"N = {bf.fmt_es(data['n_total'], 0)}; FST registrado: "
            f"{bf.fmt_es(sum(data['fst_counts'].values()), 0)}; sin FST: "
            f"{bf.fmt_es(data['n_missing_fst'], 0)}; ITA válido con FST: "
            f"{bf.fmt_es(data['n_ita_valid'], 0)}.\n"
            "FST del metadato: autor de la asignación no verificado; ITA en fondo por Otsu, "
            "no validado como fototipo. V–VI: conteos descriptivos, sin inferencia.")
    fig.text(0.055, 0.045, note, color=MUTED, fontsize=10, va="bottom")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.22, top=0.88, wspace=0.24)
    paths = bf.save_all(fig, STEM, target)
    bf.close(fig)
    summary = {key: value for key, value in data.items() if key != "ita_by_fst"}
    summary["ita_counts_by_fst"] = {bucket: len(values) for bucket, values in data["ita_by_fst"].items()}
    summary["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (target / f"{STEM}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    for path in build(args.source, args.outdir):
        print(path)
