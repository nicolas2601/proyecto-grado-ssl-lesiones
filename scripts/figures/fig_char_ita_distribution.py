"""Characterisation: ITA distribution (10-degree bins) of HAM10000, ISIC 2019, ISIC 2020 and PAD-UFES-20, stacked by ITA estimator group."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import fig_obj1_ita_estimates as ring  # noqa: E402
import thesis_style as style  # noqa: E402
from dermssl.fitzpatrick import FST_THRESHOLDS  # noqa: E402

BUCKETS = ring.BUCKETS
BIN_EDGES = np.arange(-90, 91, 10)
BIN_WIDTH_DEG = 10
PAD_SOURCE = "resultados/fst/pad_ufes_t1/pad_ufes_fst_validation.json"
PANELS = (("ham10000", "HAM10000", "ROI perilesional"), ("isic2019", "ISIC 2019", "ROI perilesional"),
          ("isic2020", "ISIC 2020", "ROI perilesional"), ("pad_ufes_20", "PAD-UFES-20", "fondo Otsu"))
NOTE = ("Histogramas apilados con intervalos contiguos de 10°; los colores son las categorías del estimador ITA (umbrales de Groh "
        "fijados en el código: 40, 23, 12, 0, −25), no fototipos clínicos Fitzpatrick. HAM10000, ISIC 2019 e ISIC 2020 usan la ROI "
        "perilesional; PAD-UFES-20 usa el fondo por Otsu y solo representa las imágenes con fototipo registrado en los metadatos. "
        "Las abstenciones (sin ITA válido) no aparecen en el histograma y se informan en cada panel. Cada panel tiene su propio eje vertical.")


def load_pad(repo: Path) -> dict:
    path = repo / PAD_SOURCE
    doc = json.loads(path.read_text(encoding="utf-8"))
    values: dict[str, list[float]] = {bucket: [] for bucket in BUCKETS}
    n_abstention = 0
    seen: set[str] = set()
    for row in doc["records"]:
        if not row["img_id"] or row["img_id"] in seen:
            raise ValueError("PAD image IDs must be unique")
        seen.add(row["img_id"])
        estimate = row["masked"]
        if estimate["status"] != "ok":
            n_abstention += 1
            continue
        angle = estimate["ita"]
        if not isinstance(angle, (int, float)) or not math.isfinite(angle) or not -90 <= angle <= 90:
            raise ValueError(f"Invalid PAD ITA for {row['img_id']}")
        values[ring.original.bucket_of(angle, list(FST_THRESHOLDS))].append(float(angle))
    n_total = len(doc["records"])
    if n_total + doc["n_without_fst"] != doc["n_images"]:
        raise ValueError("PAD totals do not reconcile")
    counts = {b: len(v) for b, v in values.items()}
    return {"dataset": "pad_ufes_20", "n_total": n_total, "n_valid": n_total - n_abstention, "n_abstention": n_abstention,
            "counts": counts, "values": values, "n_excluded_without_fst": doc["n_without_fst"], "path": path}


def load_all(repo: Path) -> dict[str, dict]:
    data = {name: ring.load_dataset(repo, name) for name in ("ham10000", "isic2019", "isic2020")}
    data["pad_ufes_20"] = load_pad(repo)
    return data


def build(repo: Path, outdir: Path) -> list[Path]:
    data = load_all(repo)
    with style.context():
        fig, axes = plt.subplots(2, 2, figsize=(style.FIGURE_WIDTH_IN, 7.4), gridspec_kw={"hspace": 1.0, "wspace": 0.42})
        for ax, (dataset, label, procedure) in zip(axes.flat, PANELS, strict=True):
            item = data[dataset]
            base = np.zeros(len(BIN_EDGES) - 1, dtype=int)
            for bucket, color in zip(BUCKETS, style.ITA_COLORS, strict=True):
                hist, _ = np.histogram(item["values"][bucket], bins=BIN_EDGES)
                ax.bar(BIN_EDGES[:-1], hist, width=BIN_WIDTH_DEG, align="edge", bottom=base, color=color,
                       edgecolor=style.COLORS["accent"], linewidth=style.LINE_WIDTHS["bar"] * 0.5)
                base += hist
            ax.set_xlim(-90, 90)
            ax.set_xticks((-90, -45, 0, 45, 90), ("−90", "−45", "0", "45", "90"))
            ax.set_ylim(0, max(1, base.max()) * 1.12)
            ax.set_xlabel("ITA (°)")
            ax.set_title(f"{label}\n{procedure}\nválidos {style.fmt_es(item['n_valid'], 0)}\nabstenciones {style.fmt_es(item['n_abstention'], 0)}",
                         loc="left", linespacing=1.2)
            style.style_axes(ax, grid_axis="y")
        axes[0, 0].set_ylabel("Imágenes")
        axes[1, 0].set_ylabel("Imágenes")
        handles = [Patch(facecolor=c, edgecolor=style.COLORS["accent"], linewidth=style.LINE_WIDTHS["bar"], label=b)
                   for b, c in zip(BUCKETS, style.ITA_COLORS, strict=True)]
        fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, title="Categoría del estimador ITA",
                   handlelength=1.2, columnspacing=1.0)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.87, bottom=0.15)
        stem = "fig_char_ita_distribution"
        paths = style.save_figure(fig, stem, outdir)
        plt.close(fig)
    out = Path(outdir)
    srcs = [repo / ring.original.E5_DIR / f"e5_fitzpatrick_{d}.json" for d in ("ham10000", "isic2019", "isic2020")] + [data["pad_ufes_20"]["path"]]
    trace = {"figure": stem, "question": "Cómo se distribuye el ITA estimado en cada conjunto y cuántas imágenes se abstienen.",
             "status": "Artefactos verificados", "bin_width_deg": BIN_WIDTH_DEG, "thresholds": list(FST_THRESHOLDS),
             "panels": {d: {"procedure": proc, "n_total": data[d]["n_total"], "n_valid": data[d]["n_valid"], "n_abstention": data[d]["n_abstention"],
                            "counts_by_group": data[d]["counts"]} for d, _, proc in PANELS},
             "pad_scope": f"solo imágenes con fototipo registrado ({data['pad_ufes_20']['n_total']}); {data['pad_ufes_20']['n_excluded_without_fst']} sin fototipo quedan fuera",
             "sources": [{"path": str(p.relative_to(repo)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in srcs],
             "script": "scripts/figures/fig_char_ita_distribution.py"}
    (out / f"{stem}.traceability.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{stem}.caption.txt").write_text("Distribución del ITA estimado en HAM10000, ISIC 2019, ISIC 2020 y PAD-UFES-20.\n", encoding="utf-8")
    (out / f"{stem}.note.txt").write_text(NOTE + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados" / "figuras")
    args = parser.parse_args()
    for path in build(REPO, args.outdir):
        print(path)
