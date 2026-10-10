"""Figure A: CIELAB distribution of perilesional-ring pixels (HAM10000, ISIC 2019, ISIC 2020), from pod-computed histograms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402

INPUT_DIR = "resultados/insumos_pod/A"
DATASETS = (("ham10000", "HAM10000", 4539), ("isic2019", "ISIC 2019", 11805), ("isic2020", "ISIC 2020", 20704))
LINESTYLES = {"ham10000": ("-", "o"), "isic2019": ("--", "s"), "isic2020": (":", "^")}
CMAP = LinearSegmentedColormap.from_list("thesis_sequential", [style.COLORS["abstention"], style.COLORS["accent"]])
DENSITY_FLOOR_PERCENT = 0.001
NOTE = ("Población: píxeles del anillo perilesional de las imágenes cuya estimación ITA por anillo es válida (4.539 HAM10000; 11.805 ISIC 2019 y 20.704 ISIC 2020, sin las que usaron el borde de la imagen como alternativa): "
        "fuera de la máscara de lesión, dilatación de Chebyshev del 25 % del diámetro equivalente (mínimo 3 px), sin pelo, reflejos, "
        "píxeles saturados ni borde negro. Se usan todos los píxeles, sin submuestreo. (a) porcentaje de los píxeles de cada conjunto por "
        "intervalo de 2 unidades de L*; (b) porcentaje de los píxeles por celda de 4 × 4 unidades en a*–b*, escala logarítmica común.")
REQUIRED = ("dataset", "n_images", "n_pixels", "L_edges", "L_counts", "a_edges", "b_edges", "ab_counts", "code_revision",
            "masks_sha256", "ids_sha256", "canonical_input", "ring_definition")
EXPECTED_L_EDGES = tuple(range(0, 101, 2))
EXPECTED_AB_EDGES = tuple(range(-128, 129, 4))


def load(repo: Path, dataset: str, expected_images: int) -> dict:
    path = repo / INPUT_DIR / f"cielab_ring_hist_{dataset}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED if key not in doc]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if doc["dataset"] != dataset:
        raise ValueError(f"{path}: dataset {doc['dataset']!r} != {dataset!r}")
    if tuple(doc["L_edges"]) != EXPECTED_L_EDGES or tuple(doc["a_edges"]) != EXPECTED_AB_EDGES or tuple(doc["b_edges"]) != EXPECTED_AB_EDGES:
        raise ValueError(f"{path}: histogram edges differ from the agreed 0-100 (step 2) / -128-128 (step 4) grids")
    l_counts, ab = np.asarray(doc["L_counts"]), np.asarray(doc["ab_counts"])
    if l_counts.dtype.kind != "i" or ab.dtype.kind != "i" or (l_counts < 0).any() or (ab < 0).any():
        raise ValueError(f"{path}: counts must be non-negative integers")
    if len(doc["L_edges"]) != len(l_counts) + 1:
        raise ValueError(f"{path}: L edges/counts length mismatch")
    if ab.shape != (len(doc["a_edges"]) - 1, len(doc["b_edges"]) - 1):
        raise ValueError(f"{path}: ab_counts shape does not match a/b edges")
    if int(doc["n_images"]) != expected_images:
        raise ValueError(f"{path}: n_images {doc['n_images']} != expected {expected_images}")
    if int(l_counts.sum()) != int(doc["n_pixels"]) or int(ab.sum()) != int(doc["n_pixels"]):
        raise ValueError(f"{path}: histogram totals do not equal n_pixels")
    return {"doc": doc, "path": path, "L": l_counts, "ab": ab}


def build(repo: Path, outdir: Path, only: tuple[str, ...] | None = None) -> list[Path]:
    chosen = tuple(item for item in DATASETS if only is None or item[0] in only)
    if not chosen:
        raise ValueError(f"no known dataset in {only}")
    partial = len(chosen) < len(DATASETS)
    data = {name: load(repo, name, n) for name, _, n in chosen}
    with style.context():
        fig = plt.figure(figsize=(style.FIGURE_WIDTH_IN, 7.4))
        grid = fig.add_gridspec(2, len(chosen), height_ratios=(1, 1.15), hspace=0.52, wspace=0.42)
        ax_l = fig.add_subplot(grid[0, :])
        for name, label, _ in chosen:
            item = data[name]
            edges = np.asarray(item["doc"]["L_edges"], dtype=float)
            centres = (edges[:-1] + edges[1:]) / 2
            share = 100 * item["L"] / item["doc"]["n_pixels"]
            linestyle, marker = LINESTYLES[name]
            ax_l.plot(centres, share, linestyle=linestyle, marker=marker, markersize=4, color=style.COLORS["accent"],
                      markerfacecolor=style.COLORS["accent"] if name == "ham10000" else style.COLORS["background"],
                      linewidth=style.LINE_WIDTHS["curve"], label=f"{label} · {style.fmt_es(item['doc']['n_pixels'], 0)} píxeles")
        ax_l.set_xlim(0, 100)
        ax_l.set_xlabel("L* del anillo perilesional")
        ax_l.set_ylabel("% de los píxeles")
        ax_l.set_title("(a) Luminosidad", loc="left")
        ax_l.legend(frameon=False, loc="upper left")
        style.style_axes(ax_l, grid_axis="y")
        shares = {n: 100 * data[n]["ab"] / data[n]["doc"]["n_pixels"] for n, _, _ in chosen}
        vmax = max(float(s.max()) for s in shares.values())
        norm = LogNorm(vmin=DENSITY_FLOOR_PERCENT, vmax=vmax)
        image = None
        for column, (name, label, _) in enumerate(chosen):
            ax = fig.add_subplot(grid[1, column])
            a_edges, b_edges = np.asarray(data[name]["doc"]["a_edges"]), np.asarray(data[name]["doc"]["b_edges"])
            masked = np.ma.masked_less(shares[name], DENSITY_FLOOR_PERCENT)
            image = ax.pcolormesh(b_edges, a_edges, masked, cmap=CMAP, norm=norm, shading="flat")
            ax.set_xlim(-10, 60)
            ax.set_ylim(-10, 50)
            ax.set_xticks((0, 25, 50))
            ax.set_yticks((0, 20, 40))
            ax.set_xlabel("b*")
            if column == 0:
                ax.set_ylabel("a*")
            ax.set_title(f"{'(b) ' if column == 0 else ''}{label}", loc="left")
            style.style_axes(ax, grid_axis="both")
        bar = fig.colorbar(image, ax=fig.axes[1:], location="bottom", pad=0.18, fraction=0.04, aspect=45)
        bar.set_label(f"% de los píxeles por celda (escala log, {style.fmt_es(DENSITY_FLOOR_PERCENT, 3)}–{style.fmt_es(vmax, 1)})")
        bar.outline.set_edgecolor(style.COLORS["frame"])
        bar.outline.set_linewidth(style.LINE_WIDTHS["frame"])
        fig.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.2)
        stem = "fig_a_cielab_ring"
        paths = style.save_figure(fig, stem, outdir)
        plt.close(fig)
    out = Path(outdir)
    trace = {"figure": stem, "question": "Cómo se distribuyen L*, a* y b* en el anillo perilesional que alimenta la estimación ITA.",
             "status": "Cálculo adicional (histogramas calculados en el pod)", "population": "píxeles del anillo, imágenes con ITA válido",
             "partial": partial, "pending_datasets": [n for n, _, _ in DATASETS if n not in data],
             "datasets": {n: {"n_images": data[n]["doc"]["n_images"], "n_pixels": data[n]["doc"]["n_pixels"],
                              "code_revision": data[n]["doc"]["code_revision"]} for n, _, _ in chosen},
             "sources": [{"path": str(data[n]["path"].relative_to(repo)), "sha256": hashlib.sha256(data[n]["path"].read_bytes()).hexdigest()}
                         for n, _, _ in chosen],
             "script": "scripts/figures/fig_a_cielab_ring.py"}
    (out / f"{stem}.traceability.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{stem}.caption.txt").write_text("Distribución CIELAB de los píxeles del anillo perilesional en HAM10000, ISIC 2019 e ISIC 2020.\n", encoding="utf-8")
    extra = " FIGURA PARCIAL: faltan " + ", ".join(label for n, label, _ in DATASETS if n not in data) + "." if partial else ""
    (out / f"{stem}.note.txt").write_text(NOTE + extra + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados" / "figuras")
    parser.add_argument("--only", nargs="+", default=None, help="render only these datasets (marks the figure partial)")
    args = parser.parse_args()
    for path in build(REPO, args.outdir, tuple(args.only) if args.only else None):
        print(path)
