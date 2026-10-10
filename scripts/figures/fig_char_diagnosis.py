"""Characterisation: diagnostic distribution of HAM10000, ISIC 2019, ISIC 2020 and PAD-UFES-20 (2 x 2, original taxonomies)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

import fig_obj1_distributions as source  # noqa: E402
import thesis_style as style  # noqa: E402

PANELS = (("ham10000", "HAM10000"), ("isic2019", "ISIC 2019"), ("isic2020", "ISIC 2020"), ("pad_ufes_20", "PAD-UFES-20"))
DISPLAY = {"BENIGN": "BENIGNO", "MALIGNANT": "MALIGNO"}
MAX_CLASSES = 8
INSIDE_LABEL_SHARE = 70
NOTE = ("Cada panel conserva la taxonomía original de su conjunto; las categorías no son equivalentes entre paneles. "
        "El eje es el porcentaje de las imágenes del conjunto completo (denominador = N del título de cada panel) y el número junto "
        "a cada barra es el conteo de imágenes. Describe la composición de los conjuntos, no la prevalencia poblacional.")


def counts(repo: Path, dataset: str) -> dict[str, int]:
    if dataset == "pad_ufes_20":
        data = dict(source.pad.EXPECTED_DIAGNOSTIC_COUNTS)
    else:
        data = source.count_e5(repo, dataset)
    total = source.TOTALS[dataset]
    if sum(data.values()) != total:
        raise ValueError(f"{dataset}: class counts sum {sum(data.values())} != documented total {total}")
    return data


def build(repo: Path, outdir: Path) -> list[Path]:
    data = {dataset: counts(repo, dataset) for dataset, _ in PANELS}
    with style.context():
        fig, axes = plt.subplots(2, 2, figsize=(style.FIGURE_WIDTH_IN, 6.2), gridspec_kw={"hspace": 0.5, "wspace": 0.55})
        for ax, (dataset, label) in zip(axes.flat, PANELS, strict=True):
            ordered = sorted(data[dataset].items(), key=lambda item: -item[1])
            total = sum(count for _, count in ordered)
            ys = range(len(ordered))
            ax.barh(list(ys), [100 * count / total for _, count in ordered], height=0.62, color=style.COLORS["bar_light"],
                    edgecolor=style.COLORS["accent"], linewidth=style.LINE_WIDTHS["bar"])
            for y, (_, count) in zip(ys, ordered, strict=True):
                share = 100 * count / total
                inside = share > INSIDE_LABEL_SHARE
                ax.text(share - 1.5 if inside else share + 1.5, y, style.fmt_es(count, 0), va="center",
                        ha="right" if inside else "left")
            ax.set_yticks(list(ys), [DISPLAY.get(name, name) for name, _ in ordered])
            ax.set_ylim(MAX_CLASSES - 0.4, -0.6)  # same slot height in every panel keeps bar thickness uniform
            ax.set_xlim(0, 100)
            ax.set_xticks((0, 50, 100), ("0", "50", "100"))
            ax.set_xlabel("% de las imágenes")
            ax.set_title(f"{label} (N = {style.fmt_es(total, 0)})", loc="left")
            style.style_axes(ax, grid_axis="x")
        fig.subplots_adjust(left=0.13, right=0.97, top=0.94, bottom=0.09)
        stem = "fig_char_diagnosis_distribution"
        paths = style.save_figure(fig, stem, outdir)
        plt.close(fig)
    out = Path(outdir)
    srcs = [repo / source.ita.E5_DIR / f"e5_fitzpatrick_{d}.json" for d in ("ham10000", "isic2019", "isic2020")]
    trace = {"figure": stem, "question": "Cómo se reparten los diagnósticos en cada conjunto, con su taxonomía original.",
             "status": "Artefactos verificados", "totals": {d: sum(data[d].values()) for d, _ in PANELS},
             "counts": data, "sources": [{"path": str(p.relative_to(repo)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in srcs],
             "pad_source": "src/dermssl/pad_ufes.py EXPECTED_DIAGNOSTIC_COUNTS (conteos verificados contra el manifiesto en docs/30)",
             "script": "scripts/figures/fig_char_diagnosis.py"}
    (out / f"{stem}.traceability.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{stem}.caption.txt").write_text("Distribución diagnóstica de HAM10000, ISIC 2019, ISIC 2020 y PAD-UFES-20.\n", encoding="utf-8")
    (out / f"{stem}.note.txt").write_text(NOTE + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados" / "figuras")
    args = parser.parse_args()
    for path in build(REPO, args.outdir):
        print(path)
