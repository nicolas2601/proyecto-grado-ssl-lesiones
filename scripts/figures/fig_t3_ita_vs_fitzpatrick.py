"""T3: exact accuracy of the ITA estimator against the PAD-UFES-20 recorded Fitzpatrick phototype, vs the majority baseline."""

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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402

SOURCE = "resultados/fst/pad_ufes_t1/pad_ufes_fst_validation.json"
CONDITIONS = (("whole_image", "Imagen completa"), ("masked", "Fondo por Otsu"))
NOTE = ("Exploratorio. Solo imágenes de PAD-UFES-20 con fototipo registrado en los metadatos (el autor de la asignación no está "
        "verificado); se excluyen las que el estimador no pudo evaluar. Barras: IC 95 % bootstrap agrupado por paciente, descriptivo. "
        "La línea de mayoría siempre predice el fototipo II. Este resultado no valida las máscaras.")


def load(repo: Path) -> dict:
    path = repo / SOURCE
    doc = json.loads(path.read_text(encoding="utf-8"))
    for key, _ in CONDITIONS:
        m = doc["metrics"][key]
        low, high = m["ci95"]["accuracy"]
        if not low <= m["accuracy"] <= high:
            raise ValueError(f"{path}: {key} accuracy outside its CI")
        if m["beats_majority"] != (m["accuracy"] > m["majority_baseline"]["accuracy"]):
            raise ValueError(f"{path}: {key} beats_majority flag disagrees with the numbers")
    return {"doc": doc, "path": path}


def build(repo: Path, outdir: Path) -> list[Path]:
    src = load(repo)
    doc = src["doc"]
    with style.context():
        fig, ax = plt.subplots(figsize=(style.FIGURE_WIDTH_IN, 3.1))
        for y, (key, label) in enumerate(CONDITIONS):
            m = doc["metrics"][key]
            acc, (lo, hi) = m["accuracy"], m["ci95"]["accuracy"]
            base = m["majority_baseline"]["accuracy"]
            ax.errorbar(acc, y - 0.12, xerr=[[acc - lo], [hi - acc]], linestyle="none", marker="s", markersize=7,
                        color=style.COLORS["accent"], markerfacecolor=style.COLORS["accent"], elinewidth=style.LINE_WIDTHS["curve"],
                        capsize=3, capthick=style.LINE_WIDTHS["curve"])
            ax.plot([base], [y + 0.12], linestyle="none", marker="o", markersize=7, markeredgecolor=style.COLORS["accent"],
                    markerfacecolor=style.COLORS["bar_light"], markeredgewidth=style.LINE_WIDTHS["bar"])
            ax.annotate(style.fmt_es(acc, 3), (acc, y - 0.12), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom")
            ax.annotate(style.fmt_es(base, 3), (base, y + 0.12), xytext=(0, -9), textcoords="offset points", ha="center", va="top")
            ax.text(1.02, y, f"n = {style.fmt_es(m['n_evaluable'], 0)}\nsin est. = {style.fmt_es(m['n_unavailable'], 0)}",
                    transform=ax.get_yaxis_transform(), va="center", ha="left")
        ax.set_ylim(len(CONDITIONS) - 0.4, -0.6)
        ax.set_yticks(range(len(CONDITIONS)), [label for _, label in CONDITIONS])
        ax.set_xlim(0, 1)
        ax.set_xticks((0, 0.5, 1), [style.fmt_es(v, 1) for v in (0, 0.5, 1)])
        ax.set_xlabel("Exactitud exacta frente al fototipo registrado")
        style.style_axes(ax, grid_axis="x")
        handles = [plt.Line2D([], [], linestyle="none", marker="s", markersize=7, color=style.COLORS["accent"], label="ITA (IC 95 %)"),
                   plt.Line2D([], [], linestyle="none", marker="o", markersize=7, markeredgecolor=style.COLORS["accent"],
                              markerfacecolor=style.COLORS["bar_light"], label="Mayoría (fototipo II)")]
        fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False)
        fig.subplots_adjust(left=0.26, right=0.72, top=0.95, bottom=0.34)
        stem = "fig_t3_ita_vs_fitzpatrick"
        paths = style.save_figure(fig, stem, outdir)
        plt.close(fig)
    out = Path(outdir)
    trace = {"figure": stem, "question": "¿El estimador ITA reproduce el fototipo registrado mejor que predecir siempre el fototipo mayoritario?",
             "status": "Artefactos verificados", "source": SOURCE, "source_sha256": hashlib.sha256(src["path"].read_bytes()).hexdigest(),
             "code_revision": doc["code_revision"], "registration_commit": doc["registration_commit"], "exploratory": doc["exploratory"],
             "n_images": doc["n_images"], "n_without_fst": doc["n_without_fst"],
             "conditions": {k: {"n_evaluable": doc["metrics"][k]["n_evaluable"], "n_unavailable": doc["metrics"][k]["n_unavailable"],
                                "accuracy": doc["metrics"][k]["accuracy"], "majority": doc["metrics"][k]["majority_baseline"]["accuracy"],
                                "beats_majority": doc["metrics"][k]["beats_majority"]} for k, _ in CONDITIONS},
             "discrepancy": "La literatura citada en docs reporta baseline de mayoría 51,3 % (Alipour et al.); este artefacto da 59,4 % y 59,8 % sobre las filas evaluables con fototipo registrado. La figura usa solo el artefacto.",
             "script": "scripts/figures/fig_t3_ita_vs_fitzpatrick.py"}
    (out / f"{stem}.traceability.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{stem}.caption.txt").write_text("Exactitud exacta del estimador ITA frente al fototipo Fitzpatrick registrado en PAD-UFES-20, "
                                              "con la línea base de predecir siempre el fototipo mayoritario.\n", encoding="utf-8")
    (out / f"{stem}.note.txt").write_text(NOTE + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados" / "figuras")
    args = parser.parse_args()
    for path in build(REPO, args.outdir):
        print(path)
