"""C3: row-normalised confusion matrix of one frozen model, from the committed confusion JSON."""

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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402

CONFUSION_DIR = "resultados/banco_completo/confusion"
CELLS = {
    "ham10000": ("ham10000/b71__congelado.json", "HAM10000"),
    "isic2019": ("isic2019/b71_original__congelado.json", "ISIC 2019"),
    "isic2020": ("isic2020/b71_original__congelado.json", "ISIC 2020"),
}
CMAP = LinearSegmentedColormap.from_list("thesis_sequential", [style.COLORS["abstention"], style.COLORS["accent"]])
DARK_CELL_FRACTION = 0.75
NOTE = ("Filas: etiqueta real; columnas: predicción. Cada fila suma 100 %. Predicciones por argmax de las "
        "probabilidades publicadas; los conteos entre paréntesis son imágenes; el color y el primer número de cada celda son el porcentaje de la fila.")


def load_cell(repo: Path, key: str) -> dict:
    relative, label = CELLS[key]
    path = repo / CONFUSION_DIR / relative
    doc = json.loads(path.read_text(encoding="utf-8"))
    counts = np.asarray(doc["conteos"], dtype=int)
    rows = np.asarray(doc["normalizada_por_fila"], dtype=float)
    classes = list(doc["clases"])
    if counts.shape != (len(classes), len(classes)) or rows.shape != counts.shape:
        raise ValueError(f"{path}: matrix shape does not match the class list")
    if int(counts.sum()) != int(doc["n"]):
        raise ValueError(f"{path}: counts sum {counts.sum()} != n {doc['n']}")
    support = counts.sum(axis=1)
    recomputed = counts / np.where(support == 0, 1, support)[:, None]
    if not np.allclose(recomputed[support > 0], rows[support > 0], atol=1e-9):
        raise ValueError(f"{path}: normalised rows differ from counts")
    return {"label": label, "doc": doc, "counts": counts, "rows": rows, "classes": classes,
            "support": support, "path": path, "relative": f"{CONFUSION_DIR}/{relative}"}


def build(repo: Path, key: str, outdir: Path) -> list[Path]:
    cell = load_cell(repo, key)
    classes, counts, rows, support = cell["classes"], cell["counts"], cell["rows"], cell["support"]
    size = len(classes)
    with style.context():
        fig, ax = plt.subplots(figsize=(style.FIGURE_WIDTH_IN, max(0.6 * size + 1.45, 3.6)))
        image = ax.imshow(rows * 100, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
        for i in range(size):
            for j in range(size):
                dark = rows[i, j] >= DARK_CELL_FRACTION
                ax.text(j, i, f"{style.fmt_es(rows[i, j] * 100, 1)}\n({style.fmt_es(counts[i, j], 0)})",
                        ha="center", va="center", linespacing=1.1,
                        color=style.COLORS["background"] if dark else style.COLORS["text"])
        ax.set_xticks(range(size), classes)
        ax.set_yticks(range(size), classes)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Etiqueta real")
        ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
        ax.grid(which="minor", color=style.COLORS["frame"], linewidth=style.LINE_WIDTHS["frame"])
        ax.tick_params(which="both", length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_color(style.COLORS["frame"])
            spine.set_linewidth(style.LINE_WIDTHS["frame"])
        ax.text(1.02, 1.01, "soporte", transform=ax.transAxes, ha="left", va="bottom")
        for i in range(size):
            ax.text(1.02, 1 - (i + 0.5) / size, f"n = {style.fmt_es(support[i], 0)}",
                    transform=ax.transAxes, ha="left", va="center")
        bar = fig.colorbar(image, ax=ax, location="bottom", pad=0.06, fraction=0.05, aspect=40, ticks=[0, 25, 50, 75, 100])
        bar.set_label("Porcentaje de la clase real (%); entre paréntesis, imágenes")
        bar.outline.set_edgecolor(style.COLORS["frame"])
        bar.outline.set_linewidth(style.LINE_WIDTHS["frame"])
        bar.ax.tick_params(length=0)
        fig.subplots_adjust(left=0.115, right=0.855, top=0.88, bottom=0.17)
        stem = f"fig_c3_confusion_{key}"
        paths = style.save_figure(fig, stem, outdir)
        plt.close(fig)
    doc = cell["doc"]
    trace = {
        "figure": stem, "question": "Qué clases confunde B71 y con qué frecuencia, normalizado por clase real.",
        "dataset": cell["label"], "model": doc["modelo"], "condition": doc["condicion"], "n": int(doc["n"]),
        "classes": classes, "support": [int(v) for v in support], "mcc": doc.get("mcc"),
        "source": cell["relative"], "source_sha256": hashlib.sha256(cell["path"].read_bytes()).hexdigest(),
        "upstream": doc.get("fuente"), "origin": doc.get("origen"), "status": "Artefactos verificados",
        "checks": ["counts sum == n", "rows recomputed from counts match normalizada_por_fila"],
        "script": "scripts/figures/fig_c3_confusion.py",
    }
    out = Path(outdir)
    (out / f"{stem}.traceability.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"{stem}.note.txt").write_text(NOTE + "\n", encoding="utf-8")
    caption = (f"Matriz de confusión de {doc['modelo']} en {cell['label']} (n = {style.fmt_es(doc['n'], 0)}), "
               f"evaluación congelada, normalizada por clase real.")
    (out / f"{stem}.caption.txt").write_text(caption + "\n", encoding="utf-8")
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(CELLS), default="ham10000")
    parser.add_argument("--outdir", type=Path, default=REPO / "resultados" / "figuras")
    args = parser.parse_args()
    for path in build(REPO, args.dataset, args.outdir):
        print(path)
