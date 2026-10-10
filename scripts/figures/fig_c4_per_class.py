"""C4: per-diagnosis precision, sensitivity, and F1 from committed matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thesis_style as style  # noqa: E402

CONFUSION_DIR = "resultados/banco_completo/confusion"
STEM = "fig_c4_per_diagnosis"
DATASETS = {
    "ham10000": {
        "label": "HAM10000",
        "sources": {
            "b71": "ham10000/b71__congelado.json",
            "a71": "ham10000/a71__congelado.json",
        },
        "protocol": "B71 y A71 congelados, partición común v6/v7.",
    },
    "isic2019": {
        "label": "ISIC 2019",
        "sources": {
            "b71": "isic2019/b71_original__congelado.json",
            "a71": "isic2019/a71_original__congelado.json",
        },
        "protocol": "B71 y A71 originales transductivos, E1 canónico.",
    },
    "isic2020": {
        "label": "ISIC 2020",
        "sources": {
            "b71": "isic2020/b71_original__congelado.json",
            "a71": "isic2020/a71_original__congelado.json",
        },
        "protocol": "B71 y A71 originales transductivos, E1 canónico.",
    },
}
MODEL_LABELS = {"b71": "B71", "a71": "A71"}
METRICS = (("precision", "Precisión"), ("sensitivity", "Sensibilidad"), ("f1", "F1"))
CAPTION = "Precisión, sensibilidad y F1 por diagnóstico para B71 y A71 en la partición indicada."
NOTE = (
    "Filas = clase real; columnas = clase predicha. Precisión = TP/(TP+FP), sensibilidad = TP/(TP+FN) "
    "y F1 = 2TP/(2TP+FP+FN); precisión se fija en 0 si no hubo predicciones de una clase. "
    "Son estimaciones puntuales derivadas de conteos; no se muestran intervalos por clase. "
    "HAM10000 excluye DINOv2 ViT-B/14 porque su matriz usa n=2.014, distinto de B71/A71 (n=2.004). "
    "Las matrices ISIC de B71/A71 son originales y transductivas; no interpretar como validación independiente."
)


def class_metrics(counts: list[list[int]], classes: list[str]) -> list[dict]:
    size = len(classes)
    if size == 0 or len(counts) != size or any(len(row) != size for row in counts):
        raise ValueError("confusion matrix must be square and match the class list")
    if len(set(classes)) != size or any(not label for label in classes):
        raise ValueError("class labels must be unique and non-empty")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for row in counts for value in row):
        raise ValueError("confusion counts must be non-negative integers")

    supports = [sum(row) for row in counts]
    if any(support == 0 for support in supports):
        raise ValueError("every class must have positive true support")
    predicted = [sum(counts[row][column] for row in range(size)) for column in range(size)]
    results = []
    for index, label in enumerate(classes):
        true_positive = counts[index][index]
        precision = true_positive / predicted[index] if predicted[index] else 0.0
        sensitivity = true_positive / supports[index]
        denominator = supports[index] + predicted[index]
        f1 = 2 * true_positive / denominator if denominator else 0.0
        results.append({"class": label, "support": supports[index], "precision": precision,
                        "sensitivity": sensitivity, "f1": f1})
    return results


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_revision(repo: Path, relative: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%H", "--", relative],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _load_model(repo: Path, key: str, relative: str) -> dict:
    path = repo / CONFUSION_DIR / relative
    document = json.loads(path.read_text(encoding="utf-8"))
    classes = document["clases"]
    counts = document["conteos"]
    metrics = class_metrics(counts, classes)
    support = [row["support"] for row in metrics]
    total = sum(support)
    if not isinstance(document.get("n"), int) or total != document["n"]:
        raise ValueError(f"{path}: counts sum {total} does not match n={document.get('n')}")
    if document.get("condicion") != "congelado":
        raise ValueError(f"{path}: C4 requires a frozen model condition")
    return {
        "key": key,
        "label": MODEL_LABELS[key],
        "model": document["modelo"],
        "condition": document["condicion"],
        "classes": list(classes),
        "support": support,
        "n": total,
        "class_metrics": metrics,
        "relative": f"{CONFUSION_DIR}/{relative}",
        "path": path,
        "origin": document.get("origen"),
        "upstream": document.get("fuente", []),
    }


def figure_data(repo: Path = REPO) -> dict:
    repo = Path(repo)
    panels = {}
    for dataset, config in DATASETS.items():
        models = [_load_model(repo, key, relative) for key, relative in config["sources"].items()]
        reference = models[0]
        for model in models[1:]:
            if model["classes"] != reference["classes"]:
                raise ValueError(f"C4 class order differs for {dataset}: {reference['classes']} vs {model['classes']}")
            if model["support"] != reference["support"] or model["n"] != reference["n"]:
                raise ValueError(f"C4 per-class support differs for {dataset}")
        panels[dataset] = {
            "dataset": dataset,
            "title": config["label"],
            "protocol": config["protocol"],
            "n": reference["n"],
            "classes": reference["classes"],
            "support": reference["support"],
            "models": models,
        }
    return {"datasets": panels, "metric_definitions": {
        "precision": "TP/(TP+FP), zero when predicted support is zero",
        "sensitivity": "TP/(TP+FN)",
        "f1": "2TP/(2TP+FP+FN)",
    }}


def _draw_panel(ax, panel: dict, metric: str, title: str) -> None:
    positions = list(range(len(panel["classes"])))
    offsets = {"b71": -0.15, "a71": 0.15}
    markers = {"b71": "o", "a71": "s"}
    faces = {"b71": style.COLORS["accent"], "a71": style.COLORS["background"]}
    for model in panel["models"]:
        values = [row[metric] for row in model["class_metrics"]]
        y_values = [position + offsets[model["key"]] for position in positions]
        ax.plot(values, y_values, linestyle="None", marker=markers[model["key"]], markersize=5,
                markerfacecolor=faces[model["key"]], markeredgecolor=style.COLORS["accent"],
                markeredgewidth=style.LINE_WIDTHS["bar"], zorder=3)
    y_labels = [f"{label} · n={style.fmt_es(support, 0)}"
                for label, support in zip(panel["classes"], panel["support"], strict=True)]
    ax.set_yticks(positions, y_labels)
    ax.invert_yaxis()
    ax.set_ylim(len(positions) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_xticks(ticks, [style.fmt_es(value, 2) for value in ticks])
    ax.set_title(title, loc="left", color=style.COLORS["text"], fontsize=style.PANEL_TITLE_PT, pad=7)
    style.style_axes(ax, "x")


def draw(panel: dict):
    height = max(5.6, 3.0 + 0.72 * len(panel["classes"]))
    with style.context():
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(style.FIGURE_WIDTH_IN, height),
                                 facecolor=style.COLORS["background"])
        for index, (axis, (metric, title)) in enumerate(zip(axes, METRICS, strict=True)):
            _draw_panel(axis, panel, metric, title)
            if index < len(axes) - 1:
                axis.tick_params(axis="x", labelbottom=False)
        axes[-1].set_xlabel("Proporción por diagnóstico")
        handles = [
            Line2D([], [], marker="o", linestyle="None", markerfacecolor=style.COLORS["accent"],
                   markeredgecolor=style.COLORS["accent"], label="B71"),
            Line2D([], [], marker="s", linestyle="None", markerfacecolor=style.COLORS["background"],
                   markeredgecolor=style.COLORS["accent"], label="A71"),
        ]
        fig.text(0.06, 0.97, f"{panel['title']} · n = {style.fmt_es(panel['n'], 0)}",
                 ha="left", va="top", fontsize=style.PANEL_TITLE_PT, color=style.COLORS["text"])
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.96, 0.985),
                   frameon=False, ncol=2, borderaxespad=0)
        fig.subplots_adjust(left=0.27, right=0.96, top=0.90, bottom=0.09, hspace=0.32)
    return fig


def _traceability(repo: Path, panel: dict) -> dict:
    sources = []
    for model in panel["models"]:
        sources.append({
            "path": model["relative"],
            "sha256": _sha256(model["path"]),
            "last_changed_commit": _git_revision(repo, model["relative"]),
            "model": model["model"],
            "condition": model["condition"],
            "n": model["n"],
            "classes": model["classes"],
            "support": model["support"],
            "metrics": model["class_metrics"],
            "origin": model["origin"],
            "upstream": model["upstream"],
        })
    return {
        "figure_id": "C4",
        "dataset": panel["dataset"],
        "evaluation_unit": "image",
        "n": panel["n"],
        "protocol": panel["protocol"],
        "metric_definitions": figure_data_metric_definitions(),
        "figure_script": {"path": "scripts/figures/fig_c4_per_class.py",
                          "sha256": _sha256(repo / "scripts/figures/fig_c4_per_class.py")},
        "models": sources,
        "limitations": ["Point metrics derived from confusion counts; no per-class confidence intervals.",
                        "HAM10000 frozen DINOv2 ViT-B/14 omitted because n=2,014 differs from B71/A71 n=2,004."],
    }


def figure_data_metric_definitions() -> dict:
    return {
        "precision": "TP/(TP+FP), zero when predicted support is zero",
        "sensitivity": "TP/(TP+FN)",
        "f1": "2TP/(2TP+FP+FN)",
    }


def build_all(repo: Path = REPO, outdir: Path | None = None) -> dict[str, list[Path]]:
    repo = Path(repo)
    target = Path(outdir) if outdir is not None else (
        repo / "resultados/figuras/thesis_c1_c2_20261009_final/C4"
    )
    datasets = figure_data(repo)["datasets"]
    stems = [f"{STEM}_{dataset}" for dataset in datasets]
    suffixes = (*style.EXPORT_SUFFIXES, ".caption.txt", ".note.txt", ".traceability.json")
    collisions = [target / f"{stem}{suffix}" for stem in stems for suffix in suffixes
                  if (target / f"{stem}{suffix}").exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite {collisions[0]}")
    target.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for dataset, panel in datasets.items():
        stem = f"{STEM}_{dataset}"
        fig = draw(panel)
        try:
            outputs[dataset] = style.save_figure(fig, stem, target)
        finally:
            plt.close(fig)
        (target / f"{stem}.caption.txt").write_text(
            f"{CAPTION} Con n = {style.fmt_es(panel['n'], 0)} en {panel['title']}\n", encoding="utf-8")
        (target / f"{stem}.note.txt").write_text(f"{panel['protocol']} {NOTE}\n", encoding="utf-8")
        trace = _traceability(repo, panel)
        (target / f"{stem}.traceability.json").write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()
    for dataset, paths in build_all(outdir=args.outdir).items():
        print(dataset, *(str(path) for path in paths))
