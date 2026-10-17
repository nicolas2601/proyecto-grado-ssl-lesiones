"""README and PNG of the confusion folder (matplotlib, Agg backend, Spanish labels)."""

from __future__ import annotations

from pathlib import Path

from dermssl.bench.confusion import CONDITIONS, NO_DIFF, SSL_MODELS, V71_JSON, rel_json

CAPTIONS = {"counts": "Conteos", "norm": "Normalizada por fila (recall por clase en la diagonal)"}


from dermssl.display_nomenclature import context_presentation, display_text  # noqa: E402


@context_presentation("bench")
def readme(arts: list[dict], pend: list[dict]) -> str:
    lines = [f"- `{rel_json(a).as_posix()}` (+ `.png`): {a['dataset']}, {a['modelo']}, {a['condicion']}; "
             f"origen: {a['origen']}." for a in arts]
    waiting = [f"- {p['dataset']}, {p['modelo']} ({p['condicion']}): {p['estado']}; {p['motivo']}; ETA {p['eta']}."
               for p in pend]
    return "\n".join([
        "# Matrices de confusión de los modelos auto-supervisados", "",
        "Generado por `scripts/build_bench_tables.py` desde los JSON del repositorio. No editar a mano. "
        "Alcance (docs/33, E4): solo los auto-supervisados, DINOv2 ViT-S/14, DINOv2 ViT-B/14 y SimSiam "
        "ResNet-50, congelados y con fine-tuning, más A71 y B71 (HAM10000) y, en ISIC, B71-limpio (canónico) y "
        "B71/A71 originales (sensibilidad: batch 16, transductivo).", "",
        "Cada celda tiene un `.json` (clases, conteos, matriz normalizada por fila, MCC, fuente) y un `.png` "
        "con dos paneles: conteos y normalizada por fila. Filas = clase real, columnas = clase predicha.", "",
        "## Origen de los números", "",
        "- Baselines: `matriz_confusion` de los JSON de `results/` (probe `logreg` para congelados, `finetune` "
        "para fine-tuning).",
        "- ISIC 2019 e ISIC 2020, B71-limpio y B71/A71 originales: `matriz_confusion` de la celda E1 "
        "(`resultados/banco_completo/isic_eval/...`); el JSON de cada matriz guarda n de la celda y n desde la matriz.",
        f"- A71 y B71 (HAM10000): `{V71_JSON}` no trae matriz de confusión. Se derivó con el argmax de las "
        "probabilidades commiteadas (`v71_final_eval.probas.npz`); el JSON de cada celda guarda la accuracy "
        "del JSON y la recomputada desde la matriz, y un test exige que coincidan.", "",
        "## Figura de diferencias frente a B71", "", NO_DIFF, "",
        "## Matrices generadas", "", *lines, "", "## Pendientes (sin matriz, no se inventa)", "", *waiting, "",
        f"Modelos incluidos: {', '.join(SSL_MODELS)}; condiciones: {', '.join(CONDITIONS)}.", "",
    ])


def _annotate(ax, matrix: list[list[float]], as_counts: bool) -> None:
    peak = max(max(row) for row in matrix) or 1
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            text = f"{value}" if as_counts else f"{value:.2f}".replace(".", ",")
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if value > 0.6 * peak else "black")


def _panel(fig, ax, matrix: list[list[float]], classes: list[str], key: str) -> None:
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(CAPTIONS[key], fontsize=10)
    _annotate(ax, matrix, key == "counts")
    fig.colorbar(image, ax=ax, fraction=0.046)


def write_png(art: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    _panel(fig, axes[0], art["conteos"], art["clases"], "counts")
    _panel(fig, axes[1], art["normalizada_por_fila"], art["clases"], "norm")
    fig.suptitle(display_text(f"{art['modelo']}, {art['condicion']}: {art['dataset']} (n = {art['n']})"))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)
