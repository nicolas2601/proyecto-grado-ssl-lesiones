"""Genera las tablas de resultados del README a partir de los JSON del banco.

Dieciséis ejecuciones por tres tablas es donde los números escritos a mano se
deterioran. El Markdown de README.md y notebooks/*/README.md se genera aquí y
se pega literalmente. El ranking siempre sigue el mejor probe de cada modelo.

    python scripts/build_results_tables.py --bank baselines
    python scripts/build_results_tables.py --bank baselines_isic2019
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.finetune import protocol  # noqa: E402

DISPLAY: dict[str, str] = {
    "alexnet": "AlexNet",
    "vgg19": "VGG19",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet-50",
    "vit_b16": "ViT-B/16",
    "simsiam_r50": "SimSiam R50",
    "dinov2_vits14": "DINOv2 ViT-S/14",
    "dinov2_vitb14": "DINOv2 ViT-B/14",
}

SUPERVISION: dict[str, str] = {
    "alexnet": "Supervisado",
    "vgg19": "Supervisado",
    "efficientnet_b0": "Supervisado",
    "resnet50": "Supervisado",
    "vit_b16": "Supervisado",
    "simsiam_r50": "SSL",
    "dinov2_vits14": "SSL",
    "dinov2_vitb14": "SSL",
}

PROBE_LABELS = {"knn": "kNN", "logreg": "LogReg", "prototype": "Prototipos"}
METRIC_COLUMNS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_precision",
                  "macro_recall", "auc_ovr_macro", "ece")
LOWER_IS_BETTER = {"ece"}


class MissingField(RuntimeError):
    """Los JSON no contienen lo necesario para la tabla solicitada."""


@dataclass(frozen=True)
class Row:
    model: str
    probe: str
    dims: int
    metrics: dict
    probes: dict
    data: dict


def best_probe(data: dict) -> str:
    probes = data["probes"]
    return max(probes, key=lambda name: probes[name]["balanced_accuracy"])


def rank(bank: dict[str, dict]) -> list[Row]:
    rows = [
        Row(model=model, probe=best_probe(data), dims=data.get("dim_features", 0),
            metrics=data["probes"][best_probe(data)], probes=data["probes"],
            data=data)
        for model, data in bank.items()
    ]
    return sorted(rows, key=lambda r: r.metrics["balanced_accuracy"], reverse=True)


def load_bank(directory: Path) -> dict[str, dict]:
    bank: dict[str, dict] = {}
    for path in sorted(directory.glob("results_*.json")):
        data = json.loads(path.read_text())
        if data.get("smoke"):
            continue
        bank[data["modelo"]] = data
    if not bank:
        raise MissingField(f"no hay resultados reales (SMOKE=False) en {directory}")
    return bank


def _fmt(value: float, bold: bool = False) -> str:
    text = f"{value:.3f}"
    return f"**{text}**" if bold else text


def _leaders(rows: list[Row]) -> dict[str, float]:
    return {
        column: (min if column in LOWER_IS_BETTER else max)(
            row.metrics[column] for row in rows)
        for column in METRIC_COLUMNS
    }


def _table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return "\n".join(lines)


def table_metrics(rows: list[Row]) -> str:
    best = _leaders(rows)
    header = ["#", "Modelo", "Tipo", "dims", "acc", "bal_acc", "F1 macro",
              "Prec macro", "Recall macro", "AUC", "ECE"]
    body = [
        [str(i), DISPLAY.get(row.model, row.model), SUPERVISION.get(row.model, "?"),
         str(row.dims)]
        + [_fmt(row.metrics[c], row.metrics[c] == best[c]) for c in METRIC_COLUMNS]
        for i, row in enumerate(rows, start=1)
    ]
    return _table(header, body)


def overlaps_leader(row: Row, leader: Row) -> bool:
    """Verdadero si el IC de bal_acc de este modelo toca el del líder.

    Un solapamiento significa que la brecha de ranking no puede resolverse con
    este conjunto de validación: hay diferencia entre "supera en ranking" y
    "gana".
    """
    return (row.metrics["ic95_bootstrap"]["balanced_accuracy"][1]
            >= leader.metrics["ic95_bootstrap"]["balanced_accuracy"][0])


def table_intervals(rows: list[Row]) -> str:
    """Bal_acc y macro-F1 con sus intervalos bootstrap al 95%.

    Se imprimen junto al ranking intencionalmente: sin intervalos, quien lee
    toma la diferencia entre las posiciones 1 y 3 como resultado, y aquí no lo es.
    """
    for row in rows:
        if "ic95_bootstrap" not in row.metrics:
            raise MissingField(
                f"{row.model}: el JSON es anterior al intervalo bootstrap")
    leader = rows[0]
    header = ["Modelo", "bal_acc", "IC95 bal_acc", "F1 macro", "IC95 F1 macro",
              "Separado del líder"]
    body = []
    for row in rows:
        ci = row.metrics["ic95_bootstrap"]
        sep = "-" if row is leader else ("sí" if not overlaps_leader(row, leader) else "no")
        body.append([
            DISPLAY.get(row.model, row.model),
            _fmt(row.metrics["balanced_accuracy"]),
            f"[{ci['balanced_accuracy'][0]:.3f}, {ci['balanced_accuracy'][1]:.3f}]",
            _fmt(row.metrics["macro_f1"]),
            f"[{ci['macro_f1'][0]:.3f}, {ci['macro_f1'][1]:.3f}]",
            sep,
        ])
    return _table(header, body)


def convergence(data: dict) -> str:
    """Three states, and the third is not the second.

    A run whose best epoch is its last one was still improving when the budget
    ran out, so its score is a floor and not a ceiling - that is what sank the
    DINOv2 rows of the HAM10000 bank. A JSON with no `truncada` at all is a
    third case: the evidence was never recorded. Folding it into "converged"
    would invent a measurement nobody took.
    """
    if "truncada" not in data:
        return "desconocida"
    return "truncada" if data["truncada"] else "convergio"


CONVERGENCE_LABEL = {"convergio": "si", "truncada": "**NO**", "desconocida": "**?**"}


def table_convergence(rows: list[Row]) -> str:
    header = ["Modelo", "Convergio", "Mejor epoca", "Epocas"]
    body = []
    for row in rows:
        state = convergence(row.data)
        best = row.data.get("mejor_epoca", "-")
        total = row.data.get("epocas_corridas") or row.data.get(
            "protocolo", {}).get("epocas", "-")
        body.append([DISPLAY.get(row.model, row.model),
                     CONVERGENCE_LABEL[state], str(best), str(total)])
    return _table(header, body)


def convergence_warning(rows: list[Row]) -> str:
    states = {convergence(row.data) for row in rows}
    if states == {"convergio"}:
        n = len(rows)
        word = {1: "El unico modelo", 2: "Los dos modelos", 8: "Los ocho modelos"}
        subject = word.get(n, f"Los {n} modelos")
        verb = "convergio" if n == 1 else "convergieron"
        return (f"> {subject} {verb} dentro del presupuesto de epocas: "
                "ninguno llego a la ultima epoca todavia mejorando.")
    parts = []
    if "truncada" in states:
        bad = [DISPLAY.get(r.model, r.model) for r in rows
               if convergence(r.data) == "truncada"]
        parts.append(
            f"**{', '.join(bad)}: corrida truncada.** No convergieron dentro del "
            "presupuesto de epocas: "
            "su mejor epoca es la ultima, o sea que seguian mejorando cuando se "
            "corto el entrenamiento. Su numero es un **piso**, no un techo, y el "
            "orden de la tabla no los compara con los demas.")
    if "desconocida" in states:
        unknown = [DISPLAY.get(r.model, r.model) for r in rows
                   if convergence(r.data) == "desconocida"]
        parts.append(
            f"**{', '.join(unknown)}: convergencia desconocida.** Su JSON no trae "
            "el campo, porque la corrida es anterior a que el generador lo "
            "escribiera. Ausente no es lo mismo que convergio: no se sabe si el "
            "numero es un techo o un piso.")
    return "> ## Atencion: convergencia\n>\n> " + "\n>\n> ".join(parts)


def probes_present(rows: list[Row]) -> list[str]:
    """The probes this bank actually ran, in the canonical order.

    A frozen bank probes a fixed backbone three ways (kNN, logistic regression,
    prototypes). A fine-tuning bank trains the whole network, so it has nothing
    to probe and ships a single entry called 'finetune'. Reading the names off
    the JSONs keeps one renderer honest for both.
    """
    seen = {name for row in rows for name in row.probes}
    ordered = [p for p in PROBE_LABELS if p in seen]
    return ordered + sorted(seen - set(ordered))


def table_probes(rows: list[Row]) -> str:
    names = probes_present(rows)
    header = ["Modelo"] + [PROBE_LABELS.get(p, p) for p in names]
    body = [
        [DISPLAY.get(row.model, row.model)]
        + [_fmt(row.probes[p]["balanced_accuracy"]) if p in row.probes else "-"
           for p in names]
        for row in rows
    ]
    return _table(header, body)


def table_per_class(rows: list[Row], field: str) -> str:
    for row in rows:
        if field not in row.metrics:
            raise MissingField(
                f"{row.model}: el probe '{row.probe}' no tiene '{field}'; "
                "el JSON proviene de una ejecución anterior a esa métrica")
    classes = list(rows[0].metrics[field])
    body = [
        [DISPLAY.get(row.model, row.model)]
        + [_fmt(row.metrics[field][c]) for c in classes]
        for row in rows
    ]
    return _table(["Modelo"] + classes, body)


def table_weights() -> str:
    """Procedencia de los pesos de cada backbone, según el protocolo.

    Escrita a mano, esta tabla quedó desactualizada cuando el banco migró a
    Hugging Face: aún afirmaba torchvision para cuatro modelos ya migrados,
    incluido vit_b16, cuyo cambio de pesos alteró la conclusión de la comparación.
    """
    notes = {
        "vgg19": "mismos tensores que torchvision IMAGENET1K_V1",
        "resnet50": "mismos tensores que torchvision IMAGENET1K_V2",
        "efficientnet_b0": "receta propia de timm (no es la de torchvision)",
        "vit_b16": "supervisado SOLO con ImageNet-1k (comparador justo)",
        "dinov2_vits14": "SSL sobre LVD-142M sin etiquetas",
        "dinov2_vitb14": "SSL sobre LVD-142M sin etiquetas",
        "alexnet": "sin pesos oficiales en HF Hub",
        "simsiam_r50": "checkpoint oficial de FAIR, 100 epocas, batch 256",
    }
    body = []
    for model in protocol.MODELS:
        if model in protocol.TIMM_HF_TAGS:
            source = f"Hugging Face via timm: `{protocol.TIMM_HF_TAGS[model]}`"
        elif model in protocol.TORCHVISION_MODELS:
            source = "torchvision `IMAGENET1K_V1`"
        else:
            source = f"`{protocol.SIMSIAM_CHECKPOINT_URL.rsplit('/', 1)[-1]}`"
        body.append([DISPLAY[model], source, notes[model]])
    return _table(["Modelo", "Checkpoint", "Nota"], body)


from dermssl.display_nomenclature import presentation  # noqa: E402


@presentation
def render(bank_dir: Path) -> str:
    rows = rank(load_bank(bank_dir))
    blocks = []
    if any("truncada" in r.data or "mejor_epoca" in r.data for r in rows):
        blocks.append(convergence_warning(rows))
        blocks.append(f"### Convergencia por modelo\n\n{table_convergence(rows)}")
    blocks.append(f"### Métricas completas (mejor probe de cada modelo)\n\n{table_metrics(rows)}")
    try:
        blocks.append("### Intervalos de confianza (bootstrap 95%)\n\n"
                      f"{table_intervals(rows)}")
    except MissingField as exc:
        blocks.append(f"### Intervalos de confianza: no disponibles\n\n> {exc}")
    names = probes_present(rows)
    if len(names) > 1:
        blocks.append(f"### Balanced accuracy por probe\n\n{table_probes(rows)}")
    else:
        # One probe means the table would repeat the metrics table column for
        # column. Name the probe instead so the reader knows what was measured.
        blocks.append("### Balanced accuracy por probe: no aplica\n\n"
                      f"> Este banco tiene un solo modo de evaluacion "
                      f"(`{names[0]}`), asi que la columna ya esta en la tabla "
                      "de metricas completas.")
    for field, title in (("recall_por_clase", "Recall por clase"),
                         ("precision_por_clase", "Precisión por clase"),
                         ("f1_por_clase", "F1 por clase")):
        try:
            blocks.append(f"### {title} (mejor probe)\n\n"
                          f"{table_per_class(rows, field=field)}")
        except MissingField as exc:
            blocks.append(f"### {title}: no disponible\n\n> {exc}")
    return "\n\n".join(blocks) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--weights", action="store_true",
                    help="muestra la tabla de procedencia de checkpoints en vez de resultados")
    args = p.parse_args(argv)
    if args.weights:
        print(table_weights())
        return
    if not args.bank and not args.results_dir:
        p.error("indique --bank <banco>, --results-dir <ruta> o --weights")
    directory = Path(args.results_dir) if args.results_dir else REPO / "results" / args.bank
    print(render(directory))


if __name__ == "__main__":
    main()
