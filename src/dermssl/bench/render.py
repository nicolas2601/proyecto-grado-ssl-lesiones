"""Markdown renderer of one bench table (decimal comma, es-CO)."""

from __future__ import annotations

from pathlib import Path

from dermssl.bench import glossary, table_extras
from dermssl.bench.rows import BLOCKS, DATASETS, METRICS, OK, Dataset, Row, Value, dataset_rows

NO_VALUE = "—"
HEADERS = {"accuracy": "Accuracy", "balanced_accuracy": "Exactitud balanceada", "macro_precision": "Precisión macro",
           "macro_recall": "Recall macro", "macro_f1": "F1 macro", "auc": "AUC OvR macro", "ece": "ECE ↓",
           "mcc": "MCC"}
YES_NO = {True: "sí", False: "no"}


def num(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def cell(value: Value) -> str:
    if value.ci is None:
        return num(value.point)
    return f"{num(value.point)} [{num(value.ci[0])}; {num(value.ci[1])}]"


def _sources(row: Row) -> str:
    return " · ".join(f"`{s}`" for s in row.sources)


def _numbered(rows: list[Row]) -> list[tuple[str, Row]]:
    count = 0
    out = []
    for row in rows:
        if row.status == OK:
            count += 1
        out.append((str(count) if row.status == OK else NO_VALUE, row))
    return out


def condition_table(rows: list[Row]) -> list[str]:
    head = "| # | Modelo | Preentrenamiento | Fine-tuning | Segmentación | Transductivo | Val n | Estado | Nota | Fuente |"
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for idx, r in _numbered(rows):
        n = f"{r.n_val:,}".replace(",", ".") if r.n_val else NO_VALUE
        lines.append(f"| {idx} | {r.name} | {r.pretraining} | {YES_NO[r.finetune]} | {r.segmentation} | "
                     f"{YES_NO[r.transductive]} | {n} | {r.status} | {r.note or NO_VALUE} | {_sources(r)} |")
    return lines


def metrics_table(rows: list[Row]) -> list[str]:
    lines = ["| # | Modelo | " + " | ".join(HEADERS[m] for m in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for idx, r in _numbered(rows):
        cells = [cell(r.values[m]) for m in METRICS] if r.status == OK else [r.status] * len(METRICS)
        lines.append(f"| {idx} | {r.name} | " + " | ".join(cells) + " |")
    return lines


def block_section(rows: list[Row], block: str) -> list[str]:
    mine = [r for r in rows if r.block == block]
    return [f"## {block[0].upper() + block[1:]}", "", "Orden: F1 macro de mayor a menor; las filas sin número van al final.",
            "", "### Condición", "", *condition_table(mine), "", "### Métricas [IC 95 % donde la fuente lo publica]",
            "", *metrics_table(mine), ""]


def render_markdown(repo: Path, ds: Dataset) -> str:
    rows, facts = dataset_rows(repo, ds), glossary.dataset_facts(repo, ds)
    return "\n".join([
        f"# Banco completo: {ds.title}, propuesto y 8 baselines", "",
        "Generado por `scripts/build_bench_tables.py` desde los JSON del repositorio. No editar a mano. "
        "Plan y reglas: `docs/33-banco-completo-multidataset.md`.", "",
        "## Glosario", "", *glossary.GLOSSARY, "", glossary.macro_note(facts), "",
        "## Cómo leer esta tabla", "", *glossary.reading_rules(repo, DATASETS), "",
        *(line for block in BLOCKS if any(r.block == block for r in rows) for line in block_section(rows, block)),
        *(table_extras.ham_section(repo) if ds.key == "ham10000" else table_extras.isic_section(repo, ds)),
    ])
