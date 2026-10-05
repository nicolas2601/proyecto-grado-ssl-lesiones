"""LaTeX (booktabs) renderer of one bench table, decimal comma. Needs ``booktabs``."""

from __future__ import annotations

from pathlib import Path

from dermssl.bench.render import YES_NO, num
from dermssl.bench.rows import BLOCKS, METRICS, OK, Dataset, Row, dataset_rows

HEADERS = {"accuracy": "Acc.", "balanced_accuracy": "Bal. acc.", "macro_precision": "Prec.", "macro_recall": "Rec.",
           "macro_f1": "F1", "auc": "AUC", "ece": r"ECE$\downarrow$", "mcc": "MCC"}
SPECIALS = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$"}
COLUMNS = 5 + len(METRICS) + 1


def escape(text: str) -> str:
    return "".join(SPECIALS.get(ch, ch) for ch in text)


def _short_pretraining(text: str) -> str:
    return text.split(" (")[0]


def _ci_f1(row: Row) -> str:
    ci = row.values["macro_f1"].ci
    return f"[{num(ci[0])}; {num(ci[1])}]" if ci else "---"


def _row(row: Row) -> str:
    head = [escape(row.name), escape(_short_pretraining(row.pretraining)), YES_NO[row.finetune], row.segmentation,
            YES_NO[row.transductive]]
    if row.status != OK:
        return " & ".join(head) + rf" & \multicolumn{{{len(METRICS) + 1}}}{{l}}{{{escape(row.status)}}} \\"
    cells = [num(row.values[m].point) for m in METRICS] + [_ci_f1(row)]
    return " & ".join(head + cells) + r" \\"


def _block(rows: list[Row], block: str) -> list[str]:
    heading = rf"\multicolumn{{{COLUMNS}}}{{l}}{{\textit{{{escape(block)}}}}} \\"
    return [heading, r"\midrule", *(_row(r) for r in rows if r.block == block)]


def _blocks(rows: list[Row]) -> list[str]:
    """Every non-empty block, separated by a midrule."""
    out: list[str] = []
    for block in (b for b in BLOCKS if any(r.block == b for r in rows)):
        out += ([r"\midrule"] if out else []) + _block(rows, block)
    return out


def render_latex(repo: Path, ds: Dataset) -> str:
    rows = dataset_rows(repo, ds)
    header = ["Modelo", "Preent.", "FT", "Seg.", "Trans.", *(HEADERS[m] for m in METRICS), r"IC 95\,\% F1"]
    return "\n".join([
        "% Generado por scripts/build_bench_tables.py. No editar a mano. Requiere el paquete booktabs.",
        r"\begin{table}[ht]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lllll" + "r" * (len(METRICS) + 1) + r"@{}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
        *_blocks(rows), r"\bottomrule", r"\end{tabular}",
        rf"\caption{{Banco completo en {escape(ds.title)}: particiones de val distintas entre HAM10000 "
        r"(v6/v7 y Kaggle), ISIC 2019 e ISIC 2020; la comparación entre particiones es solo de intervalos. "
        r"El fine-tuning elige su época sobre val (optimista). Las filas de sensibilidad usan batch 16 (no canónico) "
        r"las exploratorias no cambian el propuesto y las de pesos limpios son una sensibilidad posterior, no selección. IC solo donde la fuente lo publica.}",
        rf"\label{{tab:bench-{ds.key}}}", r"\end{table}", "",
    ])
