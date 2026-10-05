"""A small table model rendered both as Markdown and as LaTeX (booktabs / longtable, decimal comma upstream)."""

from __future__ import annotations

from dataclasses import dataclass, field

from dermssl.bench.latex import escape

LONG_ROWS = 30
UNICODE_TEX = {"Δ": r"$\Delta$", "−": "$-$", "≥": r"$\ge$", "↓": r"$\downarrow$", "→": r"$\to$"}


@dataclass(frozen=True)
class Table:
    title: str
    header: list[str]
    rows: list[list[str]]
    note: str = ""
    label: str = field(default="")


def markdown(table: Table) -> list[str]:
    """``### title``, the note (if any) and the pipe table."""
    out = [f"### {table.title}", ""]
    out += [table.note, ""] if table.note else []
    out += ["| " + " | ".join(table.header) + " |", "|" + "---|" * len(table.header)]
    return out + ["| " + " | ".join(c.replace("|", r"\|") for c in row) + " |" for row in table.rows] + [""]


def tex_cell(text: str) -> str:
    out = escape(text.replace("`", "")).replace("|", r"\textbar{}")
    for char, tex in UNICODE_TEX.items():
        out = out.replace(char, tex)
    return out


def _tex_row(cells: list[str]) -> str:
    return " & ".join(tex_cell(c) for c in cells) + r" \\"


def latex(table: Table) -> list[str]:
    """One booktabs table; ``longtable`` when it has more than ``LONG_ROWS`` rows."""
    cols = "l" * len(table.header)
    label = rf"\label{{tab:{table.label}}}" if table.label else ""
    caption = rf"\caption{{{tex_cell(table.title)}}}{label}"
    body = [r"\toprule", _tex_row(table.header), r"\midrule", *(_tex_row(r) for r in table.rows), r"\bottomrule"]
    if len(table.rows) > LONG_ROWS:
        return [r"\begingroup\scriptsize", rf"\begin{{longtable}}{{@{{}}{cols}@{{}}}}", caption + r" \\", *body,
                r"\end{longtable}", r"\endgroup", ""]
    return [r"\begin{table}[ht]", r"\centering", r"\scriptsize", r"\resizebox{\textwidth}{!}{%",
            rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}", *body, r"\end{tabular}}", caption, r"\end{table}", ""]


def latex_document(tables: list[Table], generator: str) -> str:
    head = f"% Generado por {generator}. No editar a mano. Requiere booktabs, longtable y graphicx."
    return "\n".join([head, "", *(line for t in tables for line in latex(t))])
