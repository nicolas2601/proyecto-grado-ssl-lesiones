"""Paired-contrast sections appended to the HAM10000 and ISIC tables (Spanish, decimal comma)."""

from __future__ import annotations

from pathlib import Path

from dermssl.bench import contrast, contrasts_load as L, isic_rows, pending
from dermssl.bench.contrast import Contrast, num
from dermssl.bench.rows import MODELS, MODEL_NAMES, Dataset, _load

NO_PAIR = ("Los baselines (congelados y con fine-tuning) no tienen probabilidades por imagen commiteadas: su "
           "comparación con B71-limpio es solo de intervalos (IC 95 % de cada fuente), no un test pareado.")
HAM_CAVEAT = ("Contraste solo contra baselines CONGELADOS: el fine-tuning sigue en la partición de Kaggle y no es "
              "pareado.")
LEGEND = "`*` = el IC pareado excluye 0. Δ = B71 − baseline (positivo favorece a B71)."


def _cell(c: Contrast) -> str:
    return f"{num(c.delta, True)} [{num(c.low, True)}; {num(c.high, True)}]" + (" *" if c.excludes_zero else "")


def _isic_pending(repo: Path, ds_key: str) -> list[str]:
    canon = isic_rows.CANONICAL_DIR
    lines = []
    if not (repo / isic_rows.cell_path(canon, ds_key, "B71__original.json")).is_file():
        lines.append(f"- B71-limpio contra B71 original: pendiente; {pending.CLEAN_VS_ORIGINAL.note}.")
    if not (repo / isic_rows.cell_path(canon, ds_key, "A71__clean.json")).is_file():
        lines.append(f"- A71-limpio contra A71 original y B71-limpio contra A71-limpio: pendiente; "
                     f"{pending.A71_CLEAN.note}.")
    return lines


def isic_section(repo: Path, ds: Dataset) -> list[str]:
    unit = "patient_id" if ds.key == "isic2020" else "imagen (sin grupo real)"
    out = ["## Contrastes pareados", "", f"Bootstrap pareado de 1.000 remuestras, unidad: {unit}. " + NO_PAIR, ""]
    for title, items in L.isic_all(repo, ds.key).items():
        out += [f"### {title[0].upper() + title[1:]}", "", *contrast.table(items), ""]
    pend = _isic_pending(repo, ds.key)
    return out + ["### Pendientes", "", *pend, ""] if pend else out


def _ham_above_table(repo: Path, above: list[Contrast], rel: str) -> list[str]:
    full = _load(repo, rel)["models"]
    heads = list(L.B71_VARIANTS.values())
    lines = ["| Baseline (congelado) | F1 macro (v6/v7) | " + " | ".join(f"Δ vs {h}" for h in heads) + " |",
             "|---|---|" + "---|" * len(heads)]
    for model in MODELS:
        name = MODEL_NAMES[model]
        cells = [_cell(next(c for c in above if c.arm == h and c.metric == "macro_f1"
                            and c.control == name + (" + F4" if "F4" in h else ""))) for h in heads]
        lines.append(f"| {name} | {num(full[model]['rows']['full']['macro_f1'])} | " + " | ".join(cells) + " |")
    return lines


def _tally(above: list[Contrast]) -> str:
    parts = []
    for label in L.B71_VARIANTS.values():
        mine = [c for c in above if c.arm == label and c.metric == "macro_f1"]
        parts.append(f"{label}: {sum(c.excludes_zero and c.direction == 'arm_better' for c in mine)}/{len(mine)}")
    return "; ".join(parts)


def ham_section(repo: Path) -> list[str]:
    rel = L.E3B_DIRS["canónico (batch 64)"]
    above, raised = L.e3b_contrasts(repo, rel)
    f4 = [c for c in raised if c.metric == "macro_f1"]
    hit = [c.arm.removesuffix(" + F4") for c in f4 if c.excludes_zero and c.delta > 0]
    return ["## Contrastes pareados (partición v6/v7, n = 2.004)", "",
            f"{HAM_CAVEAT} Los 8 baselines se re-evaluaron sobre la partición v6/v7 con el mismo probe "
            f"(E3b, `{rel}`); sus cifras difieren de las de la tabla de arriba (partición de Kaggle). {LEGEND}", "",
            "### B71 frente a cada baseline congelado (F1 macro)", "", *_ham_above_table(repo, above, rel), "",
            f"B71 por encima del baseline con IC que excluye 0, por variante: {_tally(above)}. Las otras métricas "
            "(exactitud balanceada, recall de melanoma) y las 120 comparaciones están en `CONTRASTES_SSL.md`.", "",
            "### F4 (segmentación, exploratorio) sobre cada baseline congelado (F1 macro)", "",
            *contrast.table(f4), "",
            f"F4 sube el F1 macro de los 8 baselines en {sum(c.delta > 0 for c in f4)} de {len(f4)} casos; el IC "
            f"excluye 0 en {len(hit)} de {len(f4)}: {', '.join(hit)}. Es exploratorio (docs/33, erratas).", ""]
