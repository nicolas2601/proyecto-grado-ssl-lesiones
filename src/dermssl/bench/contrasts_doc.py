"""CONTRASTES_SSL.md and its LaTeX twin: every paired contrast of the bench in one place."""

from __future__ import annotations

from pathlib import Path

from dermssl import ham_clean as hc
from dermssl.bench import contrasts_load as L, e2_tables, ft_contrasts, isic_rows
from dermssl.bench import glossary, pending, tables
from dermssl.bench.contrast import COMPACT_HEADER, METRIC_ES, Contrast, compact_cells, count_excluding, num
from dermssl.bench.ham_rows import COMPUTES
from dermssl.bench.rows import DATASETS, FROZEN, OK, _load, dataset_rows
from dermssl.bench.tables import Table

SCRIPT = "scripts/build_bench_contrasts.py"
NAME = "CONTRASTES_SSL"
CANON, BS32 = L.E3B_DIRS["canónico (batch 64)"], L.E3B_DIRS["batch 32 (ya commiteado)"]
METRICS3 = ("macro_f1", "balanced_accuracy", "mel_recall")
INTRO = ("Todos los contrastes pareados del banco, con dirección y si el IC excluye 0. «Δ» es brazo − control; el "
         "signo ya está resuelto para las métricas donde menor es mejor (ECE) en la columna «Lectura». Los números "
         "salen de los JSON; nada se recalcula aquí.")


def _table(title: str, items: list[Contrast], note: str, label: str) -> Table:
    return Table(title, COMPACT_HEADER, [compact_cells(c) for c in items], note, label)


def ham_e3b(repo: Path) -> tuple[list[Table], list[Contrast]]:
    above, raised = L.e3b_contrasts(repo, CANON)
    note = (f"E3b, cómputo canónico (batch 64 + autocast), HAM10000 val n = 2.004, bootstrap pareado por `lesion_id`. "
            f"Fuente: `{CANON}`. Solo baselines CONGELADOS re-evaluados sobre la partición v6/v7.")
    return ([_table("HAM10000: B71 frente a los 8 baselines congelados (4 variantes × 3 métricas)", above, note,
                    "contrast-e3b-above"),
             _table("HAM10000: F4 sobre cada baseline congelado (exploratorio)", raised, note, "contrast-e3b-f4")],
            above + raised)


def _excluded(items: list[Contrast], arm: str, metric: str) -> int:
    return sum(c.excludes_zero for c in items if c.arm == arm and c.metric == metric)


def compute_table(repo: Path) -> Table:
    rows = []
    for label in L.B71_VARIANTS.values():
        for metric in METRICS3:
            cells = [f"{_excluded(L.e3b_contrasts(repo, rel)[0], label, metric)}/8" for rel in (CANON, BS32)]
            rows.append([label, METRIC_ES[metric], *cells])
    return Table("E3b: baselines que quedan por debajo de B71 con IC que excluye 0, por versión",
                 ["Variante de B71", "Métrica", "Canónico (batch 64)", "Batch 32 (ya commiteado)"], rows,
                 "Resumen de 8 baselines por celda; el detalle del canónico está arriba.", "contrast-e3b-versions")


def mcc_table(repo: Path) -> Table:
    models = _load(repo, CANON)["models"]
    rows = [[name, num(m["rows"]["full"]["mcc"]), num(m["rows"]["F4"]["mcc"]),
             num(m["rows"]["F4"]["mcc"] - m["rows"]["full"]["mcc"], True)] for name, m in
            ((L.baseline_label(k), v) for k, v in models.items())]
    note = ("MCC de E3b: solo cifras puntuales de cada fila. Los JSON commiteados de E3 y E3b no traen contrastes "
            "pareados de MCC (se commitearon antes del commit 1d7e447, que agrega el MCC a esos contrastes): la última "
            "columna es la resta de dos puntos, SIN IC y NO es un contraste. El contraste pareado de MCC exige "
            "recomputar E3 y E3b con el código actual; ETA no declarada por el operador.")
    return Table("HAM10000: MCC puntual de los baselines sin y con F4 (E3b)", ["Baseline", "MCC sin F4", "MCC con F4",
                 "Diferencia puntual (sin IC)"], rows, note, "contrast-e3b-mcc")


def ham_f4(repo: Path) -> tuple[list[Table], list[Contrast]]:
    out, items = [], []
    runs = [(f"cómputo {tag}: {label}", rel, f"contrast-f4-{tag.lower()}") for tag, label, rel in COMPUTES]
    runs += [(f"semilla de B* {seed}" + (" (= cómputo canónico S1)" if seed == 42 else ""), rel, f"contrast-f4-seed{seed}")
             for seed, rel in L.SEED_FILES.items()]
    for title, rel, label in runs:
        got = L.f4_vs_b71(repo, rel)
        items += got
        out.append(_table(f"HAM10000: F4 − B71, {title}", got, f"Bootstrap pareado por `lesion_id`, n = 2.004. Fuente: `{rel}`.", label))
    return out, items


def ham_v71(repo: Path) -> tuple[Table, list[Contrast]]:
    got = L.v71_contrasts(repo)
    note = f"v7.1 en HAM10000, bootstrap pareado por `lesion_id`, n = 2.004. Fuente: `{L.V71_JSON}`."
    return _table("HAM10000: v7 y v7.1 (48 contrastes)", got, note, "contrast-v71"), got


CLEAN_TITLE = "HAM10000: pesos limpios (sensibilidad) frente al original y al público"


def ham_clean_table(repo: Path, label: str = "contrast-ham-clean") -> tuple[list[Table], list[Contrast]]:
    """B71-limpio / A71-limpio contrasts (with MCC); no table until the result JSON exists."""
    got = L.ham_clean_contrasts(repo)
    if not got:
        return [], []
    n = f"{_load(repo, hc.RESULT_JSON)['n_eval']:,}".replace(",", ".")
    note = ("Análisis de sensibilidad posterior, no es selección: configuración y umbrales del holdout original "
            "(los umbrales se ajustaron con los pesos originales y se aplican sin cambios a los limpios). Cómputo "
            f"canónico, HAM10000 val n = {n}, bootstrap pareado por `lesion_id`, 1.000 remuestras, semilla "
            f"20260923. Fuente: `{hc.RESULT_JSON}`.")
    return [_table(CLEAN_TITLE, got, note, label)], got


def isic(repo: Path) -> tuple[list[Table], list[Contrast]]:
    out, items = [], []
    for ds in ("isic2019", "isic2020"):
        for slot, (kind, got) in enumerate(L.isic_all(repo, ds).items()):
            items += got
            unit = "`patient_id`" if ds == "isic2020" else "imagen (sin grupo real)"
            note = f"Bootstrap pareado por {unit}, 1.000 remuestras. Fuente: `{got[0].source}` y hermanos."
            out.append(_table(f"{DATASETS[ds].title}: {kind}", got, note, f"contrast-{ds}-{('canon', 'sens')[slot]}"))
    return out, items


def _overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def ft_table(repo: Path) -> tuple[Table, list[Contrast]]:
    return ft_contrasts.table(repo), ft_contrasts.contrasts(repo)


def interval_table(repo: Path, ds_key: str) -> Table:
    rows = dataset_rows(repo, DATASETS[ds_key])
    mine = next(r for r in rows if r.name == "B71-limpio (canónico)")
    ci = mine.values["macro_f1"].ci
    out = []
    for r in (r for r in rows if r.block == FROZEN and r.status == OK and "(canónico)" not in r.name
              and "propuesto" not in r.name):
        b = r.values["macro_f1"].ci
        verdict = "los IC se solapan: no se afirma diferencia" if _overlap(ci, b) else \
            "los IC no se solapan (solo de intervalos, no es un test pareado)"
        out.append([r.name, num(r.values["macro_f1"].point) + f" [{num(b[0])}; {num(b[1])}]", verdict])
    note = (f"B71-limpio: F1 macro {num(mine.values['macro_f1'].point)} [{num(ci[0])}; {num(ci[1])}]. Los baselines no "
            "tienen probabilidades por imagen commiteadas: solo se comparan intervalos. La comparación mezcla modelo y "
            "protocolo (ver «Cómo leer este documento»).")
    return Table(f"{DATASETS[ds_key].title}: B71-limpio frente a los baselines (solo intervalos)",
                 ["Baseline congelado", "F1 macro [IC 95 %]", "Lectura"], out, note, f"contrast-interval-{ds_key}")


def pending_table(repo: Path) -> Table:
    """Contrasts whose data is missing, derived from the files."""
    rows = [[f"{DATASETS[ds].title}: {name}", pending.missing(f"{isic_rows.CANONICAL_DIR}/{ds}/contrasts/{name}.json").reason,
             pending.NO_ETA] for ds in ("isic2019", "isic2020") for name in isic_rows.CANONICAL_CONTRASTS
            if not (repo / f"{isic_rows.CANONICAL_DIR}/{ds}/contrasts/{name}.json").is_file()]
    if not e2_tables.exists(repo):
        rows.append(["D3 − D2 y demás contrastes de E2", pending.missing(e2_tables.E2_JSON).reason, pending.NO_ETA])
    above, _ = L.e3b_contrasts(repo, CANON)
    if not any(c.metric == "mcc" for c in above):
        rows.append(["Contrastes pareados de MCC en HAM10000 (E3 y E3b)",
                     "los JSON de E3/E3b no traen MCC en sus contrastes; recomputarlos con el código actual (1d7e447)",
                     pending.NO_ETA])
    if not (repo / hc.RESULT_JSON).is_file():
        rows.append(["B71-limpio y A71-limpio contra el original en HAM10000", pending.HAM_CLEAN.reason, pending.NO_ETA])
    return Table("Contrastes pendientes", ["Contraste", "Motivo", "ETA"], rows, "", "contrast-pending")


def warning(items: list[Contrast]) -> list[str]:
    n, hit = len(items), count_excluding(items)
    return ["## Advertencia de multiplicidad", "",
            f"Este documento lista {n} contrastes (el IC excluye 0 en {hit}) **sin corrección por multiplicidad**: cada "
            f"IC es de 95 %, así que aun sin ningún efecto real se esperaría que cerca del 5 % (≈ {n * 0.05:.0f}) "
            "excluyera 0. Además comparten imágenes y no son independientes, así que esa cifra es solo orientativa. Se "
            "leen por familia y con la pregunta registrada de cada una, nunca escogiendo los que salen bien.", ""]


def build(repo: Path) -> tuple[list[Table], list[Contrast]]:
    e3b, e3b_items = ham_e3b(repo)
    f4, f4_items = ham_f4(repo)
    v71, v71_items = ham_v71(repo)
    isic_tables, isic_items = isic(repo)
    clean_tabs, clean_items = ham_clean_table(repo)
    ft_tab, ft_items = ft_table(repo)
    e2 = [e2_tables.contrasts_table(repo)] if e2_tables.exists(repo) else []
    e2_items = e2_tables.contrasts(repo) if e2 else []
    tabs = [*e3b, compute_table(repo), mcc_table(repo), *f4, v71, *clean_tabs, *isic_tables, ft_tab, *e2,
            interval_table(repo, "isic2019"), interval_table(repo, "isic2020"), pending_table(repo)]
    return tabs, e3b_items + f4_items + v71_items + clean_items + isic_items + ft_items + e2_items


def render_markdown(repo: Path) -> str:
    tabs, items = build(repo)
    lines = ["# Contrastes pareados del banco completo", "", INTRO, "",
             *glossary.preamble(repo, DATASETS, SCRIPT, warning(items))]
    return "\n".join(lines + [line for t in tabs for line in tables.markdown(t)])


def render_latex(repo: Path) -> str:
    return tables.latex_document(build(repo)[0], SCRIPT)


def outputs(repo: Path) -> dict[Path, str]:
    return {Path(f"{NAME}.md"): render_markdown(repo), Path("latex") / f"{NAME}.tex": render_latex(repo)}
