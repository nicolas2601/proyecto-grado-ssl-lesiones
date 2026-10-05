"""Limitations section of resultados/TABLA_CIERRE.md: measured facts, not experiments.

No row carries a verdict. The BLAS-thread note is ``dermssl.probe_blas_limitation``
verbatim (the same text the v7.1 README and docs/31 embed); the probe
convergence and the C1 thread reproduction are read from their diagnostics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_segmentation_table import _num  # noqa: E402
from closing_table_rules import _fuente  # noqa: E402

from dermssl import probe_blas_limitation as pbl  # noqa: E402

HEADING = "## Limitaciones (no son experimentos)"
CLF = Path("resultados") / "clasificacion"
T4_DIR = CLF / "t4"
V71_EVAL = CLF / "v71" / "v71_final_eval.json"
REPRO = tuple(T4_DIR / pbl.REPRO_DIR / name for name in (pbl.IDENTITY_RUN, pbl.THREADS_RUN))
CONVERGENCE = T4_DIR / pbl.CONVERGENCE_DIR / pbl.CONVERGENCE_FILE
C1_THREADS = CLF / "c1_v7" / "diagnostico_hilos" / "c1_thread_repro.json"


def sources() -> tuple[Path, ...]:
    return (*REPRO, CONVERGENCE, V71_EVAL, C1_THREADS)


def _read(root: Path, path: Path) -> dict:
    return json.loads((root / path).read_text(encoding="utf-8"))


def blas(root: Path) -> str:
    return pbl.paragraph(pbl.load_facts(root / T4_DIR, _read(root, V71_EVAL)))


def convergence(root: Path) -> str:
    doc = _read(root, CONVERGENCE)
    s, units = doc["summary"], doc["sanity"]["units"].values()
    cells = sum(u[kind]["n_cells"] for u in units for kind in ("probas", "metrics"))
    return (f"{s['n_fits']} ajustes del probe publicado de v7.1 en {len(s['by_unit'])} unidades, max_iter="
            f"{doc['max_iter']}, {doc['blas_threads']} hilos BLAS: alcanzan max_iter {s['n_hit_max_iter']}/"
            f"{s['n_fits']}; avisos de convergencia: {s['n_convergence_warning']}. Antes, el refit reproduce "
            f"probabilidades y métricas publicadas en las {cells} comparaciones "
            f"({'sanity superado' if doc['sanity']['passed'] else 'sanity NO superado'}). Commit "
            f"{pbl.CONVERGENCE_COMMIT}; código `{doc['code_revision'][:7]}`.")


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _setting(key: str, p: dict) -> str:
    return "sin límite (default)" if p["threads"] is None else f"{key} hilo{'s' if p['threads'] != 1 else ''}"


def c1_threads(root: Path) -> str:
    doc = _read(root, C1_THREADS)
    s = doc["summary"]
    failing = [(k, p) for k, p in s["per_setting"].items() if p["n_bit_equal"] < p["n_cells"]]
    fails = "; ".join(f"con {_setting(k, p)}: {p['n_bit_equal']} de {p['n_cells']} celdas bit a bit, máx. "
                      f"|Δ proba| {_num(p['max_abs_delta'])}, hasta {p['max_argmax_flips']} imágenes cambian "
                      f"de clase, máx. |Δ macro-F1| {_num(p['max_abs_delta_macro_f1'])}" for k, p in failing)
    return (f"Refit de {s['n_gated']} de {s['n_cells']} celdas publicadas de C1-v7 que pasan el gate de "
            f"features, variando los hilos BLAS. Reproducen bit a bit con: {', '.join(s['reproducing_settings'])}"
            f" (default = sin límite). {_upper_first(fails) if fails else 'Ningún ajuste falla'}. Decisión "
            f"registrada: fijar C1 en {s['pin_to']} hilos. Código `{doc['code_revision'][:7]}`.")


def limitation_rows(root: Path) -> list[tuple[str, str, str]]:
    return [
        (_upper_first(pbl.TITLE.removeprefix("Limitación: ")), blas(root),
         _fuente((*REPRO, CONVERGENCE, V71_EVAL))),
        ("Convergencia del probe lineal (v7.1)", convergence(root), _fuente((CONVERGENCE,))),
        ("Reproducción de C1-v7 según los hilos BLAS", c1_threads(root), _fuente((C1_THREADS,))),
    ]


def section(root: Path, cell) -> list[str]:
    lines = [HEADING, "",
             "Hechos medidos que acotan cómo leer las filas de arriba. No tienen predicción ni veredicto.", "",
             "| # | Limitación | Qué se midió | Datos |", "|---:|---|---|---|"]
    for i, (title, text, fuente) in enumerate(limitation_rows(root), 1):
        lines.append(f"| L{i} | {cell(title)} | {cell(text)} | {fuente} |")
    return lines
