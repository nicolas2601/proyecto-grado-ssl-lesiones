"""Limitation note: the linear probe's sensitivity to BLAS threads, rendered from data.

Not an errata: the published v7.1 numbers reproduce bit for bit with the
registered configuration. The note is rendered by ONE function and embedded in
two generated places (the v7.1 README and a marked region of docs/31), so the
two texts cannot drift. Every figure is read from the committed diagnostics:

- ``t4/diagnostico_reproduccion`` (commit 0a691eb): the j10 run (registered
  knobs) against a second j10 run and against a j6 run (8 inner BLAS threads).
- ``t4/diagnostico_convergencia/probe_convergence.json`` (commit 278efd2).
- the paired A* v7.1 − public (same protocol) macro-F1 row of ``v71_final_eval.json``.
- the T4 P1 threshold, ``t4_sog.P1_MAX_ABS_DELTA``.
"""

from __future__ import annotations

import json
from pathlib import Path

from dermssl import t4_sog

REPRO_DIR = "diagnostico_reproduccion"
CONVERGENCE_DIR = "diagnostico_convergencia"
IDENTITY_RUN = "identity_j10_vs_j10.json"
THREADS_RUN = "identity_j10_vs_j6.json"
CONVERGENCE_FILE = "probe_convergence.json"
REPRO_COMMIT, CONVERGENCE_COMMIT = "0a691eb", "278efd2"
SSL_CONTRAST = ("A71", "public_A71", "macro_f1")
METRIC = "macro_f1"
TITLE = "Limitación: sensibilidad del probe a los hilos BLAS"
REGION_START = "<!-- limitacion-hilos-blas:inicio -->"
REGION_END = "<!-- limitacion-hilos-blas:fin -->"


def _paths(t4_dir: Path) -> tuple[Path, Path, Path]:
    t4_dir = Path(t4_dir)
    return (t4_dir / REPRO_DIR / IDENTITY_RUN, t4_dir / REPRO_DIR / THREADS_RUN,
            t4_dir / CONVERGENCE_DIR / CONVERGENCE_FILE)


def available(t4_dir: Path) -> bool:
    return all(p.exists() for p in _paths(t4_dir))


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ssl_row(v71_doc: dict) -> dict:
    for row in v71_doc["paired"]:
        if (row["arm"], row["control"], row["metric"]) == SSL_CONTRAST:
            return row
    raise ValueError(f"v71_final_eval.json no tiene el contraste {SSL_CONTRAST}")


def _reference_delta(identity: dict, threads: dict) -> float:
    """max |Δ macro-F1| of every registered-knob (j10) run against the published grid."""
    if not identity["a_vs_b"]["identical"]:
        raise ValueError("las dos corridas j10 no son idénticas: la reproducción no es bit a bit")
    deltas = [identity["a_vs_published"], identity["b_vs_published"], threads["a_vs_published"]]
    worst = max(d["max_abs_delta"][METRIC] for d in deltas)
    if worst != 0.0:
        raise ValueError(f"la corrida registrada no reproduce la grilla publicada (máx. |Δ| {worst})")
    return worst


def _cells(units: dict) -> tuple[tuple[int, int], int, int]:
    changed = [sum(not c["bit_equal"] for c in u["cells"].values()) for u in units.values()]
    sizes = {len(u["cells"]) for u in units.values()}
    if len(sizes) != 1:
        raise ValueError(f"las unidades no tienen el mismo número de celdas: {sorted(sizes)}")
    flips = max(u["max_argmax_flips"] for u in units.values())
    return (min(changed), max(changed)), sizes.pop(), flips


def load_facts(t4_dir: Path, v71_doc: dict) -> dict:
    identity_path, threads_path, convergence_path = _paths(t4_dir)
    identity, threads, conv = _read(identity_path), _read(threads_path), _read(convergence_path)
    knobs_a, knobs_b = threads["a_environment"]["knobs"], threads["b_environment"]["knobs"]
    changed, cells, flips = _cells(threads["a_vs_b"]["units"])
    iters = [n for fit in conv["fits"] for n in fit["n_iter"]]
    ssl = _ssl_row(v71_doc)
    return {"sensitivity": threads["a_vs_b"]["max_abs_delta"][METRIC],
            "threads": (knobs_a["loky_inner_threads"], knobs_b["loky_inner_threads"]),
            "probe_jobs": (knobs_a["probe_jobs"], knobs_b["probe_jobs"]),
            "reference_vs_published": _reference_delta(identity, threads),
            "changed_cells": changed, "cells_per_unit": cells, "n_units": len(threads["a_vs_b"]["units"]),
            "max_argmax_flips": flips,
            "effect": (ssl["point_delta"], ssl["ci95_low"], ssl["ci95_high"]),
            "n_fits": conv["summary"]["n_fits"], "n_hit_max_iter": conv["summary"]["n_hit_max_iter"],
            "max_iter": conv["max_iter"], "n_iter": (min(iters), max(iters)),
            "p1": t4_sog.P1_MAX_ABS_DELTA}


def number(x: float, decimals: int | None = None, signed: bool = False) -> str:
    """Spanish decimal comma; ``decimals=None`` keeps the shortest exact form (0,01, 0)."""
    text = f"{x:g}" if decimals is None else f"{x:.{decimals}f}"
    if signed and x >= 0:
        text = "+" + text
    return text.replace(".", ",")


def _span(pair: tuple[int, int]) -> str:
    return str(pair[0]) if pair[0] == pair[1] else f"{pair[0]}–{pair[1]}"


from dermssl.display_nomenclature import presentation  # noqa: E402


@presentation
def paragraph(facts: dict) -> str:
    t_a, t_b = facts["threads"]
    j_a, j_b = facts["probe_jobs"]
    eff, lo, hi = facts["effect"]
    sens = facts["sensitivity"]
    return (
        f"Limitación (no errata): el probe lineal tiene una sensibilidad de implementación de hasta "
        f"{number(sens, 3)} de macro-F1 según el número de hilos BLAS por proceso ({t_a} vs {t_b} hilos; "
        f"probe_jobs {j_a} vs {j_b}; cambian {_span(facts['changed_cells'])} de {facts['cells_per_unit']} "
        f"celdas en cada una de las {facts['n_units']} unidades, con hasta {facts['max_argmax_flips']} "
        f"imágenes que cambian de clase predicha por celda); con la configuración registrada ({t_a} hilos) "
        f"reproduce la grilla publicada bit a bit (máx. |Δ| {number(facts['reference_vs_published'])}). "
        f"El efecto SSL (A* v7.1 − público con el mismo protocolo, macro-F1 {number(eff, 3, True)} "
        f"[{number(lo, 3, True)}; {number(hi, 3, True)}]) es ~{number(eff / sens, 1)}× esa sensibilidad. "
        f"No es falta de convergencia: los {facts['n_fits']} ajustes publicados convergen por tolerancia "
        f"({facts['n_hit_max_iter']}/{facts['n_fits']} alcanzan max_iter={facts['max_iter']}; "
        f"n_iter {_span(facts['n_iter'])}); la sensibilidad es de redondeo en punto flotante dentro de la "
        f"tolerancia del optimizador. En T4 el umbral de P1 (≤ {number(facts['p1'])}) está por debajo de "
        f"esa sensibilidad, así que P1 solo se lee pareado dentro de la misma corrida con hilos fijos. "
        f"Evidencia: commits {REPRO_COMMIT} (reproducción) y {CONVERGENCE_COMMIT} (convergencia).")


def body(facts: dict) -> list[str]:
    """The note plus its provenance line; both generated places embed exactly this."""
    return [paragraph(facts), "",
            f"Generado desde `resultados/clasificacion/t4/{REPRO_DIR}/{IDENTITY_RUN}`, `{THREADS_RUN}`, "
            f"`t4/{CONVERGENCE_DIR}/{CONVERGENCE_FILE}`, el contraste pareado "
            f"`{SSL_CONTRAST[0]} − {SSL_CONTRAST[1]}` de `v71_final_eval.json` y "
            "`t4_sog.P1_MAX_ABS_DELTA`; no editar a mano."]


def fill_region(text: str, lines: list[str]) -> str:
    """Replace the generated region of docs/31; it must sit after the registered T4 block."""
    first, last = text.find(REGION_START), text.find(REGION_END)
    if first < 0 or last < first or text.count(REGION_START) != 1 or text.count(REGION_END) != 1:
        raise ValueError(f"el documento no tiene una única región {REGION_START} ... {REGION_END}")
    block_end = text.find(t4_sog.BLOCK_END)
    if block_end >= 0 and first < block_end + len(t4_sog.BLOCK_END):
        raise ValueError("la región generada tiene que ir después del bloque registrado")
    inner = "\n".join(lines)
    return text[:first + len(REGION_START)] + "\n" + inner + "\n" + text[last:]
