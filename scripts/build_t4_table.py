"""Render the T4 (Shades of Gray) single-evaluation table from its JSON; ``--check`` for CI.

    python scripts/build_t4_table.py resultados/clasificacion/t4_sog/t4_final_eval.json \
        --out resultados/clasificacion/t4_sog/README.md [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import t4_sog  # noqa: E402

WEIGHTS = (("public", "Público"), ("A", "A*"), ("B", "B*"))
METRICS = {"macro_f1": "Macro-F1", t4_sog.GAP_METRIC: "Δ brecha absoluta oscuro − claro (menor es mejor)"}
TERCILES = (("dark", "oscuro"), ("medium", "medio"), ("light", "claro"))


def _num(x: float, sign: bool = False) -> str:
    return (f"{x:+.4f}" if sign else f"{x:.4f}").replace(".", ",")


def _row(doc: dict, arm: str, control: str, metric: str) -> dict:
    return next(r for r in doc["paired"] if r["arm"] == arm and r["control"] == control and r["metric"] == metric)


def _contrast(r: dict) -> str:
    verdict = "distinguible" if r["excludes_zero"] else "indistinguible"
    return f"{_num(r['point_delta'], True)} [{_num(r['ci95_low'], True)}; {_num(r['ci95_high'], True)}] {verdict}"


def _contrast_table(doc: dict, suffix: str) -> list[str]:
    lines = ["| Peso | Configuración SoG | " + " | ".join(METRICS.values()) + " |", "|---|---|---|---|"]
    for key, label in WEIGHTS:
        arm, control = f"{key}_{suffix}", f"{key}_id"
        cells = " | ".join(_contrast(_row(doc, arm, control, m)) for m in METRICS)
        lines.append(f"| {label} | `{doc['entries'][arm]['config']}` | {cells} |")
    return lines


def _tercile_table(doc: dict) -> list[str]:
    lines = ["| Entrada | " + " | ".join(t for _, t in TERCILES) + " |", "|---|---|---|---|"]
    for key, label in WEIGHTS:
        for entry, name in ((f"{key}_id", "sin SoG"), (f"{key}_sog", "SoG")):
            values = " | ".join(_num(doc["terciles"][entry][t]) for t, _ in TERCILES)
            lines.append(f"| {label}, {name} | {values} |")
    return lines


def _verdicts(doc: dict) -> list[str]:
    v = doc["verdicts"]
    p1 = ", ".join(f"{label} {'sí' if v['P1']['per_arm'][k] else 'no'}" for k, label in WEIGHTS if k != "public")
    p2 = ", ".join(f"{label} {_num(v['P2']['abs_delta'][k])}" for k, label in WEIGHTS)
    p3 = ", ".join(f"{label} {'sí' if v['P3']['per_arm'][k] else 'no'}" for k, label in WEIGHTS)
    word = {True: "se cumple", False: "no se cumple"}
    return [
        f"- **P1: {word[v['P1']['holds']]}** (|Δ macro-F1| ≤ {_num(t4_sog.P1_MAX_ABS_DELTA)} con IC que incluye 0; por peso: {p1}).",
        f"- **P2: {word[v['P2']['holds']]}** (|Δ| del público mayor que el de A* y B*; |Δ|: {p2}).",
        f"- **P3: {word[v['P3']['holds']]}** (|Δ brecha| ≤ {_num(t4_sog.P3_MAX_ABS_DELTA)} con IC que incluye 0; por peso: {p3}).",
    ]


def render(doc: dict) -> str:
    counts = doc["tercile_counts"]
    repro = max(abs(x) for per in doc["v71_reproduction_delta"].values() for x in per.values())
    lines = [
        "# T4 — Shades of Gray (p = 6) antes del extractor, lectura única de val",
        "",
        "Generado por `scripts/build_t4_table.py` desde `t4_final_eval.json`; no editar a mano. "
        "Registro, predicciones y salvedades: docs/31. Los terciles son terciles de luminancia, no tono de piel.",
        "",
        f"Val: n = {doc['n_eval']} (oscuro {counts['dark']}, medio {counts['medium']}, claro {counts['light']}, "
        f"sin tercil {counts['unavailable']}). Bootstrap pareado: {doc['bootstrap']['n_resamples']} remuestreos, "
        f"semilla {doc['bootstrap']['rng_seed']}. Reproducción de v7.1 sin SoG: máx. |Δ| = {_num(repro)}.",
        "",
        "## Veredictos registrados",
        "",
        *_verdicts(doc),
        "",
        "## Contrastes primarios: SoG contra sin SoG en el protocolo fijo",
        "",
        *_contrast_table(doc, "sog"),
        "",
        "## Contrastes secundarios: ganador SoG del holdout contra sin SoG",
        "",
        *_contrast_table(doc, "sogwin"),
        "",
        "## Macro-F1 por tercil (descriptivo)",
        "",
        *_tercile_table(doc),
        "",
        "Potencia: el tercil claro tiene pocas imágenes, así que un IC de la brecha que incluye el 0 se lee "
        "«no se detecta», no «no hay efecto».",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("doc", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(json.loads(args.doc.read_text(encoding="utf-8")))
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else None
        if current != text:
            print(f"{args.out} desactualizado: regenerar con scripts/build_t4_table.py", file=sys.stderr)
            return 1
        return 0
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
