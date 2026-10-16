"""Render the docs/30 2-3 first pass (PAD-UFES-20 zero-shot) from its JSON; ``--check`` for CI.

    python scripts/build_pad_ufes_table.py \
        resultados/clasificacion/pad_ufes/pad_ufes_eval.json \
        --out resultados/clasificacion/pad_ufes/README.md [--check]

Every number, verdict and direction is read from the JSON (and the thread
pin from its ``.environment.json`` sidecar); nothing is re-derived. A contrast
reads "mejora" only where the JSON's direction is ``arm_better`` and the
reading carries a verdict; the phototype strata and the exploratory reading
say "a favor de" at most, or "no calculable (k/n)" (docs/30 6, rule (b)).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import pad_ufes as pu  # noqa: E402
from dermssl import pad_ufes_readings as pr  # noqa: E402
from dermssl import run_environment  # noqa: E402

METRICS = ("macro_f1", "balanced_accuracy", "auc", "ece")
METRIC_LABELS = {"macro_f1": "Macro-F1", "balanced_accuracy": "Balanced acc.", "auc": "AUC", "ece": "ECE"}
READING_LABELS = {"primary": "Lectura primaria (bcc, mel, nv)",
                  "secondary": "Lectura secundaria (akiec = ACK + SCC, bcc, mel, nv)",
                  "exploratory": "Lectura exploratoria (akiec, bcc, bkl = SEK, mel, nv), sin veredicto"}
#: docs/30 3: which paired contrast each registered prediction reads (arm, control, metric).
VERDICTS = {"PAD-P1": ("proposed", pr.PUBLIC, "macro_f1"), "PAD-P2": ("plain", pr.PUBLIC, "macro_f1"),
            "PAD-P3": ("proposed", pr.PUBLIC, "ece"), "PAD-P4": ("proposed", "plain", "macro_f1")}
VERDICT_TEXT = {"PAD-P1": "Δ puntual macro-F1 B\\* − público ≤ 0 con las dos semillas",
                "PAD-P2": "Δ puntual macro-F1 A\\* − público ≤ 0 con las dos semillas",
                "PAD-P3": "Δ puntual ECE B\\* − público < 0 con las dos semillas",
                "PAD-P4": "IC de macro-F1 B\\* − A\\* cruza el 0 con las dos semillas"}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")  # GFM splits cells on | even inside code


def _num(x) -> str:
    return "n/d" if x is None else f"{x:.4f}".replace(".", ",")


def _k(k: int, n_resamples: int) -> str:
    return f"no calculable ({k}/{n_resamples})"


def _arm_cell(arm: dict, metric: str, n_resamples: int) -> str:
    k = arm.get("n_nonfinite", {}).get(metric, 0)
    low, high = (None, None) if k else arm["ci95"][metric]
    return f"{_num(arm[metric])} " + (_k(k, n_resamples) if k else f"[{_num(low)}; {_num(high)}]")


def _direction(row: dict, n_resamples: int, with_verdict: bool) -> str:
    direction = row["direction"]
    if direction == pr.NOT_CALCULABLE:
        return _k(row["n_nonfinite"], n_resamples)
    if direction == "arm_better":
        side = "mejora de" if with_verdict else "a favor de"
        return f"{side} `{row['arm']}` (IC excluye 0)"
    if direction == "control_better":
        return f"a favor de `{row['control']}` (IC excluye 0)"
    return "indistinguible"


def _ci(row: dict) -> str:
    return "n/d" if row["ci95_low"] is None else f"[{_num(row['ci95_low'])}; {_num(row['ci95_high'])}]"


def _arms_table(entry: dict, n_resamples: int, outside: dict | None = None) -> list[str]:
    head = ["Modelo", *(METRIC_LABELS[m] + " [IC 95 %]" for m in METRICS)]
    head += ["Argmax fuera de la lectura"] if outside is not None else []
    lines = ["| " + " | ".join(head) + " |", "|---|" + "---|" * (len(head) - 1)]
    for name, arm in entry["arms"].items():
        cells = [f"`{name}`", *(_arm_cell(arm, m, n_resamples) for m in METRICS)]
        cells += [_num(outside[name])] if outside is not None else []
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _contrasts_table(rows: list[dict], n_resamples: int, with_verdict: bool) -> list[str]:
    lines = ["| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |", "|---|---|---:|---|---|"]
    for row in rows:
        lines.append(f"| `{row['arm']}` − `{row['control']}` | {METRIC_LABELS[row['metric']]} "
                     f"| {_num(row['point_delta'])} | {_ci(row)} | {_direction(row, n_resamples, with_verdict)} |")
    return lines


def _row(entry: dict, arm: str, control: str, metric: str) -> dict:
    return next(r for r in entry["paired"]
                if r["arm"] == arm and r["control"] == control and r["metric"] == metric)


def _verdict_rows(entry: dict, n_resamples: int) -> list[str]:
    lines = ["| Predicción | Semilla | Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |",
             "|---|---|---|---|---:|---|---|"]
    for name, (arm, control, metric) in VERDICTS.items():
        for seed in pr.SEEDS:
            row = _row(entry, pr.role(arm, seed), control if control == pr.PUBLIC else pr.role(control, seed), metric)
            lines.append(f"| {name} | s{seed} | `{row['arm']}` − `{row['control']}` | {METRIC_LABELS[metric]} "
                         f"| {_num(row['point_delta'])} | {_ci(row)} | {_direction(row, n_resamples, True)} |")
    return lines


def _verdicts(entry: dict, n_resamples: int) -> list[str]:
    recorded = entry["verdicts"]
    lines = ["Veredictos registrados (docs/30 §3), copiados del JSON:", ""]
    lines += [f"- **{name}**: {recorded[name]} ({VERDICT_TEXT[name]})" for name in VERDICTS]
    robust = [f"- `{_cell(key)}`: {'sí' if value else 'no'}" for key, value in recorded["mejora_robusta"].items()]
    return [*lines, "", *_verdict_rows(entry, n_resamples), "",
            "Mejora robusta a la semilla (IC excluye 0 a favor del brazo con las dos semillas, según el JSON):",
            "", *robust]


def _readings(doc: dict, n_resamples: int) -> list[str]:
    lines = []
    for name, entry in doc["readings"].items():
        with_verdict = name in pu.READINGS_WITH_VERDICT
        lines += ["", f"## {READING_LABELS[name]}", "",
                  f"{entry['n_images']} imágenes, {entry['n_patients']} pacientes.", ""]
        lines += [*_verdicts(entry, n_resamples), ""] if "verdicts" in entry else []
        lines += [*_arms_table(entry, n_resamples, entry["outside_rate"]), "",
                  *_contrasts_table(entry["paired"], n_resamples, with_verdict)]
    return lines


def _strata(doc: dict, n_resamples: int) -> list[str]:
    strata = doc["fst_strata"]
    lines = ["", "## Estratos por fototipo (lectura primaria, exploratorio, sin veredicto)", "",
             "Regla (b) de docs/30 §6: el valor puntual se reporta siempre; si alguna remuestra da un valor no "
             f"finito, el IC queda \"no calculable\" con el número exacto k/{n_resamples} de remuestras no finitas. "
             f"Un estrato con menos de {strata['min_n']} imágenes lleva el rótulo \"{pr.NO_POWER}\"."]
    for name, entry in strata["bins"].items():
        lines += ["", f"### {name}", ""]
        if not entry["n_images"]:
            lines.append(f"Sin imágenes ({entry['power']}).")
            continue
        power = "" if entry["power"] == "ok" else f" Rótulo: {entry['power']}."
        lines += [f"{entry['n_images']} imágenes, {entry['n_patients']} pacientes.{power}", "",
                  *_arms_table(entry, n_resamples), "", *_contrasts_table(entry["paired"], n_resamples, False)]
    return lines


def _threads(environment: dict | None) -> str:
    if environment is None:
        return "Sin sidecar de entorno: hilos no registrados."
    env = environment.get("env", {})
    return "Hilos: " + ", ".join(f"{var}={env.get(var)}" for var in THREAD_VARS) + " (sidecar de entorno)."


def _fst_caveat(note: str) -> str:
    """The JSON's note, plus the literal thesis sentence unless the note already ends with it."""
    return note if pu.THESIS_FST_SENTENCE in note else f"{note} {pu.THESIS_FST_SENTENCE}."


def _header(doc: dict, environment: dict | None, n_resamples: int) -> list[str]:
    full = doc["readings"]["exploratory"]
    models = [f"- `{role}`: `{_cell(spec)}`" for role, spec in doc["specs"].items()]
    return [
        "# PAD-UFES-20 zero-shot, primera pasada (docs/30 §2-3)", "",
        "Generado por `scripts/build_pad_ufes_table.py` desde `pad_ufes_eval.json`; no editar a mano (`--check` "
        "lo compara y `tests/test_build_pad_ufes_table.py` lo corre en CI). Protocolo y predicción: docs/30 §2-3. "
        f"`code_revision` {doc['code_revision']}, registro `{doc['registration_commit']}`.", "",
        f"PAD completo: {full['n_images']} imágenes, {full['n_patients']} pacientes. Probe `{doc['probe']}`. "
        f"Bootstrap pareado agrupado por paciente, semilla {doc['bootstrap']['rng_seed']}, {n_resamples} remuestras. "
        f"{_threads(environment)}", "",
        f"Fototipo: {_fst_caveat(doc['fst_strata']['note'])}", "",
        "## Modelos", "", *models,
    ]


from dermssl.display_nomenclature import presentation  # noqa: E402


@presentation
def render(doc: dict, environment: dict | None = None) -> str:
    n_resamples = doc["bootstrap"]["n_resamples"]
    lines = [*_header(doc, environment, n_resamples), *_readings(doc, n_resamples), *_strata(doc, n_resamples)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("document", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    doc = json.loads(args.document.read_text(encoding="utf-8"))
    text = render(doc, run_environment.read_sidecar(args.document))
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} no coincide con {args.document}; regenerar con scripts/build_pad_ufes_table.py")
            return 1
        return 0
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
