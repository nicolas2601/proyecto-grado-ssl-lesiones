"""Render the v7 single-evaluation table from its JSON; ``--check`` for CI.

    python scripts/build_final_eval_table.py resultados/clasificacion/v7/final_eval.json \
        --out resultados/clasificacion/v7/README.md [--check]

If ``eval_provenance.json`` sits beside the evaluation (s44: control rerun,
aborted start, sibling-captured environment), its facts are rendered too.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import final_eval as fe  # noqa: E402

PROVENANCE = "eval_provenance.json"
LABELS = {"macro_f1": "Macro-F1", "balanced_accuracy": "Bal-acc", "auc": "AUC", "ece": "ECE"}


def _num(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def _ci(low: float, high: float) -> str:
    return f"[{_num(low)}; {_num(high)}]"


def render(doc: dict) -> str:
    arms = doc["arms"]
    lines = [
        f"# Evaluación única v7 ({doc['split']})",
        "",
        f"Probe `{doc['probe']}`, n = {doc['n_eval']}, `code_revision` {doc['code_revision']}. "
        "Generado desde `final_eval.json`. No editar a mano: `--check` lo compara en CI.",
        "",
        "| Brazo | " + " | ".join(LABELS[m] for m in fe.METRICS) + " |",
        "|---|" + "---|" * len(fe.METRICS),
    ]
    for role in ("plain", "proposed", "public"):
        r = doc["results"][role]
        cells = [f"{_num(r[m])} {_ci(*r['ci95'][m])}" for m in fe.METRICS]
        lines.append(f"| {arms[role]} | " + " | ".join(cells) + " |")
    lines += ["", "| Contraste | " + " | ".join(f"Δ {LABELS[m]}" for m in fe.METRICS) + " |",
              "|---|" + "---|" * len(fe.METRICS)]
    paired = fe.paired_from_rows(doc["paired"])
    for arm, control in fe.CONTRASTS:
        rec = paired[(arm, control)]
        cells = [f"{_num(rec[m]['point_delta'])} {_ci(rec[m]['ci95_low'], rec[m]['ci95_high'])}"
                 for m in fe.METRICS]
        lines.append(f"| {arms[arm]} − {arms[control]} | " + " | ".join(cells) + " |")
    lines += ["", "| Predicción (docs/28 §5) | Criterio | Veredicto |", "|---|---|---|"]
    for name, pred in doc["predictions"].items():
        lines.append(f"| {name} | {pred['criterion']} | {pred['verdict']} |")
    if "secondary_ece_basic_ci" in doc:
        lines += _ece_basic_lines(doc["secondary_ece_basic_ci"], arms)
    return "\n".join(lines) + "\n"


def _ece_basic_lines(extra: dict, arms: dict) -> list[str]:
    lines = ["", f"## ECE con IC basic (lectura {extra['reading']})", "", extra["note"], "",
             "| Brazo o contraste | ECE / Δ ECE | IC basic | ¿Punto fuera del IC percentil? |",
             "|---|---|---|---|"]
    for role, rec in extra["arms"].items():
        lines.append(f"| {arms[role]} | {_num(rec['point'])} | {_ci(rec['ci95_low'], rec['ci95_high'])} | "
                     f"{'sí' if rec['point_outside_percentile'] else 'no'} |")
    for key, rec in extra["contrasts"].items():
        arm, control = key.split("-")
        lines.append(f"| {arms[arm]} − {arms[control]} | {_num(rec['point_delta'])} | "
                     f"{_ci(rec['ci95_low'], rec['ci95_high'])} | "
                     f"{'sí' if rec['point_outside_percentile'] else 'no'} |")
    lines += ["", f"P2 leída con el IC basic: {extra['P2_with_basic']} (el veredicto primario no cambia)."]
    return lines


def _when(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _clock(stamp: str) -> str:
    """HH:MM or HH:MM:SS, as precise as the recorded stamp and no more."""
    return f"{_when(stamp):%H:%M:%S}" if stamp.count(":") == 2 else f"{_when(stamp):%H:%M}"


def _span(start: str, end: str) -> str:
    return f"{_when(start):%Y-%m-%d} {_clock(start)} a {_clock(end)} UTC"


def _duration(start: str, end: str) -> str:
    minutes, seconds = divmod(int((_when(end) - _when(start)).total_seconds()), 60)
    return f"{minutes} min {seconds} s"


def _env_text(env: dict) -> str:
    return ", ".join(f"`{name}` sin fijar" if value is None else f"`{name}={value}`" for name, value in env.items())


def _ticks(names) -> str:
    return ", ".join(f"`{name}`" for name in names)


def provenance_lines(prov: dict) -> list[str]:
    ev, tr, ctl, ab, side = (prov[k] for k in ("evaluation", "training", "control", "aborted_start",
                                               "environment_sidecar"))
    clean = "limpio" if ev["clean_tree"] else "con cambios sin commitear"
    val_read = "no cuenta" if not ab["counts_as_val_read"] else "cuenta"
    written = _ticks(ab["outputs_written"]) if ab["outputs_written"] else "sin probas ni métricas, ningún JSON"
    return [
        "", "## Procedencia", "", f"Generado desde `{PROVENANCE}`.", "",
        f"- **Evaluación (semilla {prov['seed']}).** `{ev['command']}` desde el clon `{ev['clone']}` en "
        f"{ev['code_revision']} ({clean}), {_env_text(ev['launch_env'])}; {_span(ev['start_utc'], ev['end_utc'])}. "
        f"Log: `{ev['log']}`.",
        f"- **Entrenamiento.** Las finales de s{prov['seed']} se entrenaron en {tr['code_revision']}; las de "
        + ", ".join(f"s{s}" for s in tr["reference_seeds"]) + f" en {tr['reference_revision']}. "
        f"`{tr['guard']}` exige que el entrenamiento sea el mismo.",
        f"- **Control (semilla {ctl['seed']}), antes de la evaluación.** El mismo comando y el mismo entorno, "
        f"{_span(ctl['start_utc'], ctl['end_utc'])}. Contra `{ctl['reference']}`: las probabilidades "
        f"{_ticks(ctl['probas_bit_equal'])} son idénticas bit a bit"
        + (" (dtype incluido)" if ctl["dtype_equal"] else "")
        + f", y {_ticks(ctl['json_identical'])} del JSON son idénticos. Log: `{ctl['log']}`.",
        f"- **Arranque abortado.** {_span(ab['start_utc'], ab['killed_utc'])}: arranque abortado a los "
        f"{_duration(ab['start_utc'], ab['killed_utc'])}, antes de puntuar ({ab['stage']}): {written}; "
        f"{val_read} como lectura de val. Carpeta conservada como evidencia en el pod: `{ab['pod_folder']}`. "
        f"Log: `{ab['log']}`.",
        f"- **Entorno.** `{side['file']}` se capturó en {side['captured_in']}."
        + (f" {side['launch_env_difference']}" if "launch_env_difference" in side else ""),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("document", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(json.loads(args.document.read_text(encoding="utf-8")))
    prov_path = args.document.parent / PROVENANCE
    if prov_path.is_file():
        text += "\n".join(provenance_lines(json.loads(prov_path.read_text(encoding="utf-8")))) + "\n"
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} no coincide con {args.document}; regenerar")
            return 1
        return 0
    args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
