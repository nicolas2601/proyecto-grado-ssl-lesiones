"""Render the v7.1 single-evaluation table from its JSON; ``--check`` for CI.

    python scripts/build_v71_table.py resultados/clasificacion/v71/v71_final_eval.json \
        --out resultados/clasificacion/v71/README.md [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import v71_plan as plan  # noqa: E402

LABELS = {"A71": "A* v7.1", "B71": "B* v7.1", "public_A71": "Público, protocolo de A* v7.1",
          "public_B71": "Público, protocolo de B* v7.1", "A_v7": "A* v7 (primario)",
          "B_v7": "B* v7 (primario)", "public_v7": "Público v7 (primario)",
          "public71": "Público, su mejor protocolo"}
MAIN = {"macro_f1": "Macro-F1", "balanced_accuracy": "Bal-acc", "accuracy": "Acc", "auc": "AUC",
        "ece": "ECE", "mel_recall": "Sens. mel", "mel_auc": "AUC mel"}
OPERATING = {"spec80": "esp. 80 %", "spec90": "esp. 90 %", "sens95": "sens. 95 %"}
CONTRAST = {"macro_f1": "Macro-F1", "balanced_accuracy": "Bal-acc", "auc": "AUC", "ece": "ECE",
            "mel_recall": "Sens. mel", "mel_auc": "AUC mel"}


def _num(x) -> str:
    return "n/d" if x is None else f"{x:.4f}".replace(".", ",")


def _ci(pair) -> str:
    return f"[{_num(pair[0])}; {_num(pair[1])}]"


def _cell(result: dict, metric: str) -> str:
    return f"{_num(result[metric])} {_ci(result['ci95'][metric])}"


def _main_table(doc: dict) -> list[str]:
    lines = ["| Lectura | Configuración | " + " | ".join(MAIN.values()) + " |",
             "|---|---|" + "---|" * len(MAIN)]
    for label in plan.VAL_LABELS:
        r = doc["results"][label]
        lines.append(f"| {LABELS[label]} | `{r['config']}` | "
                     + " | ".join(_cell(r, m) for m in MAIN) + " |")
    return lines


def _operating_table(doc: dict) -> list[str]:
    head = " | ".join(f"Sens. / esp. ({v})" for v in OPERATING.values())
    lines = ["", "## Puntos de operación de melanoma (umbral fijado en el holdout)", "",
             doc["note_operating_points"], "",
             f"| Lectura | {head} |", "|---|" + "---|" * len(OPERATING)]
    for label in plan.VAL_LABELS:
        r = doc["results"][label]
        cells = [f"{_cell(r, f'mel_sens_at_thr_{k}')} / {_cell(r, f'mel_spec_at_thr_{k}')}" for k in OPERATING]
        lines.append(f"| {LABELS[label]} | " + " | ".join(cells) + " |")
    return lines


def _contrast_table(doc: dict) -> list[str]:
    rows = {(r["arm"], r["control"], r["metric"]): r for r in doc["paired"]}
    lines = ["", "## Contrastes pareados (bootstrap, semilla 20260923)", "",
             "| Contraste | " + " | ".join(f"Δ {v}" for v in CONTRAST.values()) + " |",
             "|---|" + "---|" * len(CONTRAST)]
    for arm, control in plan.VAL_CONTRASTS:
        cells = []
        for metric in CONTRAST:
            r = rows[(arm, control, metric)]
            cells.append(f"{_num(r['point_delta'])} {_ci((r['ci95_low'], r['ci95_high']))}")
        lines.append(f"| {LABELS[arm]} − {LABELS[control]} | " + " | ".join(cells) + " |")
    return lines


def render(doc: dict) -> str:
    phase2 = doc["selection"]["phase2"]
    mel = ", ".join(f"{arm} {_num(v)}" for arm, v in phase2["mel_sens_at_spec80_holdout"].items())
    lines = [
        f"# Lectura v7.1 ({doc['split']}, docs/29)", "",
        f"n = {doc['n_eval']}, `code_revision` {doc['code_revision']}. Generado desde "
        "`v71_final_eval.json`; no editar a mano (`--check` lo compara en CI). La selección se hizo "
        "en el holdout interno y validación se leyó una sola vez. El primario v7 no cambia.", "",
        *_main_table(doc), *_operating_table(doc), *_contrast_table(doc), "",
        f"Fase 2 (docs/29 §2.7 ítem 7): sensibilidad de mel a especificidad 80 % en el holdout: {mel}. "
        f"Habilitada para proponer: {'sí' if phase2['enabled'] else 'no'}.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("document", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = render(json.loads(args.document.read_text(encoding="utf-8")))
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
