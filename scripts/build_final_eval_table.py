"""Render the v7 single-evaluation table from its JSON; ``--check`` for CI.

    python scripts/build_final_eval_table.py resultados/clasificacion/v7/final_eval.json \
        --out resultados/clasificacion/v7/README.md [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import final_eval as fe  # noqa: E402

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
