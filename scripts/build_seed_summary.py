"""Combine the v7 single evaluations of both seeds into one generated table.

docs/28 4.1 fixes how the replica is read: each seed on its own, then the
mean of both with the s43 - s42 difference, no seed chosen, and P3-IC called
"no robusto a la semilla" if the seeds disagree. The numbers come from the
two committed final_eval.json; ``--check`` fails if the table drifts.

    python scripts/build_seed_summary.py            # write
    python scripts/build_seed_summary.py --check    # CI
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import final_eval as fe  # noqa: E402

V7 = REPO / "resultados" / "clasificacion" / "v7"
SOURCES = {42: V7 / "final_eval.json", 43: V7 / "s43" / "final_eval.json"}
TARGET = V7 / "SEMILLAS.md"
LABELS = {"macro_f1": "Macro-F1", "balanced_accuracy": "Bal-acc", "auc": "AUC", "ece": "ECE"}
ROLES = ("plain", "proposed", "public")
MET = "cumplida"


def _num(x: float) -> str:
    return f"{x:+.4f}".replace(".", ",") if x < 0 or x > 0 else "0,0000"


def _val(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def _summary(values: dict[int, float]) -> dict:
    s42, s43 = values[42], values[43]
    return {"s42": s42, "s43": s43, "mean": (s42 + s43) / 2, "diff": s43 - s42}


def arm_rows(docs: dict[int, dict]) -> dict:
    """``rows[role][metric]`` -> per-seed values, mean and s43 - s42."""
    return {role: {m: _summary({seed: doc["results"][role][m] for seed, doc in docs.items()})
                   for m in fe.METRICS}
            for role in ROLES}


def contrast_rows(docs: dict[int, dict]) -> dict:
    out = {}
    for arm, control in fe.CONTRASTS:
        for metric in fe.METRICS:
            recs = {seed: fe.paired_from_rows(doc["paired"])[(arm, control)][metric] for seed, doc in docs.items()}
            out[(arm, control, metric)] = recs
    return out


def robustness(docs: dict[int, dict]) -> str:
    verdicts = {doc["predictions"]["P3-IC"]["verdict"] for doc in docs.values()}
    return "robusto a la semilla" if verdicts == {MET} else "no robusto a la semilla"


def render(docs: dict[int, dict]) -> str:
    arms = docs[42]["arms"]
    lines = ["# v7: las dos semillas", "",
             "Generado desde `final_eval.json` (semilla 42) y `s43/final_eval.json` (semilla 43). "
             "No editar a mano: `--check` lo compara en CI. Regla de lectura en docs/28 §4.1.", "",
             "| Brazo | Métrica | s42 | s43 | Media | s43 − s42 |", "|---|---|---|---|---|---|"]
    for role, metrics in arm_rows(docs).items():
        for m, r in metrics.items():
            lines.append(f"| {arms[role]} | {LABELS[m]} | {_val(r['s42'])} | {_val(r['s43'])} | "
                         f"{_val(r['mean'])} | {_num(r['diff'])} |")
    lines += ["", "| Contraste | Métrica | s42 [IC 95 %] | s43 [IC 95 %] | Media |", "|---|---|---|---|---|"]
    for (arm, control, m), recs in contrast_rows(docs).items():
        cells = [f"{_num(r['point_delta'])} [{_num(r['ci95_low'])}; {_num(r['ci95_high'])}]" for r in recs.values()]
        mean = sum(r["point_delta"] for r in recs.values()) / len(recs)
        lines.append(f"| {arms[arm]} − {arms[control]} | {LABELS[m]} | {cells[0]} | {cells[1]} | {_num(mean)} |")
    lines += ["", "| Predicción | s42 | s43 |", "|---|---|---|"]
    for name in docs[42]["predictions"]:
        lines.append(f"| {name} | {docs[42]['predictions'][name]['verdict']} | {docs[43]['predictions'][name]['verdict']} |")
    lines += ["", f"P3-IC con las dos semillas: {robustness(docs)}."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    docs = {seed: json.loads(path.read_text(encoding="utf-8")) for seed, path in SOURCES.items()}
    text = render(docs)
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != text:
            print(f"{TARGET} esta desactualizado: corre scripts/build_seed_summary.py")
            return 1
        return 0
    TARGET.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
