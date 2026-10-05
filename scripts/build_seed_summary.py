"""Combine the v7 single evaluations of every registered seed into one generated table.

docs/28 4.1 fixes how the replicas are read: each seed on its own, then the
mean, no seed chosen, and P3-IC called "no robusto a la semilla" if the seeds
disagree. With two seeds the spread is the signed s43 - s42 difference; with
more it is the range, max - min (docs/29 2.5). The seeds are
``dermssl.v7_seeds.REGISTERED_SEEDS``; ``--check`` fails if the table drifts.

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
from dermssl import v7_seeds as seeds  # noqa: E402

V7 = REPO / seeds.V7_RESULTS
SOURCES = {seed: REPO / seeds.final_eval_path(seed) for seed in seeds.REGISTERED_SEEDS}
TARGET = V7 / "SEMILLAS.md"
LABELS = {"macro_f1": "Macro-F1", "balanced_accuracy": "Bal-acc", "auc": "AUC", "ece": "ECE"}
ROLES = ("plain", "proposed", "public")
MET = "cumplida"


def _num(x: float) -> str:
    return f"{x:+.4f}".replace(".", ",") if x < 0 or x > 0 else "0,0000"


def _val(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def _summary(values: dict[int, float]) -> dict:
    """Per-seed values, the mean, and the spread: signed difference of two, range of more."""
    ordered = [values[seed] for seed in sorted(values)]
    out = {f"s{seed}": values[seed] for seed in sorted(values)}
    out["mean"] = sum(ordered) / len(ordered)
    if len(ordered) == 2:
        out["diff"] = ordered[1] - ordered[0]
    else:
        out["range"] = max(ordered) - min(ordered)
    return out


def _spread_header(seed_list: list[int]) -> str:
    return f"s{seed_list[1]} − s{seed_list[0]}" if len(seed_list) == 2 else "Rango (máx − mín)"


def arm_rows(docs: dict[int, dict]) -> dict:
    """``rows[role][metric]`` -> per-seed values, mean and spread."""
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


def _row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def _sep(n: int) -> str:
    return "|" + "---|" * n


def _sources_sentence(seed_list: list[int]) -> str:
    base = seed_list[0]
    parts = [f"`{'' if s == base else f's{s}/'}final_eval.json` (semilla {s})" for s in seed_list]
    return f"Generado desde {seeds.join_es(parts)}. "


def render(docs: dict[int, dict]) -> str:
    seed_list = sorted(docs)
    base, tags = docs[seed_list[0]], [f"s{s}" for s in seed_list]
    arms, n_word = base["arms"], seeds.count_word(len(seed_list))
    spread = "diff" if len(seed_list) == 2 else "range"
    lines = [f"# v7: las {n_word} semillas", "",
             _sources_sentence(seed_list)
             + "No editar a mano: `--check` lo compara en CI. Regla de lectura en docs/28 §4.1.", "",
             _row(["Brazo", "Métrica", *tags, "Media", _spread_header(seed_list)]),
             _sep(len(tags) + 4)]
    for role, metrics in arm_rows(docs).items():
        for m, r in metrics.items():
            lines.append(_row([arms[role], LABELS[m], *(_val(r[t]) for t in tags), _val(r["mean"]),
                               _num(r[spread])]))
    lines += ["", _row(["Contraste", "Métrica", *(f"{t} [IC 95 %]" for t in tags), "Media"]),
              _sep(len(tags) + 3)]
    for (arm, control, m), recs in contrast_rows(docs).items():
        cells = [f"{_num(r['point_delta'])} [{_num(r['ci95_low'])}; {_num(r['ci95_high'])}]" for r in recs.values()]
        mean = sum(r["point_delta"] for r in recs.values()) / len(recs)
        lines.append(_row([f"{arms[arm]} − {arms[control]}", LABELS[m], *cells, _num(mean)]))
    lines += ["", _row(["Predicción", *tags]), _sep(len(tags) + 1)]
    for name in base["predictions"]:
        lines.append(_row([name, *(docs[s]["predictions"][name]["verdict"] for s in seed_list)]))
    lines += ["", f"P3-IC con las {n_word} semillas: {robustness(docs)}."]
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
