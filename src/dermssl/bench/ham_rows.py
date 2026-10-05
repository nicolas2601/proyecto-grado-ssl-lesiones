"""HAM10000 rows beyond the 8 baselines: v7.1 (proposed B71 and its arms), F4 exploratory, pending E1 cells.

B71 is the proposed model and its canonical compute is the v7.1 one (batch 64 + autocast, "S1"
in docs/33). F4 (segmentation) is exploratory: it never changes the proposed model.
"""

from __future__ import annotations

from pathlib import Path

from dermssl.bench import pending
from dermssl.bench.contrast import num
from dermssl.bench.rows import (EXPLORATORY, FROZEN, OK, OWN_A, OWN_B, PENDING, PLAN_DOC, V71_JSON, V71_NPZ, V71_ROWS,
                                Row, Value, _load, v71_derived)

E3_DIR = "resultados/banco_completo/e3"
E3B_CANONICAL = "resultados/banco_completo/e3b_bs64/e3b_summary.json"
COMPUTES = (("S0", "batch 16 + autocast", f"{E3_DIR}/e3_final_eval.json"),
            ("S1", "batch 64 + autocast (canónico, el de v7.1)", f"{E3_DIR}/robustez/e3_s1_bs64_autocast/e3_final_eval.json"),
            ("S2", "batch 64, fp32", f"{E3_DIR}/robustez/e3_s2_bs64_fp32/e3_final_eval.json"))
F4_ARM = "B71+F4_tta"
F4_SEGMENTATION = "sí (F4, exploratorio)"
F4_METRICS = {"auc": "auc_ovr_macro"}


def compute_values(repo: Path, model: str) -> dict[str, float]:
    """Macro-F1 of ``model`` ('B71' or 'F4') under S0, S1 and S2, read from the three final JSON."""
    return {tag: _load(repo, rel)["rows"][model]["macro_f1"] for tag, _, rel in COMPUTES}


def b71_note(repo: Path) -> str:
    vals = compute_values(repo, "B71")
    shown = " / ".join(num(v) for v in vals.values())
    return ("probe logreg C1; precisión, recall y MCC desde las probabilidades; cómputo canónico = S1 "
            f"(batch 64 + autocast, el de v7.1). F1 macro de B71 en S0/S1/S2: {shown} "
            "(el cómputo de extracción mueve la cifra)")


def proposed_rows(repo: Path) -> list[Row]:
    n_eval = _load(repo, V71_JSON)["n_eval"]
    out = []
    for key, name, pre in V71_ROWS:
        note = b71_note(repo) if key == "B71" else "probe logreg C1; precisión, recall y MCC desde las probabilidades"
        out.append(Row("ham10000", FROZEN, name, pre, False, False, OK, (V71_JSON, V71_NPZ), n_eval,
                       v71_derived(repo, key)[1], note))
    return out


def f4_values(record: dict) -> dict[str, Value]:
    ci = record.get("ci95_grouped", {})
    keys = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "auc", "ece", "mcc")
    return {k: Value(record[F4_METRICS.get(k, k)], tuple(ci[k]) if k in ci else None) for k in keys}


def f4_macro_f1(repo: Path, rel: str) -> dict:
    """The paired F4 - B71 macro-F1 contrast (all lesions) stored in one final JSON."""
    return next(c for c in _load(repo, rel)["contrasts"] if c["metric"] == "macro_f1" and c["subset"] == "all")


def f4_note(repo: Path) -> str:
    s1 = f4_macro_f1(repo, COMPUTES[1][2])
    excluding = sum(f4_macro_f1(repo, rel)["excludes_zero"] for _, _, rel in COMPUTES)
    verdict = ("exploratorio, no demostrable de forma robusta" if excluding < len(COMPUTES)
               else "exploratorio, mejora marginal robusta al cómputo")
    return (f"{verdict}: Δ F1 macro frente a B71 "
            f"{num(s1['point_delta'], True)} [{num(s1['ci95_low'], True)}; {num(s1['ci95_high'], True)}] en el cómputo "
            f"canónico; el IC pareado excluye 0 en {excluding} de {len(COMPUTES)} cómputos; "
            "no reemplaza al propuesto (TABLA_ABLACION.md)")


def f4_row(repo: Path) -> Row:
    record = _load(repo, E3B_CANONICAL)["proposed"]["rows"][F4_ARM]
    return Row("ham10000", EXPLORATORY, "B71 + F4 (exploratorio)", OWN_B, False, False, OK,
               (E3B_CANONICAL, COMPUTES[1][2]), record["n"], f4_values(record), f4_note(repo), F4_SEGMENTATION)


def pending_rows() -> list[Row]:
    return [Row("ham10000", FROZEN, "B71-limpio (canónico)", OWN_B, False, False, PENDING, (PLAN_DOC,),
                note=pending.HAM_CLEAN.note),
            Row("ham10000", FROZEN, "A71-limpio (canónico)", OWN_A, False, False, PENDING, (PLAN_DOC,),
                note=pending.HAM_CLEAN.note)]


def ham_rows(repo: Path) -> list[Row]:
    return proposed_rows(repo) + [f4_row(repo)] + pending_rows()
