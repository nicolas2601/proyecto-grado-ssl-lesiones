"""Verdicts for the three predictions registered in docs/28 5.

The thresholds live here, next to the tests that pin them, so the verdict
of the single evaluation is computed and not read off a table. Arms are
named by role: ``plain`` (A*), ``proposed`` (B*) and ``public`` (DINOv2).

* P1: plain vs public, macro-F1 point delta >= -0.01. If the interval also
  excludes zero upward, the anchored adaptation improves on public.
* P2: proposed vs plain, ECE better with the interval excluding zero AND
  macro-F1 point delta >= -0.01.
* P3: proposed vs public, macro-F1 point delta >= 0 AND the ECE advantage
  kept (delta below zero with the interval excluding zero).
"""

from __future__ import annotations

METRICS = ("macro_f1", "balanced_accuracy", "auc", "ece")
CONTRASTS = (("plain", "public"), ("proposed", "public"), ("proposed", "plain"))
NOT_WORSE_F1 = -0.01
MET, NOT_MET = "cumplida", "no cumplida"


def _verdict(ok: bool) -> str:
    return MET if ok else NOT_MET


def _ece_better(record: dict) -> bool:
    return record["point_delta"] < 0 and record["excludes_zero"]


def predictions(paired: dict) -> dict:
    """``paired[(arm, control)][metric]`` -> the P1-P3 verdicts with their evidence."""
    p1 = paired[("plain", "public")]["macro_f1"]
    p2 = paired[("proposed", "plain")]
    p3 = paired[("proposed", "public")]
    return {
        "P1": {
            "verdict": _verdict(p1["point_delta"] >= NOT_WORSE_F1),
            "criterion": f"macro-F1 A* - público >= {NOT_WORSE_F1}",
            "improves_public": bool(p1["point_delta"] > 0 and p1["excludes_zero"]),
            "evidence": {"macro_f1": p1},
        },
        "P2": {
            "verdict": _verdict(
                _ece_better(p2["ece"]) and p2["macro_f1"]["point_delta"] >= NOT_WORSE_F1
            ),
            "criterion": f"ECE B* - A* < 0 con IC que excluye el cero y macro-F1 >= {NOT_WORSE_F1}",
            "evidence": {"ece": p2["ece"], "macro_f1": p2["macro_f1"]},
        },
        "P3": {
            "verdict": _verdict(p3["macro_f1"]["point_delta"] >= 0 and _ece_better(p3["ece"])),
            "criterion": "macro-F1 B* - público >= 0 y ECE B* - público < 0 con IC que excluye el cero",
            "evidence": {"macro_f1": p3["macro_f1"], "ece": p3["ece"]},
        },
    }


def paired_from_rows(rows: list[dict]) -> dict:
    """Flat bootstrap rows back into ``paired[(arm, control)][metric]``."""
    out: dict = {}
    for row in rows:
        stats = {k: row[k] for k in ("point_delta", "ci95_low", "ci95_high", "excludes_zero")}
        out.setdefault((row["arm"], row["control"]), {})[row["metric"]] = stats
    return out
