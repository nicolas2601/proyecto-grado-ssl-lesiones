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

Secondary readings, reported beside the primary ones and never replacing
them (docs/28 5): P1-NI, strict non-inferiority (lower bound of the plain vs
public macro-F1 interval >= -0.01); P3-IC, the only reading that allows
writing "beats public" (lower bound of the proposed vs public macro-F1
interval > 0). With only the primary P3 met, the thesis says "does not lose".

The single evaluation reproduces the C1/v6 protocol: the probe is fitted on
the whole C1 train and scored on C1 val, pinned by their id fingerprints.
"""

from __future__ import annotations

METRICS = ("macro_f1", "balanced_accuracy", "auc", "ece")
CONTRASTS = (("plain", "public"), ("proposed", "public"), ("proposed", "plain"))
NOT_WORSE_F1 = -0.01
MET, NOT_MET = "cumplida", "no cumplida"
PRIMARY, SECONDARY = "primaria", "secundaria"
# fingerprint_ids of the C1 fold-0 train (8011) and val (2004), as in the bank cache keys.
C1_TRAIN_FINGERPRINT = "d347edd52d0d7648"
C1_VAL_FINGERPRINT = "6ff4c17f04a9eae2"


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
        **_primary(p1, p2, p3),
        "P1-NI": {
            "reading": SECONDARY,
            "verdict": _verdict(p1["ci95_low"] >= NOT_WORSE_F1),
            "criterion": f"límite inferior IC macro-F1 A* - público >= {NOT_WORSE_F1}",
            "evidence": {"macro_f1": p1},
        },
        "P3-IC": {
            "reading": SECONDARY,
            "verdict": _verdict(p3["macro_f1"]["ci95_low"] > 0),
            "criterion": "límite inferior IC macro-F1 B* - público > 0 (permite decir que le gana)",
            "evidence": {"macro_f1": p3["macro_f1"]},
        },
    }


def _primary(p1: dict, p2: dict, p3: dict) -> dict:
    return {
        "P1": {
            "reading": PRIMARY,
            "verdict": _verdict(p1["point_delta"] >= NOT_WORSE_F1),
            "criterion": f"macro-F1 A* - público >= {NOT_WORSE_F1}",
            "improves_public": bool(p1["point_delta"] > 0 and p1["excludes_zero"]),
            "evidence": {"macro_f1": p1},
        },
        "P2": {
            "reading": PRIMARY,
            "verdict": _verdict(
                _ece_better(p2["ece"]) and p2["macro_f1"]["point_delta"] >= NOT_WORSE_F1
            ),
            "criterion": f"ECE B* - A* < 0 con IC que excluye el cero y macro-F1 >= {NOT_WORSE_F1}",
            "evidence": {"ece": p2["ece"], "macro_f1": p2["macro_f1"]},
        },
        "P3": {
            "reading": PRIMARY,
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


def check_val_protocol(train_ids, val_ids) -> None:
    """Refuse a val evaluation whose fit or eval ids differ from C1/v6."""
    from dermssl.feature_cache import fingerprint_ids  # noqa: PLC0415

    for name, ids, want in (("train", train_ids, C1_TRAIN_FINGERPRINT), ("val", val_ids, C1_VAL_FINGERPRINT)):
        have = fingerprint_ids(list(ids))
        if have != want:
            raise ValueError(f"los ids de {name} dan {have}, C1 usa {want}: no es el protocolo de C1/v6")


def check_runs_consistent(runs: dict) -> None:
    """All finals must share one code revision and one exclusion list."""
    for key, label in (("code_revision", "code_revision"), ("exclude_ids_sha256", "exclude sha")):
        values = {role: run.get(key) for role, run in runs.items()}
        if len(set(values.values())) != 1:
            raise ValueError(f"{label} distinto entre finales: {values}")
