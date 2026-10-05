"""Loaders of every paired contrast of the bench, straight from the committed JSON.

Nothing is recomputed. E3b stores its contrasts with the baseline as the arm and B71 as the
control; they are flipped here so the proposed model is always the arm.
"""

from __future__ import annotations

import json
from pathlib import Path

from dermssl.bench import isic_rows
from dermssl.bench.contrast import Contrast, flipped, from_record
from dermssl.bench.ham_rows import COMPUTES, E3_DIR
from dermssl.bench.rows import MODEL_NAMES, V71_JSON, _load

E3B_DIRS = {"canónico (batch 64)": "resultados/banco_completo/e3b_bs64/e3b_summary.json",
            "batch 32 (ya commiteado)": "resultados/banco_completo/e3b/e3b_summary.json"}
CELL_LABELS = {"B71__clean": "B71-limpio", "A71__clean": "A71-limpio", "B71__original": "B71 original",
               "A71__original": "A71 original", "PUBLIC_B71": "DINOv2 público (prot. B71)",
               "PUBLIC_A71": "DINOv2 público (prot. A71)"}
V71_LABELS = {"A71": "A71", "B71": "B71", "public_A71": "DINOv2 público (prot. A71)",
              "public_B71": "DINOv2 público (prot. B71)", "public71": "DINOv2 público (mejor prot. v7.1)",
              "A_v7": "A v7", "B_v7": "B v7", "public_v7": "DINOv2 público (v7)"}
B71_VARIANTS = {"B71": "B71 (1 vista, sin TTA)", "B71_tta": "B71 (TTA, canónico)", "B71+F4": "B71 + F4 (1 vista)",
                "B71+F4_tta": "B71 + F4 (TTA)"}
SUBSETS = {"all": "todas las lesiones", "dark_tercile": "tercil L* más oscuro"}
SEED_FILES = {42: COMPUTES[1][2], 43: f"{E3_DIR}/robustez/e3_rob_s43/e3_final_eval.json",
              44: f"{E3_DIR}/robustez/e3_rob_s44/e3_final_eval.json"}
ROBUSTNESS = f"{E3_DIR}/robustez/e3_robustness/e3_robustness.json"
UNIT_HAM = "lesion_id"


def baseline_label(arm: str) -> str:
    base = arm.removesuffix("+F4")
    return MODEL_NAMES[base] + (" + F4" if arm.endswith("+F4") else "")


def isic_contrasts(repo: Path, folder: str, ds_key: str) -> list[Contrast]:
    """Paired contrasts of one ISIC E1 folder (``contrasts/*.json``), in file order."""
    out = []
    for path in sorted((repo / folder / ds_key / "contrasts").glob("*.json")):
        if path.name.endswith(".environment.json"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(repo).as_posix()
        key, result = doc["key"], doc["result"]
        out += [from_record(r, arm=CELL_LABELS[key["arm"]], control=CELL_LABELS[key["control"]],
                            unit=result["unit"], source=rel) for r in result["paired"]]
    return out


def e3b_contrasts(repo: Path, rel: str) -> tuple[list[Contrast], list[Contrast]]:
    """(B71 variants above each frozen baseline, F4 on each baseline) of one E3b summary."""
    above, raised = [], []
    unit = _load(repo, rel)["protocol"]["bootstrap"]["grouped_by"]
    for rec in _load(repo, rel)["contrasts"]:
        arm, control = rec["arm"], rec["control"]
        if control in B71_VARIANTS:
            above.append(flipped(from_record(rec, arm=baseline_label(arm), control=B71_VARIANTS[control],
                                             unit=unit, source=rel)))
        else:
            raised.append(from_record(rec, arm=baseline_label(arm), control=baseline_label(control),
                                      unit=unit, source=rel))
    return above, raised


def f4_vs_b71(repo: Path, rel: str) -> list[Contrast]:
    """F4 - B71 contrasts of one final-eval JSON (all lesions and the darkest L* tercile)."""
    return [from_record(r, arm=f"F4 ({SUBSETS[r['subset']]})", control="B71", unit=UNIT_HAM, source=rel)
            for r in _load(repo, rel)["contrasts"]]


def v71_contrasts(repo: Path) -> list[Contrast]:
    return [from_record(r, arm=V71_LABELS[r["arm"]], control=V71_LABELS[r["control"]], unit=UNIT_HAM,
                        source=V71_JSON) for r in _load(repo, V71_JSON)["paired"]]


def isic_all(repo: Path, ds_key: str) -> dict[str, list[Contrast]]:
    return {"canónico (batch 64 + autocast)": isic_contrasts(repo, isic_rows.CANONICAL_DIR, ds_key),
            "sensibilidad (batch 16, transductivo)": isic_contrasts(repo, isic_rows.SENSITIVITY_DIR, ds_key)}
