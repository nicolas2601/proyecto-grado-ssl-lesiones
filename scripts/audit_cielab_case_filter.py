"""Audit the per-case CIELAB admissibility filter for the CIRCLe skin pool.

Metric definitions are taken verbatim from ``scripts/compare_seed42_vs_seed43.py``
so the filter operates on the same quantities as the published safety table:

    pred_a           = sens * gt_a + (1 - spec) * (1 - gt_a)
    avail_skin_frac  = 1 - pred_a
    skin_pool_contam = (1 - sens) * gt_a / max(avail_skin_frac, 1e-6)
    circle_degen     = pred_a < 0.005 or pred_a > 0.95

A case is admissible for tone estimation when it is not degenerate and its skin
pool contamination does not exceed ``tau``. ``tau`` is a pre-registered constant:
it is an input to this audit, never fitted to the data.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

STRATA = [
    ("<1%", 0.0, 0.01),
    ("1-2%", 0.01, 0.02),
    ("2-5%", 0.02, 0.05),
    ("5-10%", 0.05, 0.10),
    (">=10%", 0.10, 1.01),
]

DEGENERATE_LOW = 0.005
DEGENERATE_HIGH = 0.95


def stratum_of(gt_area: float) -> str:
    for name, low, high in STRATA:
        if low <= gt_area < high:
            return name
    raise ValueError(f"gt_area outside every stratum: {gt_area}")


def label_for(path: Path) -> str:
    """Short run label, tolerant of both the cluster and the repo layout.

    Cluster: ``paired-raw-morph-<date>/<run>/<hash>/cruda/cases.jsonl``.
    Repo:    ``dermlip_isic2018_eval_2594_<run>_<variant>_<date>_<nnn>/cases.jsonl``.
    """
    if path.parent.name == "cruda":
        return path.parent.parent.parent.name
    match = re.search(r"_eval_2594_(\d{3})_", path.parent.name)
    return match.group(1) if match else path.parent.name


def load_cases(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            gt_a = float(raw["reference_foreground_fraction"])
            sens = float(raw["sensitivity"])
            spec = float(raw["specificity"])
            dice = float(raw["dice"])
            pred_a = sens * gt_a + (1.0 - spec) * (1.0 - gt_a)
            fn_f = (1.0 - sens) * gt_a
            avail_skin = 1.0 - pred_a
            # Guard identical to the reference script, kept for bit-level agreement.
            contam = fn_f / max(avail_skin, 1e-6)
            degenerate = pred_a < DEGENERATE_LOW or pred_a > DEGENERATE_HIGH
            # The Dice invariant is asserted here because the reference script
            # reconstructs pred_a instead of reading it: a corrupt sensitivity or
            # specificity field would silently contaminate every derived metric.
            denom = gt_a + pred_a
            if denom > 1e-12:
                implied = 2.0 * (sens * gt_a) / denom
                if abs(implied - dice) > 1e-4:
                    raise AssertionError(
                        f"Dice invariant violated for {raw['image_id']}: "
                        f"implied {implied!r} vs recorded {dice!r}"
                    )
            records.append(
                {
                    "image_id": raw["image_id"],
                    "gt_area": gt_a,
                    "dice": dice,
                    "avail_skin_frac": avail_skin,
                    "unprot_gt_frac": 1.0 - sens,
                    "skin_pool_contam": contam,
                    "degenerate": degenerate,
                    "stratum": stratum_of(gt_a),
                }
            )
    return records


def summarize(items: list[dict]) -> dict:
    if not items:
        return {"count": 0}
    return {
        "count": len(items),
        "dice_mean": float(np.mean([x["dice"] for x in items])),
        # nanmedian, not median: these fields are NaN when gt_area is 0.
        "median_avail_skin_frac": float(np.nanmedian([x["avail_skin_frac"] for x in items])),
        "p05_avail_skin_frac": float(np.nanpercentile([x["avail_skin_frac"] for x in items], 5)),
        "mean_unprotected_lesion_frac_of_gt": float(
            np.mean([x["unprot_gt_frac"] for x in items])
        ),
        "p95_skin_pool_lesion_contamination": float(
            np.nanpercentile([x["skin_pool_contam"] for x in items], 95)
        ),
        "max_skin_pool_lesion_contamination": float(
            np.max([x["skin_pool_contam"] for x in items])
        ),
        "degenerate_mask_count": int(sum(1 for x in items if x["degenerate"])),
    }


def audit(path: Path, tau: float) -> dict:
    records = load_cases(path)
    kept = [x for x in records if not x["degenerate"] and x["skin_pool_contam"] <= tau]
    dropped_degenerate = [x for x in records if x["degenerate"]]
    dropped_contam = [
        x for x in records if not x["degenerate"] and x["skin_pool_contam"] > tau
    ]

    per_stratum = {}
    for name, _, _ in STRATA:
        total = [x for x in records if x["stratum"] == name]
        survivors = [x for x in kept if x["stratum"] == name]
        per_stratum[name] = {
            "total": len(total),
            "kept": len(survivors),
            "retention": (len(survivors) / len(total)) if total else float("nan"),
        }

    return {
        "tau": tau,
        "cases_total": len(records),
        "cases_kept": len(kept),
        "retention_overall": len(kept) / len(records),
        "dropped_degenerate": len(dropped_degenerate),
        "dropped_contamination": len(dropped_contam),
        "dropped_contamination_ids": sorted(x["image_id"] for x in dropped_contam),
        "dropped_degenerate_ids": sorted(x["image_id"] for x in dropped_degenerate),
        "before_filter": summarize(records),
        "after_filter": summarize(kept),
        "retention_by_stratum": per_stratum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="+", help="cruda/cases.jsonl files")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.20,
        help="Pre-registered maximum admissible skin-pool contamination per case.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = {}
    for path in args.cases:
        results[label_for(path)] = audit(path, args.tau)
    if len(results) != len(args.cases):
        raise SystemExit(
            f"labels collided: {len(args.cases)} inputs produced {len(results)} rows"
        )

    print(f"tau = {args.tau}")
    header = (
        f"{'run':<6} {'N':>5} {'keep':>6} {'reten':>7} {'drop_deg':>9} "
        f"{'drop_contam':>12} {'p95_antes':>10} {'p95_desp':>9} {'max_desp':>9}"
    )
    print(header)
    print("-" * len(header))
    for label, res in results.items():
        print(
            f"{label:<6} {res['cases_total']:>5} {res['cases_kept']:>6} "
            f"{res['retention_overall']:>6.1%} {res['dropped_degenerate']:>9} "
            f"{res['dropped_contamination']:>12} "
            f"{res['before_filter']['p95_skin_pool_lesion_contamination']:>10.4f} "
            f"{res['after_filter']['p95_skin_pool_lesion_contamination']:>9.4f} "
            f"{res['after_filter']['max_skin_pool_lesion_contamination']:>9.4f}"
        )

    print("\nRetencion por estrato de tamano de lesion")
    strata_names = [name for name, _, _ in STRATA]
    print(f"{'run':<6} " + " ".join(f"{name:>15}" for name in strata_names))
    for label, res in results.items():
        cells = []
        for name in strata_names:
            cell = res["retention_by_stratum"][name]
            cells.append(f"{cell['kept']}/{cell['total']} ({cell['retention']:.0%})")
        print(f"{label:<6} " + " ".join(f"{c:>15}" for c in cells))

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
