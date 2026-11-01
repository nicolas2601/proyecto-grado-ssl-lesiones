"""Chained evaluation for the IMA++ ablation lot 026-031.

Waits for all six trainings to report ``status: completed`` at 300 epochs, then
reproduces the full evaluation for each without further intervention:

1. PH2 val 3-group benchmark (Natural 39, Synthetic 75, Repaired backgrounds 25).
2. ISIC 2018 raw inference plus paired morphological evaluation over the 2,594
   image/mask pairs.
3. Stratified metrics by lesion size and the five CIELAB safety metrics.
4. The per-case CIELAB admissibility filter at the pre-registered tau = 0.20.
5. The paired control-vs-IMA++ delta per seed, which is the point of the lot.

Each stage is skipped if its output already exists, so the pipeline is safe to
re-run after an interruption.

Follows the structure of auto_eval_seeds_024_025_pipeline.py. Two deliberate
differences:

- All six trainings must finish before any evaluation starts, so every arm is
  evaluated under identical load. Comparing an arm evaluated alone against one
  evaluated next to a live training would put a timing artifact inside the delta.
- The paired delta is computed from the raw cases.jsonl rather than from the
  stratified summaries, and uses np.nanmedian for the area ratio. The upstream
  compare script uses np.median, which returns NaN for a whole stratum if any
  case has an empty ground-truth mask.

Read-only with respect to the runs: it writes only into evaluations/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PYTHON_ENV = "/home/jovyan/work/proyectonp/.venv/bin/python"
ROOT = Path("/home/jovyan/work/dermlip_slic_v8")
RUNS = ROOT / "runs"
EVALS = ROOT / "evaluations"
AUDIT_DIR = ROOT / "audits/paired-raw-morph-20261027-001"
BENCH_DIR = EVALS / "ph2_val_3groups_benchmark_20261027"
PAIRED_DIR = EVALS / "paired-raw-morph-20261027"
LOT_DIR = EVALS / "ima_ablation_026_031"

TAU = 0.20
POLL_SECONDS = 60
EXPECTED_EPOCHS = 300

STRATA = [("<1%", 0.0, 0.01), ("1-2%", 0.01, 0.02), ("2-5%", 0.02, 0.05),
          ("5-10%", 0.05, 0.10), (">=10%", 0.10, 1.01)]
ORDER = [name for name, _, _ in STRATA]


def _run(run_num: str, seed: int, ima_batch: int, arm: str) -> dict:
    name = (
        "dermlip-semi-supervised-v8-isic2018-2594-"
        f"abl{run_num}-resnet34-ph2cp-ima{ima_batch}-s{seed}"
    )
    return {
        "run_num": run_num,
        "seed": seed,
        "ima_batch": ima_batch,
        "arm": arm,
        "run_dir": RUNS / name / "selection",
    }


LOT = [
    _run("026", 42, 0, "control"),
    _run("027", 42, 6, "imapp"),
    _run("028", 43, 0, "control"),
    _run("029", 43, 6, "imapp"),
    _run("030", 44, 0, "control"),
    _run("031", 44, 6, "imapp"),
]
PAIRS = [(42, "026", "027"), (43, "028", "029"), (44, "030", "031")]
BY_NUM = {item["run_num"]: item for item in LOT}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def wait_for_run(item: dict) -> dict:
    name = f"{item['run_num']} ({item['arm']}, seed {item['seed']})"
    log(f"waiting for {name} at {item['run_dir']}")
    summary_path = item["run_dir"] / "summary.json"
    while True:
        if summary_path.is_file():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            if data.get("status") == "completed" and data.get("total_epochs") == EXPECTED_EPOCHS:
                log(f"{name} completed")
                return data
            if data.get("status") in {"failed", "error"}:
                raise SystemExit(f"{name} reported status {data['status']!r}; not evaluating")
        time.sleep(POLL_SECONDS)


def ph2_three_groups(item: dict) -> None:
    run_num = item["run_num"]
    out_path = BENCH_DIR / f"eval_{run_num}_ima{item['ima_batch']}_s{item['seed']}_3groups.json"
    if out_path.is_file():
        log(f"PH2 3-group benchmark for {run_num} already exists; skipping")
        return
    log(f"running PH2 val 3-group benchmark for {run_num}")
    sys.path.insert(0, "/home/jovyan/work/proyectonp/repo_v8_canonical/src")
    from prepare_and_eval_019_suite import (
        build_frozen_ph2_val_3groups,
        evaluate_checkpoint_on_3groups,
    )

    manifest = ROOT / "preflight/ph2-curator-exclusions-20261022-001/manifest.json"
    # seed=42 is the frozen benchmark construction seed, not the training seed:
    # the benchmark must stay identical across arms to be comparable.
    benchmark = build_frozen_ph2_val_3groups(manifest, seed=42)
    checkpoint = item["run_dir"] / "best_teacher_model.pt"
    result = evaluate_checkpoint_on_3groups(checkpoint, benchmark, device="cpu")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"wrote {out_path}")


def create_eval_config(item: dict) -> Path:
    run_num = item["run_num"]
    run_dir = item["run_dir"]
    base = json.loads((AUDIT_DIR / "config-023.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))

    cfg = dict(base)
    cfg["corrida"] = run_num
    cfg["entrenamiento"] = str(run_dir)
    cfg["run_id"] = summary["run_id"]
    cfg["identity_fingerprint"] = identity["identity_fingerprint"]
    cfg["dispositivo"] = "cuda:0"
    cfg["salida"] = str(PAIRED_DIR / run_num)

    out_path = AUDIT_DIR / f"config-{run_num}.json"
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out_path


def cases_jsonl_for(item: dict) -> Path:
    summary = json.loads((item["run_dir"] / "summary.json").read_text(encoding="utf-8"))
    checkpoint_sha = summary["best_teacher_checkpoint_sha256"][:16]
    return PAIRED_DIR / item["run_num"] / checkpoint_sha / "cruda" / "cases.jsonl"


def isic_evaluation(item: dict) -> None:
    run_num = item["run_num"]
    if cases_jsonl_for(item).is_file():
        log(f"ISIC evaluation for {run_num} already exists; skipping")
        return
    config_path = create_eval_config(item)
    log(f"running evaluacion_pareada for {run_num} with {config_path}")
    subprocess.run(
        [PYTHON_ENV, str(AUDIT_DIR / "evaluacion_pareada.py"), str(config_path)],
        check=True,
    )


def cielab_filter_audit(item: dict) -> None:
    run_num = item["run_num"]
    cases = cases_jsonl_for(item)
    if not cases.is_file():
        raise SystemExit(f"expected {cases} after evaluation of {run_num}")
    out_path = EVALS / f"cielab_case_filter_tau020_{run_num}.json"
    if out_path.is_file():
        log(f"CIELAB filter audit for {run_num} already exists; skipping")
        return
    log(f"auditing CIELAB per-case filter for {run_num} at tau = {TAU}")
    subprocess.run(
        [
            PYTHON_ENV,
            str(ROOT / "audit_cielab_case_filter.py"),
            str(cases),
            "--tau",
            str(TAU),
            "--json-out",
            str(out_path),
        ],
        check=True,
    )


def load_cases(path: Path) -> list[dict]:
    """Derive per-case metrics, asserting the Dice identity along the way.

    Predicted area is not stored; it is reconstructed as TP + FP from
    sensitivity and specificity. That reconstruction carries precision, area
    ratio and all five CIELAB metrics, so it is checked rather than trusted:
    for a binary mask, dice must equal 2*TP/(gt + pred).
    """
    records: list[dict] = []
    worst_identity_error = 0.0

    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        gt = float(rec["reference_foreground_fraction"])
        sens = float(rec["sensitivity"])
        spec = float(rec["specificity"])
        dice = float(rec["dice"])

        pred = sens * gt + (1.0 - spec) * (1.0 - gt)
        tp, fn = sens * gt, (1.0 - sens) * gt

        denom = gt + pred
        if denom > 1e-12:
            worst_identity_error = max(worst_identity_error, abs(2.0 * tp / denom - dice))

        avail_skin = 1.0 - pred
        records.append({
            "image_id": rec["image_id"],
            "gt_area": gt,
            "dice": dice,
            "precision": (tp / pred) if pred > 1e-12 else (1.0 if gt == 0 else 0.0),
            "area_ratio": (pred / gt) if gt > 1e-12 else float("nan"),
            "zero_overlap": dice == 0.0 and gt > 0,
            "avail_skin_frac": avail_skin,
            "unprot_gt_frac": 1.0 - sens,
            "skin_pool_contam": fn / max(avail_skin, 1e-6),
            "circle_degen": (pred < 0.005) or (pred > 0.95),
        })

    if worst_identity_error > 1e-4:
        raise SystemExit(
            f"{path}: Dice identity violated by {worst_identity_error:.3e}; "
            "the stored sensitivity/specificity do not reproduce the stored dice, "
            "so every derived metric is suspect"
        )
    return records


def summarize(items: list[dict]) -> dict:
    if not items:
        return {"count": 0}
    ratios = np.array([x["area_ratio"] for x in items], dtype=float)
    return {
        "count": len(items),
        "dice_mean": float(np.mean([x["dice"] for x in items])),
        "precision_mean": float(np.mean([x["precision"] for x in items])),
        # nanmedian, not median: an empty ground-truth mask makes area_ratio NaN
        # and np.median would poison the whole stratum silently.
        "median_area_ratio_pred_gt": (
            float(np.nanmedian(ratios)) if not np.all(np.isnan(ratios)) else float("nan")
        ),
        "zero_overlap_count": int(sum(1 for x in items if x["zero_overlap"])),
        "cielab_median_avail_skin_frac": float(np.median([x["avail_skin_frac"] for x in items])),
        "cielab_p05_avail_skin_frac": float(np.percentile([x["avail_skin_frac"] for x in items], 5)),
        "cielab_mean_unprotected_lesion_frac_of_gt": float(np.mean([x["unprot_gt_frac"] for x in items])),
        "cielab_p95_skin_pool_lesion_contamination": float(np.percentile([x["skin_pool_contam"] for x in items], 95)),
        "cielab_circle_degenerate_mask_count": int(sum(1 for x in items if x["circle_degen"])),
    }


def stratify(records: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = {name: [] for name in ORDER}
    for item in records:
        for name, low, high in STRATA:
            if low <= item["gt_area"] < high:
                buckets[name].append(item)
                break
    return {
        "overall_2594": summarize(records),
        "by_size_stratum": {name: summarize(buckets[name]) for name in ORDER},
    }


def paired_delta() -> None:
    """Control vs IMA++ per seed. This is what the lot exists to measure."""
    out_path = LOT_DIR / "ima_ablation_paired_delta.json"
    if out_path.is_file():
        log("paired delta already exists; skipping")
        return

    per_run = {num: stratify(load_cases(cases_jsonl_for(item)))
               for num, item in BY_NUM.items()}

    deltas = {}
    for seed, control_num, imapp_num in PAIRS:
        control, imapp = per_run[control_num], per_run[imapp_num]
        entry = {
            "control_run": control_num,
            "imapp_run": imapp_num,
            "overall_2594": {
                "dice_control": control["overall_2594"]["dice_mean"],
                "dice_imapp": imapp["overall_2594"]["dice_mean"],
                "delta_dice": imapp["overall_2594"]["dice_mean"] - control["overall_2594"]["dice_mean"],
            },
            "by_size_stratum": {},
        }
        for name in ORDER:
            c, i = control["by_size_stratum"][name], imapp["by_size_stratum"][name]
            entry["by_size_stratum"][name] = {
                "count": c["count"],
                "dice_control": c["dice_mean"],
                "dice_imapp": i["dice_mean"],
                "delta_dice": i["dice_mean"] - c["dice_mean"],
            }
        deltas[f"seed{seed}"] = entry

    # Sign agreement across the three seeds, per stratum. The copy-paste ablation
    # reversed sign between seeds in the two largest strata, so a delta that does
    # not agree across seeds is not a finding.
    agreement = {}
    for name in ORDER + ["overall_2594"]:
        signs = []
        for seed, _, _ in PAIRS:
            entry = deltas[f"seed{seed}"]
            value = (entry["overall_2594"]["delta_dice"] if name == "overall_2594"
                     else entry["by_size_stratum"][name]["delta_dice"])
            signs.append(value > 0)
        agreement[name] = {
            "positive_seeds": sum(signs),
            "sign_consistent_3of3": len(set(signs)) == 1,
        }

    LOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "tau_pre_registered": TAU,
                "factor": "ima_batch (0 = control, 6 = IMA++); labelled batch held at 16",
                "per_run_stratified": per_run,
                "paired_deltas": deltas,
                "sign_agreement_across_seeds": agreement,
                "caveats": [
                    "patient_independence_accredited is false for every run in this lot",
                    "the 2,594 evaluation manifest is role post_run_evaluation_only_candidate",
                    "source-cohort disjointness is accredited; patient independence is not",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"wrote {out_path}")

    print("\nPaired delta, control -> IMA++ (Dice, percentage points):")
    header = f"{'Stratum':<10} | {'N':>5} | " + " | ".join(f"s{s} delta" for s, _, _ in PAIRS) + " | sign 3/3"
    print(header)
    print("-" * len(header))
    for name in ORDER + ["overall_2594"]:
        cells, count = [], None
        for seed, _, _ in PAIRS:
            entry = deltas[f"seed{seed}"]
            if name == "overall_2594":
                cells.append(entry["overall_2594"]["delta_dice"] * 100)
                count = 2594
            else:
                cells.append(entry["by_size_stratum"][name]["delta_dice"] * 100)
                count = entry["by_size_stratum"][name]["count"]
        flag = "yes" if agreement[name]["sign_consistent_3of3"] else "NO"
        print(f"{name:<10} | {count:>5} | " + " | ".join(f"{v:+8.2f}" for v in cells) + f" |   {flag}")


def main() -> None:
    for item in LOT:
        wait_for_run(item)

    # Every arm is evaluated only after all six trainings are done, so none of
    # them shares the GPU with a live training.
    for item in LOT:
        ph2_three_groups(item)
        isic_evaluation(item)
        cielab_filter_audit(item)

    paired_delta()

    log("chained evaluation finished for 026-031")
    for item in LOT:
        log(f"  {item['run_num']} ({item['arm']}): {cases_jsonl_for(item)}")


if __name__ == "__main__":
    main()
