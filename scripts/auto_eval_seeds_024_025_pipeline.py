"""Chained evaluation for runs 024 (seed 43) and 025 (seed 44).

Waits for each training to report ``status: completed`` at 300 epochs and then
reproduces the full evaluation without further intervention:

1. PH2 val 3-group benchmark (Natural 39, Synthetic 75, Repaired backgrounds 25).
   Run 023 never got this benchmark, so it is produced here for both replicas.
2. ISIC 2018 raw inference plus paired morphological evaluation over the 2,594
   image/mask pairs.
3. Stratified metrics by lesion size and the five CIELAB safety metrics.
4. The per-case CIELAB admissibility filter at the pre-registered tau = 0.20.

Each stage is skipped if its output already exists, so the pipeline is safe to
re-run after an interruption.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

PYTHON_ENV = "/home/jovyan/work/proyectonp/.venv/bin/python"
ROOT = Path("/home/jovyan/work/dermlip_slic_v8")
RUNS = ROOT / "runs"
EVALS = ROOT / "evaluations"
AUDIT_DIR = ROOT / "audits/paired-raw-morph-20261027-001"
BENCH_DIR = EVALS / "ph2_val_3groups_benchmark_20261027"
PAIRED_DIR = EVALS / "paired-raw-morph-20261027"

TAU = 0.20
POLL_SECONDS = 60
EXPECTED_EPOCHS = 300

REPLICAS = [
    {
        "run_num": "024",
        "seed": 43,
        "run_dir": RUNS
        / "dermlip-semi-supervised-v8-isic2018-2594-abl024-resnet34-ph2cp-imapp100-s43/selection",
    },
    {
        "run_num": "025",
        "seed": 44,
        "run_dir": RUNS
        / "dermlip-semi-supervised-v8-isic2018-2594-abl025-resnet34-ph2cp-imapp100-s44/selection",
    },
]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def wait_for_run(run_dir: Path, name: str) -> dict:
    log(f"waiting for {name} at {run_dir}")
    summary_path = run_dir / "summary.json"
    while True:
        if summary_path.is_file():
            try:
                data = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # The trainer may be mid-write; retry on the next poll.
                data = {}
            if data.get("status") == "completed" and data.get("total_epochs") == EXPECTED_EPOCHS:
                log(f"{name} completed")
                return data
            if data.get("status") in {"failed", "error"}:
                raise SystemExit(f"{name} reported status {data['status']!r}; not evaluating")
        time.sleep(POLL_SECONDS)


def ph2_three_groups(replica: dict) -> None:
    run_num = replica["run_num"]
    out_path = BENCH_DIR / f"eval_{run_num}_s{replica['seed']}_3groups.json"
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
    # seed=42 here is the frozen benchmark construction seed, not the training
    # seed: the benchmark must stay identical across replicas to be comparable.
    benchmark = build_frozen_ph2_val_3groups(manifest, seed=42)
    checkpoint = replica["run_dir"] / "best_teacher_model.pt"
    result = evaluate_checkpoint_on_3groups(checkpoint, benchmark, device="cpu")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"wrote {out_path}")


def create_eval_config(replica: dict) -> Path:
    run_num = replica["run_num"]
    run_dir = replica["run_dir"]
    base = json.loads((AUDIT_DIR / "config-023.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))

    cfg = dict(base)
    cfg["corrida"] = run_num
    cfg["entrenamiento"] = str(run_dir)
    cfg["run_id"] = summary["run_id"]
    cfg["identity_fingerprint"] = identity["identity_fingerprint"]
    # Inference only, and both training GPUs are free by the time this runs.
    cfg["dispositivo"] = "cuda:0"
    cfg["salida"] = str(PAIRED_DIR / run_num)

    out_path = AUDIT_DIR / f"config-{run_num}.json"
    out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return out_path


def cases_jsonl_for(replica: dict) -> Path:
    summary = json.loads((replica["run_dir"] / "summary.json").read_text(encoding="utf-8"))
    checkpoint_sha = summary["best_teacher_checkpoint_sha256"][:16]
    return PAIRED_DIR / replica["run_num"] / checkpoint_sha / "cruda" / "cases.jsonl"


def isic_evaluation(replica: dict) -> None:
    run_num = replica["run_num"]
    if cases_jsonl_for(replica).is_file():
        log(f"ISIC evaluation for {run_num} already exists; skipping")
        return
    config_path = create_eval_config(replica)
    log(f"running evaluacion_pareada for {run_num} with {config_path}")
    subprocess.run(
        [PYTHON_ENV, str(AUDIT_DIR / "evaluacion_pareada.py"), str(config_path)],
        check=True,
    )


def cielab_filter_audit(replica: dict) -> None:
    run_num = replica["run_num"]
    cases = cases_jsonl_for(replica)
    if not cases.is_file():
        raise SystemExit(f"expected {cases} after evaluation of {run_num}")
    out_path = EVALS / f"cielab_case_filter_tau020_{run_num}.json"
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


def main() -> None:
    for replica in REPLICAS:
        name = f"{replica['run_num']} (seed {replica['seed']})"
        wait_for_run(replica["run_dir"], name)

    # Both trainings are done before any evaluation starts, so the evaluation has
    # the GPUs to itself and the two replicas are evaluated under identical load.
    for replica in REPLICAS:
        ph2_three_groups(replica)
        isic_evaluation(replica)
        cielab_filter_audit(replica)

    log("chained evaluation finished for 024 and 025")
    for replica in REPLICAS:
        log(f"  {replica['run_num']}: {cases_jsonl_for(replica)}")


if __name__ == "__main__":
    main()
