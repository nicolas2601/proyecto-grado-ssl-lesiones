"""Detached runner for the baseline bank on the current split (frozen reader and supervised fine-tuning).

Usage: python scripts/bank_runner_20261030.py --gpu {0,1} [--init-queues]
State lives in results/baseline-bank-20261030/gpu{N}/ (queue.json, STATUS.json, ALL_TERMINAL.json, logs/).
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from bank_models_20261030 import MODELS  # noqa: E402
from ssl_followup_runner_20261029 import all_finished, atomic_json, gpu_is_free, read_queue, update_arm  # noqa: E402

STATE = ROOT / "results/baseline-bank-20261030"
PYTHON = "/home/jovyan/work/proyectonp/.venv/bin/python"
OVERLAY = ROOT / "results/content-exclusion-overlay-v1"
TIMEOUT = {"frozen": 3600, "finetune": 4 * 3600, "stats": 900}
MODE_TEXT = {"frozen": "congelado + lector balanceado", "finetune": "fine-tuning supervisado"}
READOUTS = (("pad_adapt", "PAD", "PAD-readout.json", "/home/jovyan/work/pad_ufes_20/metadata.csv",
             "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"),
            ("ham_dev", "HAM", "HAM-readout-06aee5.json", "/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv",
             "06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5"))
PLAN = {0: [("frozen", m) for m in MODELS] + [("finetune", m) for m in ("vgg19", "resnet50", "vit-b16", "dinov2-vits14")],
        1: [("finetune", m) for m in ("alexnet", "efficientnet-b0", "simsiam-r50", "dinov2-vitb14")]}


def job_entry(mode, model):
    return {"id": f"bank-{model}-{mode}", "model": model, "mode": mode, "status": "pending", "error": None}


def init_queues():
    for gpu, jobs in PLAN.items():
        path = STATE / f"gpu{gpu}/queue.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        atomic_json(path, [job_entry(mode, model) for mode, model in jobs])
        (path.parent / "QUEUE_SEALED").touch()


def step(command, gpu, logfile, timeout):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(ROOT / "scripts"),
           "OMP_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1", "PYTHONHASHSEED": "0"}
    with open(logfile, "w") as stream:
        code = subprocess.run(command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                              timeout=timeout).returncode
    if code:
        raise RuntimeError(f"{Path(logfile).stem} exit {code}")


def evaluate_frozen(job, gpu, out, logs):
    receipt = OVERLAY / "receipt.json"
    receipt_sha = _sha(receipt)
    for role, kind, name, metadata, metadata_sha in READOUTS:
        features = out / f"{role}.npz"
        step([PYTHON, "scripts/bank_features_20261030.py", "--model", job["model"], "--role", role,
              "--output", str(features)], gpu, logs / f"{job['id']}-features-{role}.log", TIMEOUT["frozen"])
        step([PYTHON, "scripts/evaluate_overlay_readout.py", "--kind", kind, "--features", str(features),
              "--manifest", str(OVERLAY / f"{role}.jsonl"), "--metadata", metadata, "--receipt", str(receipt),
              "--output", str(out / name), "--metadata-sha", metadata_sha, "--receipt-sha", receipt_sha],
             gpu, logs / f"{job['id']}-readout-{kind}.log", TIMEOUT["frozen"])


def evaluate_finetune(job, gpu, out, logs):
    for _, kind, name, _, _ in READOUTS:
        step([PYTHON, "scripts/bank_finetune_20261030.py", "--model", job["model"], "--kind", kind,
              "--output", str(out / name)], gpu, logs / f"{job['id']}-finetune-{kind}.log", TIMEOUT["finetune"])


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry_row(job, stats, seconds):
    info = MODELS[job["model"]]
    pad, ham = stats["PAD"], stats["HAM"]
    return {"name": f"{info['label']}, {MODE_TEXT[job['mode']]}", "internal": job["id"], "section": info["section"],
            "type": "baseline en el split nuevo (sin SSL propio)", "seeds": "CV 3",
            "pad_macro_f1": pad["mean_f1"], "delta_pad": [round(pad["dmean"], 4), pad["ic"][0], pad["ic"][1]],
            "delta_pad_mean": round(pad["dseed"], 4), "ham_macro_f1": ham["mean_f1"],
            "delta_ham": [round(ham["dmean"], 4), ham["ic"][0], ham["ic"][1]],
            "delta_ham_mean": round(ham["dseed"], 4), "auc_cancer_mean": round(stats["auc_cancer"], 4),
            "recall_mean": {k: round(v, 4) for k, v in stats["recall"].items()},
            "gpu_h": round(seconds / 3600, 2), "status": "banco de baselines 2026-10-30"}


def append_registry(state, row):
    path = state / "registry.json"
    data = json.loads(path.read_text()) if path.exists() else {"rows": [], "notes": ["Baselines on the current split; reference is the SSL SimSiam seed-42 readout."]}
    data["rows"] = [r for r in data["rows"] if r["internal"] != row["internal"]] + [row]
    atomic_json(path, data)


def render(state):
    try:
        subprocess.run([PYTHON, "-c", "from render_results_table import main; import sys; main(sys.argv[1], sys.argv[2])",
                        str(state / "registry.json"), str(state / "table")], cwd=ROOT, timeout=300, check=True,
                       env={**os.environ, "PYTHONPATH": "scripts"})
    except (subprocess.SubprocessError, OSError) as error:
        print(json.dumps({"render_warning": str(error)}), flush=True)


def run_job(job, gpu, state):
    out = ROOT / f"results/{job['id']}-evaluation-v1"
    out.mkdir(parents=True, exist_ok=False)
    logs = state / "logs"
    logs.mkdir(exist_ok=True)
    started = time.monotonic()
    (evaluate_frozen if job["mode"] == "frozen" else evaluate_finetune)(job, gpu, out, logs)
    result = subprocess.run([PYTHON, "scripts/arm_stats.py", str(out)], cwd=ROOT, capture_output=True, text=True,
                            timeout=TIMEOUT["stats"], env={**os.environ, "PYTHONPATH": "scripts"})
    if result.returncode:
        raise RuntimeError("stats failed: " + result.stderr[-300:])
    (out / "stats.json").write_text(result.stdout)
    append_registry(state, registry_row(job, json.loads(result.stdout), time.monotonic() - started))
    render(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1))
    parser.add_argument("--init-queues", action="store_true")
    args = parser.parse_args()
    if args.init_queues:
        init_queues()
        return
    state = STATE / f"gpu{args.gpu}"
    queue = state / "queue.json"
    with (state / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            rows = read_queue(queue)
            atomic_json(state / "STATUS.json", {"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "jobs": rows})
            pending = [job for job in rows if job["status"] == "pending"]
            if not pending:
                if (state / "QUEUE_SEALED").exists() and all_finished(rows):
                    atomic_json(state / "ALL_TERMINAL.json", {"jobs": rows})
                    return
                time.sleep(30)
                continue
            if (state / "PAUSE").exists() or not gpu_is_free(args.gpu):
                time.sleep(30)
                continue
            job = pending[0]
            update_arm(queue, job["id"], "running")
            try:
                run_job(job, args.gpu, state)
                update_arm(queue, job["id"], "done")
            except Exception as error:
                update_arm(queue, job["id"], "FAILED", f"{type(error).__name__}: {error}")
                print(json.dumps({"job": job["id"], "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
