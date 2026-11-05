"""Detached, non-overwriting SSL follow-up runner with architecture-specific export."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

from followup_dispatch_20261029 import rewrite_command

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results/ssl-followup-20261029"
BASE_PYTHON = "/home/jovyan/work/proyectonp/.venv/bin/python"
GUIDED_SECTION = "SSL con muestreo guiado por etiquetas"


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def all_finished(rows):
    return bool(rows) and all(arm.get("status") in ("done", "FAILED") for arm in rows)


def read_queue(path):
    return json.loads(path.read_text())


def update_arm(path, arm_id, status, error=None):
    rows = read_queue(path)
    row = next(row for row in rows if row["id"] == arm_id)
    row.update(status=status, error=error)
    atomic_json(path, rows)


def validate_arm(arm):
    if arm["seed"] not in (42, 43, 44):
        raise ValueError("Seed outside the authorized guard")
    uses_labels = arm.get("encoder_uses_diagnostic_labels")
    guided = arm.get("label_guided_sampling") is True
    if uses_labels is not guided:
        raise ValueError("Explicit diagnostic-label declaration required and it must match label_guided_sampling")
    if guided and arm.get("section") != GUIDED_SECTION:
        raise ValueError("Label-guided arms must be reported in the label-guided section")
    for name in (arm["script"], arm.get("exporter", arm["script"])):
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Missing or out-of-project script: {name}")
    if arm.get("skip_train"):
        return
    for kind, config_name in (("smoke", "smoke.json"), ("replay", "replay.json"), ("full", "run.json")):
        config = json.loads((ROOT / arm["config_dir"] / config_name).read_text())
        output = Path(config["base_config"]["output_dir"])
        if output.resolve() != (ROOT / arm[f"{kind}_out"]).resolve():
            raise ValueError(f"Output mismatch for {kind}")
        if output.exists():
            raise ValueError(f"Refusing to overwrite {kind} output: {output}")
        if config["base_config"]["seed"] != arm["seed"]:
            raise ValueError("Config seed mismatch")


def source_digests(arm):
    names = {arm["script"], *arm.get("source_files", [])}
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(names)}


def load_driver(state, arm):
    spec = importlib.util.spec_from_file_location("historical_driver", ROOT / "scripts/ssl_runner_v2.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.ROOT, driver.LOGS, driver.REG = ROOT, state / "logs", state / "registry.json"
    driver.LOGS.mkdir(exist_ok=True)
    if not driver.REG.exists():
        atomic_json(driver.REG, {"rows": [], "notes": ["Exploratory SSL follow-up; historical bootstrap reference is SimSiam seed 42."]})
    original_run = driver.run
    expected = source_digests(arm)

    def routed_run(command, env, timeout, logfile):
        atomic_json(state / "CURRENT_STAGE.json", {"arm": arm["id"], "stage": Path(logfile).stem,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log": str(logfile)})
        if source_digests(arm) != expected:
            raise RuntimeError("Source changed during the experiment")
        code = original_run(rewrite_command(command, arm), env, timeout, logfile)
        if source_digests(arm) != expected:
            raise RuntimeError("Source changed during the experiment")
        return code

    def render():
        command = [BASE_PYTHON, "-c", "from render_results_table import main; import sys; main(sys.argv[1], sys.argv[2])",
                   str(driver.REG), str(state / "table")]
        subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": "scripts"}, check=True, timeout=300)

    driver.run, driver.render_table = routed_run, render
    atomic_json(state / f"{arm['id']}-source-digests.json", expected)
    return driver


def gpu_is_free(gpu):
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=20)
        return any(int(index) == gpu and int(memory) < 1000
                   for index, memory in (line.split(",") for line in result.stdout.splitlines()))
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        print(json.dumps({"gpu_probe_error": str(error), "action": "wait_without_allocating"}), flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    state = STATE / f"gpu{args.gpu}"
    state.mkdir(parents=True, exist_ok=True)
    queue = state / "queue.json"
    if args.dry_run:
        for arm in read_queue(queue):
            if arm["status"] == "pending":
                validate_arm(arm)
                print(json.dumps({"validated": arm["id"], "script": arm["script"], "gpu": args.gpu}))
        return
    with (state / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.chdir(ROOT)
        while True:
            rows = read_queue(queue)
            atomic_json(state / "STATUS.json", {"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "arms": rows})
            pending = [arm for arm in rows if arm["status"] == "pending"]
            if not pending:
                if (state / "QUEUE_SEALED").exists() and all_finished(rows):
                    atomic_json(state / "ALL_TERMINAL.json", {"arms": rows})
                    return
                time.sleep(30)
                continue
            if (state / "PAUSE").exists() or not gpu_is_free(args.gpu):
                time.sleep(30)
                continue
            arm = pending[0]
            try:
                validate_arm(arm)
                driver = load_driver(state, arm)
                if not driver.mem_ok():
                    time.sleep(30)
                    continue
                update_arm(queue, arm["id"], "running")
                atomic_json(state / "STATUS.json", {"updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "arms": read_queue(queue)})
                environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": str(ROOT / "scripts"),
                               "OMP_NUM_THREADS": "2", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
                driver.arm_steps(arm, environment)
                update_arm(queue, arm["id"], "done")
            except Exception as error:
                update_arm(queue, arm["id"], "FAILED", f"{type(error).__name__}: {error}")
                print(json.dumps({"arm": arm["id"], "error": str(error)}), flush=True)


if __name__ == "__main__":
    main()
