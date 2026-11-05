"""Detached supervisor for the two predeclared phone-source SSL arms only."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("same-quota-without-skindisnet", "plus-skindisnet-phone")
PAIR_SHA = "98bb43179cbccf9575a4b700869380cd929138cd616bdd032483a925426f08f0"


def free_memory():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True)
    return {int(a): int(b) for a, b in (line.split(",") for line in text.strip().splitlines())}


def main(arm):
    if arm not in ARMS:
        raise ValueError("Only predeclared phone-source arms may launch")
    folder = ROOT / "config/skindisnet-phone-pair-v1"
    protocol_path = folder / "pair-protocol.json"
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != PAIR_SHA:
        raise ValueError("Pair protocol changed")
    protocol = json.loads(protocol_path.read_text())
    config_path = folder / arm / "run.json"
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != protocol["runs"][arm]["run_config_sha256"]:
        raise ValueError("Run config changed")
    config = json.loads(config_path.read_text())
    run = Path(config["base_config"]["output_dir"])
    if run.exists():
        raise FileExistsError("Refusing duplicate or implicit resume")
    smoke = ROOT / "runs" / (arm + "-seed42-smoke-v1/replay-smoke.json")
    if not smoke.exists() or json.loads(smoke.read_text())["status"] != "PASS":
        raise ValueError("Real CUDA replay smoke must pass first")
    if arm == "plus-skindisnet-phone":
        previous = ROOT / "runs/same-quota-without-skindisnet-seed42-v1/result.json"
        if not previous.exists() or json.loads(previous.read_text()).get("status") != "COMPLETE":
            raise ValueError("Matched control must complete before adding SkinDisNet")
    memory = free_memory()
    gpu = max(memory, key=memory.get)
    if memory[gpu] <= 12288:
        raise ValueError("No GPU with more than12GiB free; no job started")
    seconds = int(config["training_gpu_hours_limit"] * 3600) - 10
    assert 0 < seconds + 5 < 1.3 * 3600
    control = ROOT / "results" / (arm + "-seed42-launch-v1")
    control.mkdir(exist_ok=False)
    receipt = {"arm": arm, "protocol_sha256": PAIR_SHA, "run_config_sha256": protocol["runs"][arm]["run_config_sha256"],
               "gpu": gpu, "free_memory_at_launch_mib": memory[gpu], "maximum_child_seconds": seconds,
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "scope": "SSL image-only training; no diagnostic encoder gradients, no W&B"}
    (control / "receipt.json").write_text(json.dumps(receipt, indent=2))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
           "PYTHONPATH": str(ROOT / "scripts")}
    started = time.monotonic()
    reason = "process_exit"
    with (control / "training.log").open("xb") as log:
        child = subprocess.Popen(["/usr/bin/timeout", "--signal=TERM", "--kill-after=5s", str(seconds),
                                  sys.executable, str(ROOT / "scripts/phone_mix_ssl_runtime.py"), str(config_path)],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 env=env, start_new_session=True)
        (control / "child.json").write_text(json.dumps({"pid": child.pid, **receipt}, indent=2))
        (control / "state").write_text(f"RUNNING gpu={gpu} pid={child.pid}\n")
        while True:
            try:
                exit_code = child.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                if free_memory()[gpu] < 2048:
                    reason = "own_job_stopped_to_preserve_GPU_memory"
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        exit_code = child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        exit_code = child.wait()
                    break
    record = {"exit_code": exit_code, "reason": reason, "allocated_gpu_hours_upper_bound": (time.monotonic() - started) / 3600,
              "arm": arm, "gpu": gpu}
    (control / "execution-receipt.json").write_text(json.dumps(record, indent=2))
    (control / "exit-code").write_text(str(exit_code) + "\n")
    (control / "state").write_text(f"FINISHED exit={exit_code}\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    main(parser.parse_args().arm)
