"""Serial, ledger-backed coordinator for reviewed V8 stage plans.

This script never creates gates or chooses pseudo-labels. A plan lists each stage/cycle,
its exact manifest, reviewed gate, config, and output directory. Execution is opt-in.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_segmentation_v8.py"


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def inventory(path):
    path = Path(path)
    if not path.is_dir():
        return {}
    return {str(p.relative_to(path)): file_sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def sample_resources(out, phase, pid=None):
    import datetime
    sample = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "phase": phase}
    if pid:
        try:
            fields = Path(f"/proc/{pid}/status").read_text().splitlines()
            sample["process_rss_mb"] = round(int(next(x.split()[1] for x in fields if x.startswith("VmRSS:"))) / 1024, 2)
        except (OSError, StopIteration, ValueError):
            sample["process_rss_mb"] = None
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, val = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                mem[key] = int(val.split()[0]) * 1024
        sample["host_ram_total_bytes"] = mem.get("MemTotal")
        sample["host_ram_available_bytes"] = mem.get("MemAvailable")
    except OSError:
        pass
    try:
        usage = shutil.disk_usage(out)
        sample["output_filesystem_total_bytes"] = usage.total
        sample["output_filesystem_free_bytes"] = usage.free
    except OSError:
        pass
    cgroup = Path("/sys/fs/cgroup")
    try:
        for key in ("memory.max", "memory.current", "cpu.max", "pids.max"):
            value = (cgroup / key).read_text().strip()
            sample["cgroup_" + key.replace(".", "_")] = value
    except OSError:
        sample["cgroup_v2"] = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        sample["gpus"] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        sample["gpus"] = None
    with (Path(out) / "resources.jsonl").open("a") as f:
        f.write(json.dumps(sample) + "\n")


def validate_plan(plan, path):
    if plan.get("schema_version") != 1 or plan.get("protocol") != "w3-four-methods-v8":
        raise ValueError("Unsupported controller plan schema/protocol")
    if plan.get("method") not in {"sled", "usl_net", "rpi_net", "tridermnet_hago"}:
        raise ValueError("Unknown method")
    steps = plan.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("A nonempty, explicitly ordered step list is required")
    seen = set()
    for step in steps:
        for key in ("id", "stage", "manifest", "gate", "out", "config"):
            if not step.get(key):
                raise ValueError(f"Each step requires {key}")
        if step["id"] in seen:
            raise ValueError("Step IDs must be unique")
        seen.add(step["id"])
        if step["stage"] not in {"pretrain", "ccam", "generate", "fit", "predict", "refine"}:
            raise ValueError("Controller plan only accepts concrete pipeline stages")
        if not Path(step["config"]).is_file():
            raise ValueError(f"Missing config for step {step['id']}: {step['config']}")
        if step["stage"] == "pretrain" and not step.get("ssl_origin"):
            raise ValueError(f"Pretrain step {step['id']} requires ssl_origin")
        if step["stage"] == "refine" and not step.get("predictions"):
            raise ValueError(f"Refine step {step['id']} requires its completed predictions directory")
        if step["stage"] != "refine" and step.get("predictions"):
            raise ValueError("predictions is only valid for refine steps")
    if not plan.get("wandb_project"):
        raise ValueError("W&B project must be explicit")
    return steps


def identity_for(plan_path, plan, step):
    return {
        "plan_sha256": file_sha(plan_path),
        "method": plan["method"],
        "step": step,
        "manifest_sha256": file_sha(step["manifest"]),
        "gate_sha256": file_sha(step["gate"]),
        "config_sha256": file_sha(step["config"]),
    }


def command_for(plan, step, resume=False):
    cmd = [sys.executable, str(LAUNCHER), "--_worker", "--method", plan["method"],
           "--stage", step["stage"], "--config", step["config"], "--manifest", step["manifest"],
           "--gate", step["gate"], "--out", step["out"], "--wandb-project", plan["wandb_project"]]
    if step.get("predictions"):
        cmd += ["--predictions", step["predictions"]]
    if step.get("ssl_origin"):
        cmd += ["--ssl-origin", step["ssl_origin"]]
    if resume:
        cmd.append("--resume")
    return cmd


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="Required to launch any pipeline stage")
    p.add_argument("--resume", action="store_true", help="Resume the exact plan recorded in ledger.json")
    return p.parse_args()


def main():
    args = arguments()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    steps = validate_plan(plan, plan_path)
    if not plan.get("run_root"):
        raise ValueError("Plan must specify a dedicated run_root")
    run_root = Path(plan["run_root"]).resolve()
    if run_root == Path("/"):
        raise ValueError("Plan must specify a dedicated run_root")
    if not args.execute:
        print(json.dumps({"method": plan["method"], "step_order": [s["id"] for s in steps],
                          "execution_started": False, "run_root": str(run_root)}, indent=2))
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    ledger_path = run_root / "ledger.json"
    plan_hash = file_sha(plan_path)
    if ledger_path.exists():
        if not args.resume:
            raise ValueError("ledger.json already exists; use --resume with the unchanged plan")
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("plan_sha256") != plan_hash:
            raise ValueError("Plan changed since the previous controller session")
    else:
        if args.resume:
            raise ValueError("No ledger to resume")
        ledger = {"protocol": plan["protocol"], "method": plan["method"], "plan_sha256": plan_hash,
                  "status": "prepared", "steps": {}}
        atomic_json(ledger_path, ledger)

    lock_path = run_root / "controller.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Controller lock exists; inspect its owner before recovery") from error
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    try:
        for step in steps:
            sid = step["id"]
            record = ledger["steps"].get(sid)
            out = Path(step["out"])
            missing = [key for key in ("manifest", "gate") if not Path(step[key]).is_file()]
            if step["stage"] == "refine" and step.get("predictions") and not (Path(step["predictions"]) / "summary.json").is_file():
                missing.append("predictions/summary.json")
            if missing:
                ledger["status"] = "awaiting_review"
                ledger["awaiting_step"] = sid
                ledger["missing_inputs"] = {key: step.get(key) for key in missing}
                atomic_json(ledger_path, ledger)
                print(json.dumps({"status": "awaiting_review", "step": sid,
                                  "missing_inputs": ledger["missing_inputs"]}, indent=2))
                return 3
            identity = identity_for(plan_path, plan, step)
            if record and record.get("status") == "completed":
                if record.get("identity") != identity or record.get("output_inventory") != inventory(out):
                    raise ValueError(f"Completed step {sid} changed on disk; refusing to reuse it")
                continue
            recovering_fit = bool(record and record.get("status") in {"running", "interrupted", "failed"}
                                  and step["stage"] == "fit" and out.is_dir() and (out / "resume.pt").is_file())
            if out.exists() and not recovering_fit:
                raise ValueError(f"Unrecorded/partial output exists for {sid}; preserve and review it before a new output")
            if not record and out.exists():
                raise ValueError(f"Output already exists for new step {sid}")
            ledger["status"] = "running"
            ledger["steps"][sid] = {"status": "running", "identity": identity, "started_utc": time.time()}
            atomic_json(ledger_path, ledger)
            out.parent.mkdir(parents=True, exist_ok=True)
            sample_resources(out.parent, "before")
            cmd = command_for(plan, step, resume=recovering_fit)
            if recovering_fit:
                cmd[cmd.index("--_worker")] = "--_worker"
            config = json.loads(Path(step["config"]).read_text())
            limits = config.get("resources", {})
            timeout = limits.get("timeout_seconds")
            max_rss = limits.get("max_rss_mb")
            grace = limits.get("shutdown_grace_seconds", 120)
            interval = float(plan.get("resource_sample_seconds", 10))
            if any(not isinstance(x, (int, float)) or x <= 0 for x in (timeout, max_rss, grace, interval)):
                raise ValueError("Positive timeout, RSS, shutdown grace, and sample interval required")
            process = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)
            started = time.monotonic()
            try:
                while process.poll() is None:
                    sample_resources(out.parent, "running", process.pid)
                    try:
                        status = Path(f"/proc/{process.pid}/status").read_text().splitlines()
                        rss = int(next(x.split()[1] for x in status if x.startswith("VmRSS:"))) / 1024
                    except (OSError, StopIteration, ValueError):
                        rss = 0
                    expired = time.monotonic() - started > timeout
                    over_rss = rss > max_rss
                    if expired or over_rss:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=grace)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        raise RuntimeError("Registered wall-time/RSS limit exceeded; output evidence preserved")
                    time.sleep(interval)
            except KeyboardInterrupt:
                ledger["steps"][sid]["status"] = "interrupted"
                ledger["steps"][sid]["interrupted_utc"] = time.time()
                ledger["status"] = "interrupted"
                atomic_json(ledger_path, ledger)
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
                raise
            if process.returncode != 0:
                ledger["steps"][sid]["status"] = "failed"
                ledger["steps"][sid]["returncode"] = process.returncode
                ledger["steps"][sid]["finished_utc"] = time.time()
                ledger["status"] = "failed"
                atomic_json(ledger_path, ledger)
                raise RuntimeError(f"Pipeline stage {sid} failed with exit code {process.returncode}")
            ledger["steps"][sid].update(status="completed", finished_utc=time.time(),
                                        output_inventory=inventory(out))
            atomic_json(ledger_path, ledger)
            sample_resources(out.parent, "after")
        ledger["status"] = "completed"
        ledger["finished_utc"] = time.time()
        atomic_json(ledger_path, ledger)
        return 0
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
