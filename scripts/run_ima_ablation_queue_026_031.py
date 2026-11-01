"""Queue runner for the IMA++ ablation lot 026-031: three waves, two runs each.

One run per GPU, two concurrent, three sequential waves. A wave only starts once
both runs of the previous wave reported ``status: completed`` with
``total_epochs: 300``. Any run reporting ``failed`` or ``error``, or exiting
without a summary, aborts the whole queue: a partial lot is worse than no lot,
because a missing seed silently turns a three-seed claim back into a one-seed one.

    wave 1  026 control cuda:0   027 IMA++ cuda:1
    wave 2  028 control cuda:1   029 IMA++ cuda:0
    wave 3  030 control cuda:0   031 IMA++ cuda:1

Device alternates across waves on purpose so that GPU is not perfectly
correlated with arm. See prepare_dermlip_ima_ablation_026_031.py.

This runner launches training. It refuses to start unless --execute is passed,
and it verifies every config hash against the launch record before launching
anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/jovyan/work/dermlip_slic_v8")
LOG_PATH = ROOT / "runs/ima_ablation_026_031_queue.log"
POLL_SECONDS = 120
# 023 took 11,085 s (3 h 5 min) for 300 epochs. Three times that is a generous
# ceiling that still catches a hung run instead of waiting forever.
WAVE_TIMEOUT_SECONDS = 3 * 11085

WAVES = [
    [
        {"run_num": "026", "arm": "control", "seed": 42, "ima_batch": 0, "device": "cuda:0"},
        {"run_num": "027", "arm": "imapp", "seed": 42, "ima_batch": 6, "device": "cuda:1"},
    ],
    [
        {"run_num": "028", "arm": "control", "seed": 43, "ima_batch": 0, "device": "cuda:1"},
        {"run_num": "029", "arm": "imapp", "seed": 43, "ima_batch": 6, "device": "cuda:0"},
    ],
    [
        {"run_num": "030", "arm": "control", "seed": 44, "ima_batch": 0, "device": "cuda:0"},
        {"run_num": "031", "arm": "imapp", "seed": 44, "ima_batch": 6, "device": "cuda:1"},
    ],
]

EXPECTED_EPOCHS = 300


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_name(run: dict) -> str:
    return (
        "dermlip-semi-supervised-v8-isic2018-2594-"
        f"abl{run['run_num']}-resnet34-ph2cp-ima{run['ima_batch']}-s{run['seed']}"
    )


def selection_dir(run: dict) -> Path:
    return ROOT / "runs" / run_name(run) / "selection"


def launcher_path(run: dict) -> Path:
    return ROOT / f"launch_train_{run['run_num']}_ima{run['ima_batch']}_s{run['seed']}.sh"


def config_path(run: dict) -> Path:
    return (
        ROOT / "runtime_poolv2_20261022_001/plan-segmentacion/metodos-v8/"
        "launch-preparation/configs"
        / f"dermlip.selection.runtime.20261101.{run['run_num']}.json"
    )


def read_summary(run: dict) -> dict | None:
    path = selection_dir(run) / "summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A half-written summary is not a failure yet; the next poll re-reads it.
        return None


def preflight(launch_record: dict[str, str] | None) -> None:
    problems: list[str] = []
    for wave in WAVES:
        for run in wave:
            cfg, launcher = config_path(run), launcher_path(run)
            if not cfg.exists():
                problems.append(f"missing config: {cfg}")
                continue
            if not launcher.exists():
                problems.append(f"missing launcher: {launcher}")
            if selection_dir(run).exists():
                problems.append(f"output already exists: {selection_dir(run)}")
            if launch_record is not None:
                expected = launch_record.get(run["run_num"])
                if expected is None:
                    problems.append(f"no config sha recorded for run {run['run_num']}")
                else:
                    actual = sha256_file(cfg)
                    if actual != expected:
                        problems.append(
                            f"config sha mismatch for {run['run_num']}\n"
                            f"      expected {expected}\n      actual   {actual}"
                        )
    if problems:
        raise SystemExit("queue preflight failed:\n  " + "\n  ".join(problems))


def launch(run: dict) -> subprocess.Popen:
    log(f"launching {run['run_num']} ({run['arm']}, ima_batch={run['ima_batch']}, "
        f"seed {run['seed']}, {run['device']})")
    return subprocess.Popen(
        ["bash", str(launcher_path(run))],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def wait_for_wave(wave: list[dict], index: int) -> None:
    pending = {run["run_num"]: run for run in wave}
    deadline = time.monotonic() + WAVE_TIMEOUT_SECONDS

    while pending:
        if time.monotonic() > deadline:
            raise SystemExit(
                f"ABORT: wave {index} exceeded {WAVE_TIMEOUT_SECONDS} s with "
                f"{sorted(pending)} still unfinished"
            )
        time.sleep(POLL_SECONDS)

        for run_num in sorted(pending):
            run = pending[run_num]
            summary = read_summary(run)
            if summary is None:
                continue

            status = summary.get("status")
            epochs = summary.get("total_epochs")

            if status in ("failed", "error"):
                raise SystemExit(
                    f"ABORT: run {run_num} reported status={status!r}. "
                    f"Queue stopped; waves after {index} not launched."
                )
            if status == "completed":
                if epochs != EXPECTED_EPOCHS:
                    raise SystemExit(
                        f"ABORT: run {run_num} completed with total_epochs={epochs!r}, "
                        f"expected {EXPECTED_EPOCHS}. A short run is not a replicate."
                    )
                log(f"  {run_num} completed, {epochs} epochs")
                del pending[run_num]
            # Any other status means still running; keep polling.

    log(f"wave {index} complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually launch. Without it the runner only runs its preflight.",
    )
    parser.add_argument(
        "--launch-record",
        type=Path,
        help=(
            "JSON mapping run number to expected config sha256, e.g. "
            '{"026": "ab12..."}. When given, a mismatch aborts before launching.'
        ),
    )
    args = parser.parse_args()

    record = None
    if args.launch_record:
        record = json.loads(args.launch_record.read_text(encoding="utf-8"))

    preflight(record)
    log("queue preflight OK: 6 configs and launchers present, no output dirs, "
        f"config hashes {'verified' if record else 'NOT verified (no launch record given)'}")

    if not args.execute:
        print("\n--execute not set: nothing launched.")
        return

    for index, wave in enumerate(WAVES, start=1):
        log(f"=== wave {index}/{len(WAVES)}")
        processes = [launch(run) for run in wave]
        for process in processes:
            # The launchers background the trainer with nohup and return at once;
            # this only reaps the wrapper, not the training itself.
            process.wait()
        wait_for_wave(wave, index)

    log("all three waves complete: 026-031 finished with 300 epochs each")


if __name__ == "__main__":
    main()
