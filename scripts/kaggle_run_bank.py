"""Run a whole Kaggle bank end to end, respecting the two-GPU-session limit.

Kaggle only accepts two concurrent batch GPU sessions per account, and
``kaggle kernels push`` exits 0 even when it refuses the push, so a naive loop
over the eight models silently runs two of them. This driver keeps at most
``--max-concurrent`` kernels in flight, polls their status, downloads each
result as soon as it finishes, and records progress in a state file so it can
be interrupted and resumed.

    python scripts/kaggle_run_bank.py --bank baselines_isic2020 --user nicorghost --real
    python scripts/kaggle_run_bank.py --bank baselines_isic2020 --user nicorghost --real --report

A downloaded result is only accepted when its ``smoke`` flag matches the run
mode: Kaggle reports the *kernel*'s status, so a stale COMPLETE left by an
earlier smoke version would otherwise pass as a real run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kaggle_push import BANKS, REPO, kernel_slug, stage  # noqa: E402

# A bank run lasts hours and is always piped to a log. Python block-buffers
# stdout when it is not a tty, so without this the log stays EMPTY until the
# process exits - which is precisely when the progress no longer matters. Every
# false "failed" diagnosed on 2026-09-11 had to be reconstructed from files on
# disk because the reason never reached the log.
sys.stdout.reconfigure(line_buffering=True)

GPU_SESSION_LIMIT = 2
STATUS_RE = re.compile(r"KernelWorkerStatus\.(\w+)")
BUSY_MARKER = "Maximum batch GPU session count"
TERMINAL = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


class StaleResult(RuntimeError):
    """The downloaded JSON does not belong to the run we just launched."""


def parse_push(stdout: str, stderr: str) -> tuple[bool, str]:
    text = f"{stdout}\n{stderr}".strip()
    for line in text.splitlines():
        if "Kernel push error" in line:
            return False, line.strip()
    for line in text.splitlines():
        if "successfully pushed" in line:
            return True, line.strip()
    return False, text or "no output from kaggle kernels push"


def is_slot_busy(text: str) -> bool:
    return BUSY_MARKER in text


def parse_status(text: str) -> str:
    match = STATUS_RE.search(text)
    return match.group(1) if match else "UNKNOWN"


def output_dir(bank: str, results_root: Path) -> Path:
    """Where a bank's JSONs land.

    Fine-tuning banks mirror the cluster layout, `<root>/finetune/<dataset>/`,
    because `scripts/finetune_baseline.py` writes there and the whole point of
    running both is comparing them without translating paths. Frozen banks have
    no cluster counterpart and keep their flat `<root>/<bank>/`.
    """
    if bank.startswith("finetune"):
        sys.path.insert(0, str(REPO / "src"))
        from dermssl.finetune import protocol
        return results_root / "finetune" / protocol.DATASETS[BANKS[bank][0]].name
    return results_root / bank


def result_name(bank: str, model: str) -> str:
    """Derived from the protocol, never from the bank name.

    Matching on `endswith("isic2020")` silently returned the HAM10000 file name
    for every other ISIC bank, so the driver would download one bank and verify
    another one's results.
    """
    sys.path.insert(0, str(REPO / "src"))
    from dermssl.finetune import protocol
    suffix = protocol.DATASETS[BANKS[bank][0]].json_suffix
    # The fine-tuning notebooks write results_<model>_ft<suffix>.json. Deriving
    # only from json_suffix produced results_<model>.json for ham10000 - a name
    # nothing ever writes - and the driver retired four finished runs as
    # "missing result file" before its log became readable enough to say so.
    infix = "_ft" if bank.startswith("finetune") else ""
    return f"results_{model}{infix}{suffix}.json"


def verify_result(path: Path, real: bool) -> dict:
    if not path.is_file():
        raise StaleResult(f"missing result file {path}")
    data = json.loads(path.read_text())
    if bool(data.get("smoke")) is real:
        raise StaleResult(
            f"{path.name} has smoke={data.get('smoke')} for a "
            f"{'real' if real else 'smoke'} run - stale kernel output")
    return data


def load_state(path: Path) -> dict[str, str]:
    return json.loads(path.read_text()) if path.is_file() else {}


def save_state(path: Path, state: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def pending(models: list[str], state: dict[str, str]) -> list[str]:
    return [m for m in models if m not in state]


def free_slots(state: dict[str, str], limit: int = GPU_SESSION_LIMIT) -> int:
    return limit - sum(1 for v in state.values() if v == "running")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def push(bank: str, model: str, user: str, real: bool, staging: Path) -> tuple[bool, str]:
    folder = stage(bank, model, user, real, staging)
    out = _run(["kaggle", "kernels", "push", "-p", str(folder)])
    return parse_push(out.stdout, out.stderr)


def status(slug: str) -> str:
    out = _run(["kaggle", "kernels", "status", slug])
    return parse_status(out.stdout + out.stderr)


# How many unreadable polls in a row before the driver says so. One is a blip;
# a run of them means the CLI cannot see the kernel at all.
UNKNOWN_TOLERANCE = 5

_unknown_streak: dict[str, int] = {}


def status_or_report(slug: str, reader=status, consecutive: int = UNKNOWN_TOLERANCE,
                     report=print) -> str:
    """Poll a kernel, turning a persistent unreadable status into a report.

    UNKNOWN is what `parse_status` returns when the CLI prints nothing
    parseable - a dropped network, an expired token. It is not terminal, so the
    driver treated it as "still running" and polled a kernel it could no longer
    see, forever, printing nothing. The network did drop during this session.
    """
    phase = reader(slug)
    if phase != "UNKNOWN":
        _unknown_streak.pop(slug, None)
        return phase
    streak = _unknown_streak.get(slug, 0) + 1
    _unknown_streak[slug] = streak
    if streak >= consecutive:
        report(f"  [{slug}] estado ilegible {streak} veces seguidas: "
               "el CLI no puede ver el kernel (red o token). Revisar a mano.")
        return "UNREACHABLE"
    return "UNKNOWN"


def download_with_retry(slug: str, dest: Path, runner=_run, attempts: int = 3,
                        delay: float = 20.0) -> bool:
    """Fetch a kernel's output, retrying a transient miss.

    Kaggle reports COMPLETE slightly before the output is fetchable, and the CLI
    exits non-zero without saying much. Ignoring that exit code cost a real GPU
    run: dinov2_vitb14 on the ISIC 2019 bank was written off as "missing result
    file" while its JSON downloaded fine seconds later.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        out = runner(["kaggle", "kernels", "output", slug, "-p", str(dest)])
        if out.returncode == 0:
            return True
        if attempt < attempts:
            print(f"  [{slug}] descarga fallo (intento {attempt}/{attempts}), reintento")
            time.sleep(delay)
    return False


def verify_with_refetch(path: Path, real: bool, slug: str, dest: Path,
                        downloader=download_with_retry, attempts: int = 3,
                        delay: float = 30.0) -> bool:
    """Verify the result, re-fetching when it looks like the previous version.

    Kaggle can keep serving the old version's output for a while after a new
    one completes. The first download of resnet50's real fine-tuning run
    returned the smoke JSON, and a single check retired a finished GPU run.
    """
    for attempt in range(1, attempts + 1):
        try:
            verify_result(path, real=real)
            return True
        except StaleResult as exc:
            if attempt == attempts:
                print(f"  [{path.name}] {exc}")
                return False
            print(f"  [{path.name}] resultado viejo (intento {attempt}/{attempts}), "
                  "re-descargando")
            time.sleep(delay)
            downloader(slug, dest)
    return False


def collect(bank: str, model: str, user: str, real: bool, out_dir: Path) -> str:
    """Download a finished kernel and keep only a result matching the run mode."""
    slug = f"{user}/{kernel_slug(bank, model)}"
    if not download_with_retry(slug, out_dir):
        print(f"  [{model}] no se pudo descargar el output de {slug}")
        return "failed"
    path = out_dir / result_name(bank, model)
    if not verify_with_refetch(path, real=real, slug=slug, dest=out_dir):
        return "failed"
    return "done"


def drive(bank: str, models: list[str], user: str, real: bool, out_dir: Path,
          state_file: Path, poll: int, limit: int, staging: Path) -> dict[str, str]:
    state = load_state(state_file)
    while pending(models, state) or any(v == "running" for v in state.values()):
        for model in list(state):
            if state[model] != "running":
                continue
            phase = status_or_report(f"{user}/{kernel_slug(bank, model)}")
            if phase not in TERMINAL:
                continue
            state[model] = collect(bank, model, user, real, out_dir) \
                if phase == "COMPLETE" else "failed"
            print(f"  [{model}] {phase} -> {state[model]}")
            save_state(state_file, state)
        for model in pending(models, state):
            if free_slots(state, limit) <= 0:
                break
            ok, detail = push(bank, model, user, real, staging)
            if ok:
                state[model] = "running"
                print(f"  [{model}] pushed")
                save_state(state_file, state)
            elif not is_slot_busy(detail):
                state[model] = "failed"
                print(f"  [{model}] push failed: {detail}")
                save_state(state_file, state)
            else:
                # Without this the log goes silent for as long as the other
                # sessions last, and a driver waiting correctly is
                # indistinguishable from one that hung.
                print(f"  [{model}] sin slot GPU libre ({GPU_SESSION_LIMIT} en uso), "
                      "esperando")
                break
        if pending(models, state) or any(v == "running" for v in state.values()):
            # heartbeat: a bank whose models take an hour each printed nothing
            # for that hour, so a healthy run looked exactly like a hung one.
            corriendo = [m for m, v in state.items() if v == "running"]
            print(f"  [{time.strftime('%H:%M')}] corriendo={corriendo or '-'} "
                  f"hechos={sum(1 for v in state.values() if v == 'done')}/{len(models)} "
                  f"| siguiente poll en {poll}s")
            time.sleep(poll)
    return state


def report(bank: str, models: list[str], user: str, state_file: Path) -> None:
    state = load_state(state_file)
    for model in models:
        slug = f"{user}/{kernel_slug(bank, model)}"
        print(f"{model:18} state={state.get(model, 'pending'):8} kaggle={status(slug)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", required=True, choices=tuple(BANKS))
    p.add_argument("--user", required=True)
    p.add_argument("--models", nargs="*")
    p.add_argument("--real", action="store_true")
    p.add_argument("--report", action="store_true", help="print state and exit")
    p.add_argument("--poll", type=int, default=120, help="seconds between status polls")
    p.add_argument("--max-concurrent", type=int, default=GPU_SESSION_LIMIT)
    p.add_argument("--staging", default="/tmp/kaggle-staging")
    args = p.parse_args(argv)

    sys.path.insert(0, str(REPO / "src"))
    from dermssl.finetune import protocol

    models = args.models or list(protocol.MODELS)
    out_dir = output_dir(args.bank, REPO / "results")
    state_file = out_dir / ".run_state.json"

    if args.report:
        report(args.bank, models, args.user, state_file)
        return

    print(f"bank={args.bank} models={len(models)} mode={'real' if args.real else 'smoke'}")
    state = drive(args.bank, models, args.user, args.real, out_dir, state_file,
                  args.poll, args.max_concurrent, Path(args.staging))
    failed = [m for m, v in state.items() if v != "done"]
    print(f"\ndone: {len(models) - len(failed)}/{len(models)}")
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
