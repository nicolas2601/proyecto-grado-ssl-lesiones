"""Ejecuta un banco completo de Kaggle de punta a punta, respetando el límite de dos sesiones GPU.

Kaggle solo acepta dos sesiones GPU por lotes simultáneas por cuenta y
``kaggle kernels push`` termina con 0 incluso cuando rechaza la publicación.
Este controlador mantiene como máximo ``--max-concurrent`` kernels en curso,
consulta su estado, descarga cada resultado al finalizar y registra el
progreso en un archivo de estado para poder interrumpirlo y reanudarlo.

    python scripts/kaggle_run_bank.py --bank baselines_isic2020 --user nicorghost --real
    python scripts/kaggle_run_bank.py --bank baselines_isic2020 --user nicorghost --real --report

Un resultado descargado solo se acepta si su indicador ``smoke`` coincide con
el modo de ejecución: Kaggle informa el estado del *kernel*, por lo que un
COMPLETE obsoleto de una versión smoke podría pasar por una ejecución real.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kaggle_push import BANKS, REPO, kernel_slug, stage  # noqa: E402
from dermssl.cli import build_spanish_parser  # noqa: E402

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
STATE_LABELS = {
    "pending": "pendiente",
    "running": "en curso",
    "done": "completado",
    "failed": "fallido",
    "COMPLETE": "completado",
    "ERROR": "error",
    "CANCEL_ACKNOWLEDGED": "cancelado",
    "UNKNOWN": "desconocido",
}


class StaleResult(RuntimeError):
    """El JSON descargado no pertenece a la ejecución recién iniciada."""


def parse_push(stdout: str, stderr: str) -> tuple[bool, str]:
    text = f"{stdout}\n{stderr}".strip()
    for line in text.splitlines():
        if "Kernel push error" in line:
            return False, line.strip()
    for line in text.splitlines():
        if "successfully pushed" in line:
            return True, line.strip()
    return False, text or "sin salida de kaggle kernels push"


def is_slot_busy(text: str) -> bool:
    return BUSY_MARKER in text


def parse_status(text: str) -> str:
    match = STATUS_RE.search(text)
    return match.group(1) if match else "UNKNOWN"


def display_state(value: str) -> str:
    return STATE_LABELS.get(value, value)


def output_dir(bank: str, results_root: Path) -> Path:
    """Where a bank's JSONs land.

    Fine-tuning banks mirror the cluster layout, `<root>/finetune/<dataset>/`,
    because `scripts/finetune_baseline.py` writes there and the whole point of
    running both is comparing them without translating paths. Frozen banks have
    no cluster counterpart and keep their flat `<root>/<bank>/`.
    """
    # The LR probe shares ham10000 with the main bank but is NOT part of it, so
    # it gets its own directory. Without this it resolved to the published
    # bank's folder, read that bank's state file and exited claiming 8/8.
    if bank == "finetune_lrprobe":
        return results_root / bank
    if bank.startswith("finetune"):
        sys.path.insert(0, str(REPO / "src"))
        from dermssl.finetune import protocol
        return results_root / "finetune" / protocol.DATASETS[BANKS[bank][0]].name
    return results_root / bank


def result_name(bank: str, model: str) -> str:
    """Se deriva del protocolo, nunca del nombre del banco.

    Usar `endswith("isic2020")` devolvía silenciosamente el nombre de archivo
    de HAM10000 para los otros bancos ISIC: se descargaba un banco y se
    verificaban los resultados de otro.
    """
    sys.path.insert(0, str(REPO / "src"))
    from dermssl.finetune import protocol
    suffix = protocol.DATASETS[BANKS[bank][0]].json_suffix
    # The fine-tuning notebooks write results_<model>_ft<suffix>.json. Deriving
    # only from json_suffix produced results_<model>.json for ham10000 - a name
    # nothing ever writes - and the driver retired four finished runs as
    # "missing result file" before its log became readable enough to say so.
    infix = "_ft" if bank.startswith("finetune") else ""
    # The LR probe shares ham10000 with the main finetune bank, so without its
    # own suffix it would overwrite published rows with diagnostic numbers.
    if bank == "finetune_lrprobe":
        infix, suffix = "_ft_lrprobe", ""
    return f"results_{model}{infix}{suffix}.json"


def verify_result(path: Path, *, real: bool, expected_run_token: str) -> dict:
    if not path.is_file():
        raise StaleResult(f"falta el archivo de resultados {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise StaleResult(f"{path.name} no contiene JSON valido") from exc
    if not isinstance(data, dict):
        raise StaleResult(f"{path.name} debe contener un objeto JSON")
    if data.get("smoke") is not (not real):
        raise StaleResult(
            f"{path.name} tiene smoke={data.get('smoke')} para una ejecución "
            f"{'real' if real else 'smoke'}; la salida del kernel está desactualizada")
    if data.get("kaggle_run_token") != expected_run_token:
        raise StaleResult(f"{path.name} tiene un token de ejecucion distinto")
    return data


def _state_entry(value: str | dict) -> dict[str, str]:
    if isinstance(value, str):
        return {"status": value, "run_token": ""}
    return {"status": str(value.get("status", "pending")),
            "run_token": str(value.get("run_token", ""))}


def _status(value: str | dict) -> str:
    return _state_entry(value)["status"]


def load_state(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    return {model: _state_entry(value) for model, value in json.loads(path.read_text()).items()}


def save_state(path: Path, state: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def pending(models: list[str], state: dict[str, str | dict]) -> list[str]:
    return [m for m in models if m not in state or _status(state[m]) == "pending"]


def free_slots(state: dict[str, str | dict], limit: int = GPU_SESSION_LIMIT) -> int:
    return limit - sum(
        1 for value in state.values() if _status(value) in {"pushing", "running"}
    )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def push(bank: str, model: str, user: str, real: bool, staging: Path,
         run_token: str) -> tuple[bool, str]:
    folder = stage(bank, model, user, real, staging, run_token=run_token)
    out = _run(["kaggle", "kernels", "push", "-p", str(folder)])
    return parse_push(out.stdout, out.stderr)


def status(slug: str) -> str:
    out = _run(["kaggle", "kernels", "status", slug])
    return parse_status(out.stdout + out.stderr)


# How many unreadable polls in a row before the driver says so. One is a blip;
# a run of them means the CLI cannot see the kernel at all.
UNKNOWN_TOLERANCE = 5

_unknown_streak: dict[str, int] = {}


def status_or_report(slug: str, reader=None, consecutive: int = UNKNOWN_TOLERANCE,
                     report=print) -> str:
    """Poll a kernel, turning a persistent unreadable status into a report.

    UNKNOWN is what `parse_status` returns when the CLI prints nothing
    parseable - a dropped network, an expired token. It is not terminal, so the
    driver treated it as "still running" and polled a kernel it could no longer
    see, forever, printing nothing. The network did drop during this session.

    The reader is resolved on each call, not captured as a default: bound at
    import time it made `status` unsubstitutable, so a test that replaced it
    still shelled out to the real `kaggle` CLI and hung.
    """
    phase = (reader or status)(slug)
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
    """Descarga la salida de un kernel y reintenta ante una ausencia transitoria.

    Kaggle informa COMPLETE poco antes de que la salida esté disponible, y la
    CLI termina distinto de cero sin mucho detalle. Ignorar ese código de salida
    costó una ejecución GPU real: dinov2_vitb14 en ISIC 2019 se descartó por
    "falta el archivo de resultados", aunque su JSON se descargó segundos después.
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
                        downloader=None, attempts: int = 3,
                        delay: float = 30.0,
                        expected_run_token: str | None = None) -> bool:
    """Verify the result, re-fetching when it looks like the previous version.

    Kaggle can keep serving the old version's output for a while after a new
    one completes. The first download of resnet50's real fine-tuning run
    returned the smoke JSON, and a single check retired a finished GPU run.
    The run token catches the same staleness earlier, but only a re-download
    can turn it into a pass, so both checks share this retry loop.
    """
    fetch = downloader or download_with_retry
    for attempt in range(1, attempts + 1):
        try:
            verify_result(path, real=real, expected_run_token=expected_run_token)
            return True
        except StaleResult as exc:
            if attempt == attempts:
                print(f"  [{path.name}] {exc}")
                return False
            print(f"  [{path.name}] resultado viejo (intento {attempt}/{attempts}), "
                  "re-descargando")
            time.sleep(delay)
            fetch(slug, dest)
    return False


def collect(bank: str, model: str, user: str, real: bool, out_dir: Path,
            run_token: str, attempts: int = 3, delay: float = 30.0) -> str:
    """Descarga un kernel terminado y conserva solo un resultado del modo ejecutado.

    La descarga cae en un directorio temporal propio: validar sobre `out_dir`
    dejaba pasar un JSON que ya estaba ahi de una corrida anterior. Solo se
    copia al destino final lo que pasa la verificacion, y se copia por archivo
    temporal + `replace` para que nadie lea un JSON a medio escribir.
    """
    slug = f"{user}/{kernel_slug(bank, model)}"
    downloads = out_dir / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=downloads, prefix=f"{model}-") as temporary:
        download_dir = Path(temporary)
        if not download_with_retry(slug, download_dir):
            print(f"  [{model}] no se pudo descargar la salida de {slug}")
            return "failed"
        source = download_dir / result_name(bank, model)
        if not verify_with_refetch(source, real=real, slug=slug, dest=download_dir,
                                   downloader=download_with_retry, attempts=attempts,
                                   delay=delay, expected_run_token=run_token):
            return "failed"

        destination = out_dir / source.name
        staged_copy = out_dir / f".{source.name}.{uuid.uuid4().hex}.tmp"
        shutil.copy2(source, staged_copy)
        staged_copy.replace(destination)
    return "done"


def drive(bank: str, models: list[str], user: str, real: bool, out_dir: Path,
          state_file: Path, poll: int, limit: int, staging: Path) -> dict[str, dict[str, str]]:
    state = load_state(state_file)
    while pending(models, state) or any(
        _status(value) in {"pushing", "running"} for value in state.values()
    ):
        for model in list(state):
            entry = state[model]
            if _status(entry) not in {"pushing", "running"}:
                continue
            phase = status_or_report(f"{user}/{kernel_slug(bank, model)}")
            if phase not in TERMINAL:
                if _status(entry) == "pushing" and phase in {"QUEUED", "RUNNING"}:
                    entry["status"] = "running"
                    save_state(state_file, state)
                continue
            outcome = collect(bank, model, user, real, out_dir, entry["run_token"]) \
                if phase == "COMPLETE" else "failed"
            entry["status"] = outcome
            print(f"  [{model}] {display_state(phase)} -> {display_state(outcome)}")
            save_state(state_file, state)
        for model in pending(models, state):
            if free_slots(state, limit) <= 0:
                break
            run_token = uuid.uuid4().hex
            state[model] = {"status": "pushing", "run_token": run_token}
            save_state(state_file, state)
            ok, detail = push(bank, model, user, real, staging, run_token)
            if ok:
                state[model]["status"] = "running"
                print(f"  [{model}] publicación enviada")
                save_state(state_file, state)
            elif not is_slot_busy(detail):
                state[model]["status"] = "failed"
                print(f"  [{model}] falló la publicación: {detail}")
                save_state(state_file, state)
            else:
                # Without this the log goes silent for as long as the other
                # sessions last, and a driver waiting correctly is
                # indistinguishable from one that hung.
                print(f"  [{model}] sin slot GPU libre ({GPU_SESSION_LIMIT} en uso), "
                      "esperando")
                # The entry was written before the push so a crash mid-push is
                # recoverable; a rejected push never ran, so its run token must
                # not survive into the next round.
                del state[model]
                save_state(state_file, state)
                break
        if pending(models, state) or any(
            _status(value) in {"pushing", "running"} for value in state.values()
        ):
            # heartbeat: a bank whose models take an hour each printed nothing
            # for that hour, so a healthy run looked exactly like a hung one.
            corriendo = [m for m, v in state.items() if _status(v) == "running"]
            print(f"  [{time.strftime('%H:%M')}] corriendo={corriendo or '-'} "
                  f"hechos={sum(1 for v in state.values() if _status(v) == 'done')}"
                  f"/{len(models)} | siguiente poll en {poll}s")
            time.sleep(poll)
    return state


def report(bank: str, models: list[str], user: str, state_file: Path) -> None:
    state = load_state(state_file)
    for model in models:
        slug = f"{user}/{kernel_slug(bank, model)}"
        print(
            f"{model:18} estado={display_state(_status(state.get(model, 'pending'))):8} "
            f"kaggle={display_state(status(slug))}"
        )


def main(argv: list[str] | None = None) -> None:
    p = build_spanish_parser(description=__doc__)
    p.add_argument("--bank", required=True, choices=tuple(BANKS), help="Banco de experimentos Kaggle")
    p.add_argument("--user", required=True, help="Usuario de Kaggle")
    p.add_argument("--models", nargs="*", help="Modelos específicos del banco")
    p.add_argument("--real", action="store_true", help="Ejecuta el modo real en lugar de la prueba reducida")
    p.add_argument("--report", action="store_true", help="Muestra el estado y termina")
    p.add_argument("--poll", type=int, default=120, help="segundos entre consultas de estado")
    p.add_argument("--max-concurrent", type=int, default=GPU_SESSION_LIMIT, help="Máximo de sesiones GPU simultáneas")
    p.add_argument("--staging", default="/tmp/kaggle-staging", help="Directorio temporal de staging")
    p.add_argument("--via", choices=("kaggle",), default="kaggle",
                   help="Transporte del banco; solo la CLI oficial conserva este contrato")
    args = p.parse_args(argv)

    sys.path.insert(0, str(REPO / "src"))
    from dermssl.finetune import protocol

    models = args.models or list(protocol.MODELS)
    out_dir = output_dir(args.bank, REPO / "results")
    state_file = out_dir / ".run_state.json"

    if args.report:
        report(args.bank, models, args.user, state_file)
        return

    print(f"banco={args.bank} modelos={len(models)} modo={'real' if args.real else 'prueba'}")
    state = drive(args.bank, models, args.user, args.real, out_dir, state_file,
                  args.poll, args.max_concurrent, Path(args.staging))
    failed = [m for m, value in state.items() if _status(value) != "done"]
    print(f"\ncompletados: {len(models) - len(failed)}/{len(models)}")
    if failed:
        sys.exit(f"fallaron: {', '.join(failed)}")


if __name__ == "__main__":
    main()
