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


def result_name(bank: str, model: str) -> str:
    """Se deriva del protocolo, nunca del nombre del banco.

    Usar `endswith("isic2020")` devolvía silenciosamente el nombre de archivo
    de HAM10000 para los otros bancos ISIC: se descargaba un banco y se
    verificaban los resultados de otro.
    """
    sys.path.insert(0, str(REPO / "src"))
    from dermssl.finetune import protocol
    return f"results_{model}{protocol.DATASETS[BANKS[bank][0]].json_suffix}.json"


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


def collect(bank: str, model: str, user: str, real: bool, out_dir: Path,
            run_token: str) -> str:
    """Descarga un kernel terminado y conserva solo un resultado del modo ejecutado."""
    slug = f"{user}/{kernel_slug(bank, model)}"
    downloads = out_dir / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=downloads, prefix=f"{model}-") as temporary:
        download_dir = Path(temporary)
        if not download_with_retry(slug, download_dir):
            print(f"  [{model}] no se pudo descargar la salida de {slug}")
            return "failed"
        source = download_dir / result_name(bank, model)
        try:
            verify_result(source, real=real, expected_run_token=run_token)
        except StaleResult as exc:
            print(f"  [{model}] {exc}")
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
            phase = status(f"{user}/{kernel_slug(bank, model)}")
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
                del state[model]
                save_state(state_file, state)
                break
        if pending(models, state) or any(
            _status(value) in {"pushing", "running"} for value in state.values()
        ):
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
    out_dir = REPO / "results" / args.bank
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
