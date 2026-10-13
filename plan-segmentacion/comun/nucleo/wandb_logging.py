"""Optional Weights & Biases sink for ``JsonlLogger`` and the evaluation stage.

The cluster has no reachable TensorBoard or MLflow UI, so runs are mirrored to
W&B when credentials exist. Everything is opt-in: with no environment or
private netrc credential (or ``WANDB_MODE=offline``), no project, or no installed
client, every method is a no-op and training keeps going. ``WANDB_NETRC_FILE``
selects an explicit file; otherwise ~/.netrc is used. File keys are exposed to
the client only during initialization and are scrubbed from warnings/config.
"""

from __future__ import annotations

import hashlib
import netrc
import os
from contextlib import contextmanager
from threading import RLock
from pathlib import Path as _Path
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

OFFLINE_MODE = "offline"
RUN_ID_LENGTH = 12
NETRC_HOST = "api.wandb.ai"
NETRC_FILE_ENV = "WANDB_NETRC_FILE"
_INIT_LOCK = RLock()


def _scrub(text: str, api_key: str | None = None) -> str:
    """Remove the API key from a message before it reaches the terminal."""
    for key in (api_key, os.environ.get("WANDB_API_KEY")):
        if key:
            text = text.replace(key, "***")
    return text


def _safe_payload(value, api_key: str | None):
    """Keep the resolved credential out of nested run configuration and records."""
    if isinstance(value, str):
        return _scrub(value, api_key)
    if isinstance(value, Mapping):
        return {k: _safe_payload(v, api_key) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_payload(v, api_key) for v in value]
    return value


@contextmanager
def _client_credential(api_key: str | None):
    """Expose a file credential only during SDK initialization, then restore it."""
    with _INIT_LOCK:
        previous = os.environ.get("WANDB_API_KEY")
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("WANDB_API_KEY", None)
            else:
                os.environ["WANDB_API_KEY"] = previous


def run_id(group: str | None, run_name: str) -> str:
    """Deterministic identifier so a resumed training reuses the same run."""
    digest = hashlib.sha1(f"{group or ''}/{run_name}".encode())
    return digest.hexdigest()[:RUN_ID_LENGTH]


def _netrc_key() -> str | None:
    """Read an explicit netrc path, or ~/.netrc when none was configured."""
    try:
        configured = os.environ.get(NETRC_FILE_ENV, "").strip()
        path = _Path(configured).expanduser() if configured else _Path.home() / ".netrc"
        if not path.is_file():
            return None
        if os.name == "posix" and path.stat().st_mode & 0o077:
            return None
        entry = netrc.netrc(str(path)).authenticators(NETRC_HOST)
    except Exception:  # a malformed or unreadable file is simply no credential
        return None
    return entry[2] if entry and entry[2] else None


def _netrc_has_wandb() -> bool:
    return bool(_netrc_key())


def credentials_available() -> bool:
    """True when the client can reach a backend or write an offline run."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WANDB_MODE", "").strip().lower() == OFFLINE_MODE:
        return True
    return _netrc_has_wandb()


def _warn(action: str, error: Exception, api_key: str | None = None) -> None:
    """One line, no credentials, so a broken backend is visible but harmless."""
    detail = _scrub(f"{type(error).__name__}: {error}", api_key)
    print(f"aviso: seguimiento W&B deshabilitado en {action} ({detail})", flush=True)


def _disabled(reason: str) -> None:
    """Say once, out loud, that a requested run will not be tracked."""
    print(f"aviso: seguimiento W&B deshabilitado: {reason}", flush=True)


def _import_wandb():
    """Return the client module, or None when it is not installed."""
    try:
        import wandb
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return wandb


def record_step(record: Mapping) -> int | None:
    """Epoch wins over step; without either the client picks its own counter."""
    for key in ("epoch", "step"):
        if key in record:
            return int(record[key])
    return None


class WandbSink:
    """Mirror of the JSONL records; disabled unless the run is fully configured."""

    def __init__(
        self,
        project: str,
        run_name: str,
        config: Mapping | None = None,
        group: str | None = None,
        tags: Sequence[str] = (),
    ) -> None:
        self._run = None
        self._wandb = None
        self.enabled = False
        self._api_key = None
        if not project:
            return
        offline = os.environ.get("WANDB_MODE", "").strip().lower() == OFFLINE_MODE
        self._api_key = os.environ.get("WANDB_API_KEY") or (None if offline else _netrc_key())
        if not self._api_key and not offline:
            _disabled("no hay credenciales (WANDB_API_KEY, WANDB_MODE=offline, WANDB_NETRC_FILE o ~/.netrc)")
            return
        client = _import_wandb()
        if client is None:
            _disabled("el paquete wandb no esta instalado")
            return
        self._wandb = client
        resolved_group = group or os.environ.get("WANDB_RUN_GROUP") or None
        try:
            with _client_credential(self._api_key):
                self._run = client.init(
                    project=project,
                    name=run_name or None,
                    id=run_id(resolved_group, run_name),
                    resume="allow",
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    group=resolved_group,
                    tags=list(tags),
                    config=_safe_payload(dict(config or {}), self._api_key),
                    reinit=True,
                )
        except Exception as error:  # tracking must never stop the training
            _warn("init", error, self._api_key)
            return
        self.enabled = True

    def _safely(self, action: str, operation: Callable[[], None]) -> None:
        """Run a client call; the first failure disables the sink for good."""
        try:
            operation()
        except Exception as error:  # tracking must never stop the training
            self.enabled = False
            _warn(action, error, self._api_key)

    def log(self, record: Mapping) -> None:
        if not self.enabled:
            return
        step = record_step(record)
        payload = _safe_payload(dict(record), self._api_key)
        self._safely(
            "log",
            lambda: self._run.log(payload)
            if step is None
            else self._run.log(payload, step=step),
        )

    def log_images(self, key: str, images: Sequence, captions: Sequence[str] = ()) -> None:
        """``images`` accepts PNG paths or HxWx3 uint8 arrays."""
        if not self.enabled or not len(images):
            return
        labels = (list(captions) + [None] * len(images))[: len(images)]
        self._safely("log_images", lambda: self._run.log({key: self._images(images, labels)}))

    def _images(self, images: Sequence, labels: Sequence) -> list:
        return [
            self._wandb.Image(
                str(item) if isinstance(item, (str, os.PathLike)) else item, caption=label
            )
            for item, label in zip(images, labels, strict=True)
        ]

    def log_table(self, key: str, rows: Sequence[Mapping]) -> None:
        if not self.enabled or not len(rows):
            return
        columns = list(rows[0].keys())
        data = [[row.get(column) for column in columns] for row in rows]
        self._safely(
            "log_table",
            lambda: self._run.log({key: self._wandb.Table(columns=columns, data=data)}),
        )

    def log_histogram(self, key: str, values: Sequence[float]) -> None:
        if not self.enabled or not len(values):
            return
        self._safely(
            "log_histogram",
            lambda: self._run.log({key: self._wandb.Histogram(list(values))}),
        )

    def finish(self) -> None:
        if not self.enabled:
            return
        self._safely("finish", self._run.finish)
        self.enabled = False


def default_run_name(args, stage: str) -> str:
    """Explicit name wins; otherwise ``<stage>-<output folder name>``."""
    explicit = getattr(args, "wandb_run_name", "")
    return explicit or f"{stage}-{Path(args.output_dir).name}"


def sink_from_args(args, stage: str, *, is_main: bool = True) -> WandbSink:
    """Sink for a training stage; always disabled outside the main rank."""
    return WandbSink(
        project=getattr(args, "wandb_project", "") if is_main else "",
        run_name=default_run_name(args, stage),
        config=vars(args),
        group=os.environ.get("WANDB_RUN_GROUP") or None,
        tags=(stage,),
    )


def add_tracking_arguments(parser) -> None:
    """Shared ``--wandb-*`` flags for the training entry points."""
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", ""),
        help="Proyecto de Weights & Biases (vacio = sin seguimiento)",
    )
    parser.add_argument(
        "--wandb-run-name", default="", help="Nombre de la corrida; por defecto <etapa>-<salida>"
    )

