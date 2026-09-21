"""Optional Weights & Biases sink for ``JsonlLogger`` and the evaluation stage.

The cluster has no reachable TensorBoard or MLflow UI, so runs are mirrored to
W&B when credentials exist. Everything is opt-in: with no ``WANDB_API_KEY``
(or ``WANDB_MODE=offline``), no project, or no installed client, every method
is a no-op and training keeps going. The key is read from the environment by
the client itself and is never logged, printed or stored in the run config.
"""

from __future__ import annotations

import hashlib
import netrc
import os
from pathlib import Path as _Path
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

OFFLINE_MODE = "offline"
RUN_ID_LENGTH = 12
NETRC_HOST = "api.wandb.ai"


def _scrub(text: str) -> str:
    """Remove the API key from a message before it reaches the terminal."""
    key = os.environ.get("WANDB_API_KEY")
    return text.replace(key, "***") if key else text


def run_id(group: str | None, run_name: str) -> str:
    """Deterministic identifier so a resumed training reuses the same run."""
    digest = hashlib.sha1(f"{group or ''}/{run_name}".encode())
    return digest.hexdigest()[:RUN_ID_LENGTH]


def _netrc_has_wandb() -> bool:
    """True when ~/.netrc holds a password for the W&B host (``wandb login`` writes it)."""
    path = _Path.home() / ".netrc"
    if not path.is_file():
        return False
    try:
        entry = netrc.netrc(str(path)).authenticators(NETRC_HOST)
    except Exception:  # a malformed or unreadable file is simply no credential
        return False
    return bool(entry and entry[2])


def credentials_available() -> bool:
    """True when the client can reach a backend or write an offline run."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    if os.environ.get("WANDB_MODE", "").strip().lower() == OFFLINE_MODE:
        return True
    return _netrc_has_wandb()


def _warn(action: str, error: Exception) -> None:
    """One line, no credentials, so a broken backend is visible but harmless."""
    detail = _scrub(f"{type(error).__name__}: {error}")
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
        if not project:
            return
        if not credentials_available():
            _disabled("no hay credenciales (WANDB_API_KEY, WANDB_MODE=offline o ~/.netrc)")
            return
        client = _import_wandb()
        if client is None:
            _disabled("el paquete wandb no esta instalado")
            return
        self._wandb = client
        resolved_group = group or os.environ.get("WANDB_RUN_GROUP") or None
        try:
            self._run = client.init(
                project=project,
                name=run_name or None,
                id=run_id(resolved_group, run_name),
                resume="allow",
                entity=os.environ.get("WANDB_ENTITY") or None,
                group=resolved_group,
                tags=list(tags),
                config=dict(config or {}),
                reinit=True,
            )
        except Exception as error:  # tracking must never stop the training
            _warn("init", error)
            return
        self.enabled = True

    def _safely(self, action: str, operation: Callable[[], None]) -> None:
        """Run a client call; the first failure disables the sink for good."""
        try:
            operation()
        except Exception as error:  # tracking must never stop the training
            self.enabled = False
            _warn(action, error)

    def log(self, record: Mapping) -> None:
        if not self.enabled:
            return
        step = record_step(record)
        payload = dict(record)
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
