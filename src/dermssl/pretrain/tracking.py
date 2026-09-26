"""Experiment tracking facade over MLflow.

Why this module exists
----------------------
MLflow is how the project gets traceability: every cluster run records its
params (full config + git SHA), metrics against step AND wall time (the
equal-compute curves read straight from this), and artifacts (checkpoints).
But the training loop must not depend on mlflow being importable - a run on a
machine without it should still train and say LOUDLY that nothing is being
recorded. Hence a facade with two implementations:

* ``MlflowTracker`` delegates to the mlflow module it is HANDED (dependency
  injection - which is also what makes it testable without mlflow installed).
* ``NullTracker`` accepts everything and records nothing, used both as the
  fallback and on nonzero DDP ranks, where logging would create one duplicate
  MLflow run per GPU.
* ``WandbTracker`` mirrors the same calls in Weights & Biases when a project
  is requested; ``CompositeTracker`` fans out to both sinks.

Usage on the cluster (rank 0):
    import mlflow
    tracker = build_tracker(mlflow, experiment=cfg.mlflow_experiment,
                            tracking_uri=cfg.tracking_uri,
                            run=run_name(cfg.method, cfg.backbone, cfg.seed),
                            rank=rank)
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from typing import Any


class WandbUnavailable(RuntimeError):
    """Tracking was asked for and cannot be delivered.

    Separate from the fail-safe path on purpose. Not asking for W&B is fine and
    stays silent; asking for it and not getting it is an error, because the run
    would spend GPU hours and leave nothing anyone can link to.
    """


# ``offline`` and ``dryrun`` write to disk and never reach wandb.ai, so the run
# has no URL. Every result is reported to Pau with its link, so a mode that
# cannot produce one is refused rather than discovered afterwards.
_MODES_WITHOUT_A_URL = frozenset({"offline", "dryrun", "disabled"})


def require_wandb(
    wandb_project: str,
    *,
    importer=importlib.import_module,
    environ: dict[str, str] | None = None,
) -> Any | None:
    """The wandb module when a project was requested, or ``None`` when it was not.

    Raises :class:`WandbUnavailable` when tracking was requested and cannot be
    provided. Checked here rather than at ``init`` because ``WandbTracker``
    swallows its own exceptions by design, so a missing credential surfaces as
    one printed warning in the middle of a training log and nothing else.

    Engram #1590 is the concrete case: wandb lived in
    ``~/work/proyectonp/.venv`` and the run used ``/opt/conda/bin/python``, so
    the import failed and the whole budget produced no dashboard. The message
    names the interpreter because the fix is always "use the other python".
    """
    if not wandb_project:
        return None

    environ = os.environ if environ is None else environ

    mode = environ.get("WANDB_MODE", "").strip().lower()
    if mode in _MODES_WITHOUT_A_URL:
        raise WandbUnavailable(
            f"WANDB_MODE={mode!r} no publica el run en wandb.ai, asi que la corrida "
            "no tendria link que reportar. Quita WANDB_MODE o no pidas "
            "--wandb-project."
        )

    try:
        module = importer("wandb")
    except ImportError as error:
        raise WandbUnavailable(
            "se pidio seguimiento en Weights & Biases con --wandb-project pero el "
            f"paquete wandb no esta disponible en {sys.executable}: instalalo en ese "
            f"interprete o quita --wandb-project ({error})"
        ) from error

    if not _has_credential(module, environ):
        raise WandbUnavailable(
            "wandb esta instalado pero no hay credencial: ni WANDB_API_KEY en el "
            "entorno ni una clave guardada por `wandb login`. Sin credencial "
            "wandb.init falla y el tracker se lo traga, asi que la corrida quedaria "
            "sin registrar."
        )
    return module


def _has_credential(wandb_module: Any, environ: dict[str, str]) -> bool:
    """A key in the environment, or one ``wandb login`` left in ~/.netrc."""
    if environ.get("WANDB_API_KEY"):
        return True
    api = getattr(wandb_module, "api", None)
    return bool(getattr(api, "api_key", None))


def run_name(method: str, backbone: str, seed: int, arm: str = "") -> str:
    """Readable run name; ``arm`` tells two runs of the same recipe apart.

    Both arms of the thesis share method, backbone and seed - what separates
    them is the second view - so without the arm they land in the dashboard
    under the same name.
    """
    base = f"{method}-{backbone}-s{seed}"
    return f"{base}-{arm}" if arm else base


def run_id_for(output_dir: str | Path) -> str:
    """Stable W&B id derived from the run's output directory.

    A pod restart relaunches the same command, and the training resumes from
    ``resume.pt``; without a stable id W&B opens a second run and the arm's
    curves arrive split in two. The output directory is the one input that is
    unique per arm and identical across a relaunch.
    """
    resolved = str(Path(output_dir)).rstrip("/")
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]


class NullTracker:
    """Records nothing; every method is a safe no-op."""

    def start(self, params: dict[str, Any]) -> None:
        pass

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        pass

    def log_artifact(self, path: str) -> None:
        pass

    def end(self) -> None:
        pass


class MlflowTracker:
    """Thin, ordered delegation to an injected mlflow-compatible module."""

    def __init__(self, mlflow_module: Any, experiment: str, tracking_uri: str, run: str):
        self._mlflow = mlflow_module
        self._experiment = experiment
        self._tracking_uri = tracking_uri
        self._run = run

    def start(self, params: dict[str, Any]) -> None:
        if self._tracking_uri:
            if self._tracking_uri.startswith("file:"):
                # MLflow 3.x gates the filesystem backend behind this opt-in;
                # without it set_experiment raises before the run even starts.
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            self._mlflow.set_tracking_uri(self._tracking_uri)
        self._mlflow.set_experiment(self._experiment)
        self._mlflow.start_run(run_name=self._run)
        self._mlflow.log_params(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, path: str) -> None:
        self._mlflow.log_artifact(path)

    def end(self) -> None:
        self._mlflow.end_run()


class WandbTracker:
    """Mirror of the run in Weights & Biases; every call is fail-safe.

    The sink is optional visibility, never a dependency of the training: an
    offline pod, a lost ``~/.netrc`` or a network hiccup must not kill hours
    of GPU work, so each call swallows its exception and prints once.
    """

    def __init__(
        self,
        wandb_module: Any,
        project: str,
        run: str,
        group: str = "",
        run_id: str = "",
        strict: bool = False,
    ):
        self._wandb = wandb_module
        self._project = project
        self._run = run
        self._group = group
        self._run_id = run_id
        self._active = None
        self._warned = False
        # Fail-safe is right for a 20-hour pretrain, where losing the sink is
        # cheaper than losing the run. It is wrong for a verification gate whose
        # entire output IS the dashboard entry, so that caller opts in here.
        self._strict = strict

    @property
    def run_url(self) -> str:
        """The link to report, or "" before ``start`` (or when the sink failed)."""
        return str(getattr(self._active, "url", "") or "")

    def _guard(self, action, *args: Any) -> None:
        try:
            action(*args)
        except Exception as error:  # noqa: BLE001 - a sink failure is not a run failure
            if self._strict:
                raise
            if not self._warned:
                print(f"ADVERTENCIA: Weights & Biases dejo de registrar ({error}).")
                self._warned = True

    def start(self, params: dict[str, Any]) -> None:
        def action() -> None:
            self._active = self._wandb.init(
                project=self._project, name=self._run, group=self._group or None,
                config=dict(params), resume="allow", id=self._run_id or None,
            )

        self._guard(action)
        if self._strict and self._active is None:
            raise WandbUnavailable(
                "wandb.init no devolvio un run, asi que cada log posterior seria un "
                "no-op silencioso y la corrida quedaria sin registrar."
            )

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        if self._active is None:
            return
        self._guard(lambda: self._active.log({key: value}, step=step))

    def log_artifact(self, path: str) -> None:
        if self._active is None:
            return
        self._guard(lambda: self._wandb.save(path, base_path=os.path.dirname(path)))

    def end(self) -> None:
        if self._active is None:
            return
        self._guard(self._wandb.finish)


class CompositeTracker:
    """Fans every call out to each sink, in order."""

    def __init__(self, sinks: list[Any]):
        self._sinks = list(sinks)

    def start(self, params: dict[str, Any]) -> None:
        for sink in self._sinks:
            sink.start(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        for sink in self._sinks:
            sink.log_metric(key, value, step=step)

    def log_artifact(self, path: str) -> None:
        for sink in self._sinks:
            sink.log_artifact(path)

    def end(self) -> None:
        for sink in self._sinks:
            sink.end()


def build_tracker(
    mlflow_module: Any | None,
    experiment: str,
    tracking_uri: str,
    run: str,
    rank: int,
    wandb_module: Any | None = None,
    wandb_project: str = "",
    wandb_group: str = "",
    wandb_run_id: str = "",
) -> NullTracker | MlflowTracker | WandbTracker | CompositeTracker:
    """The sinks available on rank 0, or NullTracker.

    MLflow when its module exists; Weights & Biases when its module exists
    AND a project was requested; both together when both apply. The caller
    decides how loudly to announce a NullTracker on rank 0 - the launch
    script prints a warning so an untracked run is never a surprise.
    """
    if rank != 0:
        return NullTracker()
    sinks: list[Any] = []
    if mlflow_module is not None:
        sinks.append(MlflowTracker(
            mlflow_module, experiment=experiment, tracking_uri=tracking_uri, run=run
        ))
    if wandb_module is not None and wandb_project:
        sinks.append(WandbTracker(
            wandb_module, project=wandb_project, run=run, group=wandb_group,
            run_id=wandb_run_id,
        ))
    if not sinks:
        return NullTracker()
    if len(sinks) == 1:
        return sinks[0]
    return CompositeTracker(sinks)
