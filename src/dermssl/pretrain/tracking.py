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

import os
from typing import Any


def run_name(method: str, backbone: str, seed: int) -> str:
    return f"{method}-{backbone}-s{seed}"


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

    def __init__(self, wandb_module: Any, project: str, run: str, group: str = ""):
        self._wandb = wandb_module
        self._project = project
        self._run = run
        self._group = group
        self._active = None
        self._warned = False

    def _guard(self, action, *args: Any) -> None:
        try:
            action(*args)
        except Exception as error:  # noqa: BLE001 - a sink failure is not a run failure
            if not self._warned:
                print(f"ADVERTENCIA: Weights & Biases dejo de registrar ({error}).")
                self._warned = True

    def start(self, params: dict[str, Any]) -> None:
        def action() -> None:
            self._active = self._wandb.init(
                project=self._project, name=self._run, group=self._group or None,
                config=dict(params), resume="allow",
            )

        self._guard(action)

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
        sinks.append(WandbTracker(wandb_module, project=wandb_project, run=run, group=wandb_group))
    if not sinks:
        return NullTracker()
    if len(sinks) == 1:
        return sinks[0]
    return CompositeTracker(sinks)
