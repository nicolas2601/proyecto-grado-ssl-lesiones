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

Usage on the cluster (rank 0):
    import mlflow
    tracker = build_tracker(mlflow, experiment=cfg.mlflow_experiment,
                            tracking_uri=cfg.tracking_uri,
                            run=run_name(cfg.method, cfg.backbone, cfg.seed),
                            rank=rank)
"""

from __future__ import annotations

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


def build_tracker(
    mlflow_module: Any | None,
    experiment: str,
    tracking_uri: str,
    run: str,
    rank: int,
) -> NullTracker | MlflowTracker:
    """MlflowTracker on rank 0 when mlflow exists; NullTracker otherwise.

    The caller decides how loudly to announce a NullTracker on rank 0 - the
    launch script prints a warning so an untracked run is never a surprise.
    """
    if rank != 0 or mlflow_module is None:
        return NullTracker()
    return MlflowTracker(
        mlflow_module, experiment=experiment, tracking_uri=tracking_uri, run=run
    )
