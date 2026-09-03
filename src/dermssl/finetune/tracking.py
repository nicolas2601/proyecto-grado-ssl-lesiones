"""Experiment tracking for the fine-tuning bank: W&B + MLflow, both optional.

Why two backends
----------------
Dr. Karen's lab already works with Weights & Biases (offline on the cluster,
synced later). That is the dashboard the whole team can read. MLflow's file
store stays as the on-cluster audit trail the pretraining runs already use.
Neither is required: a missing library degrades to a NullTracker that says
so loudly, and the JSON results file is always written regardless.

Both trackers take the library module by injection so the tests run without
wandb or mlflow installed.
"""

from __future__ import annotations

from typing import Any

from dermssl.pretrain.tracking import MlflowTracker, NullTracker


class WandbTracker:
    """Thin delegation to an injected ``wandb`` module."""

    def __init__(self, wandb_module: Any, project: str, run: str, mode: str,
                 entity: str = "", tags: list[str] | None = None,
                 group: str = ""):
        self._wandb = wandb_module
        self._project = project
        self._run = run
        self._mode = mode
        self._entity = entity or None
        self._tags = list(tags or [])
        self._group = group or None
        self._active = None

    def start(self, params: dict[str, Any]) -> None:
        self._active = self._wandb.init(
            project=self._project, entity=self._entity, name=self._run,
            mode=self._mode, config=params, tags=self._tags, group=self._group,
            reinit=True,
        )

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        payload = {key: value}
        if step is None:
            self._wandb.log(payload)
        else:
            self._wandb.log(payload, step=step)

    def log_artifact(self, path: str) -> None:
        self._wandb.save(path)

    def end(self) -> None:
        self._wandb.finish()


class CompositeTracker:
    """Fan every call out to several trackers, in order."""

    def __init__(self, trackers: list[Any]):
        self.trackers = [t for t in trackers if not isinstance(t, NullTracker)]

    @property
    def is_null(self) -> bool:
        return not self.trackers

    def start(self, params: dict[str, Any]) -> None:
        for t in self.trackers:
            t.start(params)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        for t in self.trackers:
            t.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.log_metric(key, float(value), step=step)

    def log_artifact(self, path: str) -> None:
        for t in self.trackers:
            t.log_artifact(path)

    def end(self) -> None:
        for t in self.trackers:
            t.end()


def build_trackers(
    *,
    wandb_module: Any | None,
    mlflow_module: Any | None,
    run: str,
    wandb_project: str,
    wandb_mode: str,
    wandb_entity: str = "",
    wandb_tags: list[str] | None = None,
    wandb_group: str = "",
    mlflow_experiment: str = "",
    mlflow_tracking_uri: str = "",
) -> CompositeTracker:
    trackers: list[Any] = []
    if wandb_module is not None and wandb_mode != "disabled":
        trackers.append(WandbTracker(
            wandb_module, project=wandb_project, run=run, mode=wandb_mode,
            entity=wandb_entity, tags=wandb_tags, group=wandb_group))
    if mlflow_module is not None and mlflow_tracking_uri:
        trackers.append(MlflowTracker(
            mlflow_module, experiment=mlflow_experiment,
            tracking_uri=mlflow_tracking_uri, run=run))
    return CompositeTracker(trackers)
