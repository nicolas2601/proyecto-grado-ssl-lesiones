"""Experiment tracking for the fine-tuning bank: W&B, MLflow and TensorBoard.

Why three backends
------------------
Dr. Karen's lab already works with Weights & Biases (offline on the cluster,
synced later). MLflow's file store stays as the on-cluster audit trail the
pretraining runs already use. TensorBoard is what the UNAB cluster asks for:
its MLflow server is shared by every student, so a private event-file
directory per run is the dashboard the director reads without exposing the
experiments to the whole platform. None is required: a missing library
degrades to a NullTracker that says so loudly, and the JSON results file is
always written regardless.

Every tracker takes the library by injection so the tests run without wandb,
mlflow or torch installed.
"""

from __future__ import annotations

from pathlib import Path
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


class TensorBoardTracker:
    """Event files under ``<root>/<run>/`` through an injected SummaryWriter class.

    Hyper-parameters go in as a text card rather than ``add_hparams``: the
    hparams plugin needs numeric metrics at start time and rewrites the run
    layout, which only muddies a directory that one job owns entirely.
    """

    def __init__(self, writer_cls: Any, root: str, run: str):
        self._writer_cls = writer_cls
        self._log_dir = str(Path(root) / run)
        self._writer: Any = None

    def start(self, params: dict[str, Any]) -> None:
        self._writer = self._writer_cls(log_dir=self._log_dir)
        card = "\n".join(f"{k}: {v}" for k, v in sorted(params.items()))
        self._writer.add_text("hparams", card)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._writer.add_scalar(key, value, global_step=step)

    def log_artifact(self, path: str) -> None:
        self._writer.add_text("artifacts", path)

    def end(self) -> None:
        self._writer.flush()
        self._writer.close()


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
    summary_writer_cls: Any | None = None,
    tensorboard_dir: str = "",
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
    if summary_writer_cls is not None and tensorboard_dir:
        trackers.append(TensorBoardTracker(summary_writer_cls, root=tensorboard_dir, run=run))
    return CompositeTracker(trackers)
