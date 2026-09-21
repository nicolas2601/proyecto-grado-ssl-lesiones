"""Checkpoint selection without ground truth.

``best-val`` keeps the epoch with the lowest pseudo-validation loss (computed
on confident pseudo-label pixels of the held-out pool split). ``last`` keeps the
final epoch. Neither mode receives a manual mask or a label.
"""

from __future__ import annotations

import math

MODES = ("best-val", "last")


class CheckpointSelector:
    def __init__(self, mode: str = "best-val") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown selection mode {mode!r}; expected one of {MODES}")
        self.mode = mode
        self.best_epoch = -1
        self.best_value = math.inf

    def update(self, epoch: int, value: float) -> bool:
        """Record an epoch; return True when it becomes the selected checkpoint."""
        if self.mode == "last":
            self.best_epoch, self.best_value = epoch, value
            return True
        if not math.isfinite(value) or value >= self.best_value:
            return False
        self.best_epoch, self.best_value = epoch, value
        return True

    def state_dict(self) -> dict:
        return {"mode": self.mode, "best_epoch": self.best_epoch, "best_value": self.best_value}

    @classmethod
    def from_state(cls, state: dict) -> CheckpointSelector:
        selector = cls(state["mode"])
        selector.best_epoch = int(state["best_epoch"])
        selector.best_value = float(state["best_value"])
        return selector
