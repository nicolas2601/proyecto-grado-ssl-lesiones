"""Contabilidad del presupuesto de horas-GPU para comparaciones con cómputo igual.

Why this module exists
----------------------
The first thesis novelty (docs/15, section 10.1) is comparing SSL methods at
EQUAL COMPUTE, not equal epochs: macro-F1 and AUC as a function of GPU-hours
spent. That only works if every run measures compute the same way and stops
and checkpoints at the same compute fractions. Centralizing the arithmetic
here keeps the training loop trivial ("ask the budget, obey") and makes the
comparison auditable: the same tested code decides the x-axis of the curve
for every arm.

GPU-hours = wall-clock seconds x number of GPUs / 3600. A 10 GPU-hour budget
means 2.5 wall-clock hours on 4 GPUs.
"""

from __future__ import annotations


class ComputeBudget:
    """Registra el cómputo consumido frente a un presupuesto fijo de horas-GPU."""

    def __init__(self, gpu_hours: float, n_gpus: int, fractions: tuple[float, ...]):
        if gpu_hours <= 0:
            raise ValueError(f"gpu_hours debe ser positivo; se recibió {gpu_hours}")
        if n_gpus < 0:
            raise ValueError(f"n_gpus no puede ser negativo; se recibió {n_gpus}")
        self.gpu_hours = float(gpu_hours)
        self.n_gpus = int(n_gpus)
        self.fractions = tuple(fractions)

    def gpu_hours_elapsed(self, wall_seconds: float) -> float:
        return wall_seconds * self.n_gpus / 3600.0

    def fraction_done(self, wall_seconds: float) -> float:
        return min(1.0, self.gpu_hours_elapsed(wall_seconds) / self.gpu_hours)

    def should_stop(self, wall_seconds: float) -> bool:
        return self.gpu_hours_elapsed(wall_seconds) >= self.gpu_hours

    def due_checkpoints(
        self, wall_seconds: float, already_saved: set[float]
    ) -> list[float]:
        """Fracciones alcanzadas cuyo checkpoint todavía no se guardó."""
        done = self.fraction_done(wall_seconds)
        return [f for f in self.fractions if f <= done and f not in already_saved]


def checkpoint_tag(fraction: float) -> str:
    """Etiqueta segura para nombre de archivo: 0.25 -> 'ckpt_c025'."""
    return f"ckpt_c{round(fraction * 100):03d}"
