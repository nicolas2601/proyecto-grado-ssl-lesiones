"""Compute-fraction-driven schedules for LR and EMA momentum.

Why this module exists
----------------------
The runs stop at a GPU-hour budget, not at an epoch count, so schedules keyed
to epochs would end mid-decay or decay too fast depending on hardware speed.
Keying them to the BUDGET FRACTION instead means two arms compared at 50% of
their compute are also at the same phase of their schedule - one less
confound in the equal-compute comparison.

Pure math, no torch: the loop calls these once per step with the fraction
reported by ``budget.ComputeBudget.fraction_done``.
"""

from __future__ import annotations

import math


def lr_at(
    fraction_done: float,
    base_lr: float,
    warmup_fraction: float,
    floor: float = 0.0,
) -> float:
    """Linear warmup to ``base_lr`` then cosine decay to ``floor``."""
    fraction_done = min(max(fraction_done, 0.0), 1.0)
    if warmup_fraction > 0 and fraction_done < warmup_fraction:
        return base_lr * fraction_done / warmup_fraction
    span = 1.0 - warmup_fraction
    progress = 0.0 if span <= 0 else (fraction_done - warmup_fraction) / span
    return floor + (base_lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_momentum_at(fraction_done: float, base: float) -> float:
    """Cosine ramp of the teacher EMA momentum from ``base`` to 1.0 (DINO).

    Early on the teacher tracks the student quickly; near the end it barely
    moves, which stabilizes the targets exactly when they matter most.
    """
    fraction_done = min(max(fraction_done, 0.0), 1.0)
    return 1.0 - (1.0 - base) * 0.5 * (1.0 + math.cos(math.pi * fraction_done))
