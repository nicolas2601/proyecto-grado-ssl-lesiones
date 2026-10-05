"""Learning-rate schedule shared by every training stage."""

from __future__ import annotations

import math


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup to ``base_lr`` followed by cosine decay towards zero."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
