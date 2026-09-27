"""The docs/28 adaptation levers as plain values, importable without torch.

``config`` validates them and must stay torch-free; ``adaptation`` applies
them. Every default reproduces the v6 run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Levers:
    freeze_blocks: int = 0
    layer_decay: float = 1.0
    anchor_l2sp: float = 0.0
    anchor_distill: float = 0.0
    tone_consistency_weight: float = 0.0

    def is_default(self) -> bool:
        return self == Levers()

    def validate(self, *, paired: bool) -> "Levers":
        if self.freeze_blocks < 0:
            raise ValueError(f"freeze_blocks={self.freeze_blocks}: debe ser >= 0")
        if not 0.0 < self.layer_decay <= 1.0:
            raise ValueError(f"layer_decay={self.layer_decay}: debe estar en (0, 1]")
        for name in ("anchor_l2sp", "anchor_distill", "tone_consistency_weight"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}={getattr(self, name)}: debe ser >= 0")
        if self.tone_consistency_weight > 0 and not paired:
            raise ValueError(
                "tone_consistency_weight > 0 exige adaptation_mode="
                "original_transformed_consistency: sin la vista de tono no hay par"
            )
        return self
