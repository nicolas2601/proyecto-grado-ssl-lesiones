"""One paired contrast of the bench and its Spanish table rendering.

A contrast is read from a committed JSON record, never computed here: the direction field
already accounts for lower-is-better metrics (ECE), so it is trusted as stored. ``flipped``
swaps arm and control (negating and reversing the interval) so a table can always put the
proposed model on the left.
"""

from __future__ import annotations

from dataclasses import dataclass

NO_VALUE = "—"
METRIC_ES = {"macro_f1": "F1 macro", "balanced_accuracy": "Exactitud balanceada", "accuracy": "Accuracy",
             "auc": "AUC OvR macro", "ece": "ECE (menor es mejor)", "mcc": "MCC", "mel_recall": "Recall de melanoma",
             "mel_auc": "AUC de melanoma"}
SWAPPED = {"arm_better": "control_better", "control_better": "arm_better", "indistinguishable": "indistinguishable"}
HEADER = ["Brazo", "Control", "Métrica", "Δ (brazo − control) [IC 95 %]", "El IC excluye 0", "Lectura", "Unidad",
          "Fuente"]


@dataclass(frozen=True)
class Contrast:
    arm: str
    control: str
    metric: str
    point_arm: float
    point_control: float
    delta: float
    low: float
    high: float
    excludes_zero: bool
    direction: str
    unit: str
    source: str


def from_record(rec: dict, *, arm: str, control: str, unit: str, source: str) -> Contrast:
    """Build a contrast from a JSON record (``point_*``, ``ci95_*``, ``excludes_zero``, ``direction``)."""
    return Contrast(arm, control, rec["metric"], rec["point_arm"], rec["point_control"], rec["point_delta"],
                    rec["ci95_low"], rec["ci95_high"], rec["excludes_zero"], rec["direction"], unit, source)


def flipped(c: Contrast) -> Contrast:
    """The same contrast with arm and control swapped (delta negated, interval reversed)."""
    return Contrast(c.control, c.arm, c.metric, c.point_control, c.point_arm, -c.delta, -c.high, -c.low,
                    c.excludes_zero, SWAPPED[c.direction], c.unit, c.source)


def num(x: float, signed: bool = False) -> str:
    text = f"{x:+.4f}" if signed else f"{x:.4f}"
    return text.replace(".", ",")


def interval(c: Contrast) -> str:
    return f"{num(c.delta, True)} [{num(c.low, True)}; {num(c.high, True)}]"


def reading(c: Contrast) -> str:
    """Spanish reading: «mejor» only when the interval excludes 0."""
    if c.direction == "indistinguishable":
        return "sin diferencia demostrable (el IC incluye 0)"
    winner = c.arm if c.direction == "arm_better" else c.control
    return f"{winner} mejor (el IC excluye 0)"


def row(c: Contrast) -> str:
    cells = [c.arm, c.control, METRIC_ES.get(c.metric, c.metric), interval(c), "sí" if c.excludes_zero else "no",
             reading(c), c.unit, f"`{c.source}`"]
    return "| " + " | ".join(cells) + " |"


def table(contrasts: list[Contrast]) -> list[str]:
    """Markdown table of contrasts, in the given order."""
    return ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER), *(row(c) for c in contrasts)]


def count_excluding(contrasts: list[Contrast]) -> int:
    return sum(c.excludes_zero for c in contrasts)
