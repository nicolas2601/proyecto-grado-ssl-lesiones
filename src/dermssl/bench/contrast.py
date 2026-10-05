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
    unavailable: str | None = None  # the stored reason when the contrast could not be computed; no number then


def from_record(rec: dict, *, arm: str, control: str, unit: str, source: str) -> Contrast:
    """Build a contrast from a JSON record (``point_*``, ``ci95_*``, ``excludes_zero``, ``direction``).

    A record stored as ``available: false`` carries only its ``reason``: it becomes an unavailable contrast.
    """
    if rec.get("available") is False:
        nan = float("nan")
        return Contrast(arm, control, rec["metric"], nan, nan, nan, nan, nan, False, "indistinguishable", unit,
                        source, rec["reason"])
    return Contrast(arm, control, rec["metric"], rec["point_arm"], rec["point_control"], rec["point_delta"],
                    rec["ci95_low"], rec["ci95_high"], rec["excludes_zero"], rec["direction"], unit, source)


def flipped(c: Contrast) -> Contrast:
    """The same contrast with arm and control swapped (delta negated, interval reversed)."""
    if c.unavailable:
        return Contrast(c.control, c.arm, c.metric, c.point_control, c.point_arm, c.delta, c.high, c.low,
                        False, c.direction, c.unit, c.source, c.unavailable)
    return Contrast(c.control, c.arm, c.metric, c.point_control, c.point_arm, -c.delta, -c.high, -c.low,
                    c.excludes_zero, SWAPPED[c.direction], c.unit, c.source)


def num(x: float, signed: bool = False) -> str:
    text = f"{x:+.4f}" if signed else f"{x:.4f}"
    return text.replace(".", ",")


def interval(c: Contrast) -> str:
    if c.unavailable:
        return c.unavailable
    return f"{num(c.delta, True)} [{num(c.low, True)}; {num(c.high, True)}]"


def _excludes(c: Contrast) -> str:
    return NO_VALUE if c.unavailable else ("sí" if c.excludes_zero else "no")


def reading(c: Contrast) -> str:
    """Spanish reading: «mejor» only when the interval excludes 0."""
    if c.unavailable:
        return NO_VALUE
    if c.direction == "indistinguishable":
        return "sin diferencia demostrable (el IC incluye 0)"
    winner = c.arm if c.direction == "arm_better" else c.control
    return f"{winner} mejor (el IC excluye 0)"


def row(c: Contrast) -> str:
    cells = [c.arm, c.control, METRIC_ES.get(c.metric, c.metric), interval(c), _excludes(c), reading(c), c.unit,
             f"`{c.source}`"]
    return "| " + " | ".join(cells) + " |"


def table(contrasts: list[Contrast]) -> list[str]:
    """Markdown table of contrasts, in the given order."""
    return ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER), *(row(c) for c in contrasts)]


COMPACT_HEADER = ["Brazo", "Control", "Métrica", "Δ (brazo − control) [IC 95 %]", "El IC excluye 0", "Lectura"]


def compact_cells(c: Contrast) -> list[str]:
    """Cells without unit and source (stated once in the section note)."""
    return [c.arm, c.control, METRIC_ES.get(c.metric, c.metric), interval(c), _excludes(c), reading(c)]


def count_excluding(contrasts: list[Contrast]) -> int:
    return sum(c.excludes_zero for c in contrasts)
