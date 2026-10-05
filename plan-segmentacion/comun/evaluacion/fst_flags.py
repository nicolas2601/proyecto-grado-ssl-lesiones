"""Tone and FST distributions, and the records whose tone reading is suspect.

docs/26 §6.4 (step 1): the estimator already knows when a frame has dark,
high-contrast structures (hair, rulers, ink) or a lowered confidence, but the
summary used to mix those frames with the clean ones. The flagged block counts
them apart, with their own distributions, and never changes the overall ones.

A record is flagged when ``artifact_detected`` is true or ``confidence < 1``.
Records without a bucket (``ita_indeterminate``, quarantine) carry confidence 0
by construction, so ``n_confidence_below_1`` includes them; the
``artifact_detected`` sub-block isolates the hair/artifact question.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

TONE_LABELS = ("dark", "medium", "light")
TONE_UNAVAILABLE = "unavailable"
FST_UNAVAILABLE = "unavailable"
FULL_CONFIDENCE = 1.0
FLAG_RULE = "artifact_detected or confidence < 1"


def fst_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Bucket -> count, skipping records without a bucket."""
    buckets = Counter(
        str(row["FST_bucket"]) for row in rows if row["FST_bucket"] != FST_UNAVAILABLE
    )
    return dict(sorted(buckets.items()))


def tone_distribution(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Tercile -> count in dark/medium/light order, skipping unavailable ones."""
    tones = Counter(
        str(row["tone_tercile"]) for row in rows if row["tone_tercile"] != TONE_UNAVAILABLE
    )
    return {label: tones[label] for label in TONE_LABELS if tones[label]}


def _below_full_confidence(row: dict[str, Any]) -> bool:
    return float(row["confidence"]) < FULL_CONFIDENCE


def is_flagged(row: dict[str, Any]) -> bool:
    return bool(row.get("artifact_detected")) or _below_full_confidence(row)


def _distributions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "FST_distribution": fst_distribution(rows),
        "tone_distribution": tone_distribution(rows),
    }


def flagged_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts and distributions of the flagged records of one dataset."""
    flagged = [row for row in rows if is_flagged(row)]
    artifact = [row for row in flagged if row.get("artifact_detected")]
    return {
        "rule": FLAG_RULE,
        **_distributions(flagged),
        "n_artifact_detected": len(artifact),
        "n_confidence_below_1": sum(_below_full_confidence(row) for row in flagged),
        "states": dict(sorted(Counter(row["status"] for row in flagged).items())),
        "artifact_detected": _distributions(artifact),
    }
