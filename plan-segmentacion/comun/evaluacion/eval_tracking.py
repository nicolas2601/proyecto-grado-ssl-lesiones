"""Publish an evaluation report to Weights & Biases.

Split from ``eval_report`` on purpose: computing the report and shipping it
to a dashboard are different jobs with different failure modes. Nothing here
is required for the CSV and the JSON, which stay the source of truth.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from eval_report import STRATA_METRICS

def flatten_summary(summary: Mapping, prefix: str) -> dict:
    """``{"dice": {"mean": x}} -> {"<prefix>/dice_mean": x}``; skips non-metrics."""
    flat = {}
    for metric, stats in summary.items():
        if not isinstance(stats, Mapping):
            continue
        for statistic, value in stats.items():
            flat[f"{prefix}/{metric}_{statistic}"] = value
    return flat


def strata_table_rows(strata: Mapping) -> list[dict]:
    """Flat rows for a tracking table: one per bucket, sorted by bucket name."""
    table = []
    for bucket, entry in strata["buckets"].items():
        row = {"bucket": bucket, "status": entry["status"], "n": entry["n"]}
        for metric in STRATA_METRICS:
            row[f"{metric}_mean"] = entry[metric]["mean"]
            row[f"{metric}_median"] = entry[metric]["median"]
        table.append(row)
    return table


def track_evaluation(args, summary: Mapping, strata, rows: Sequence[Mapping], overlays) -> bool:
    """Mirror the report to Weights & Biases; returns False when disabled."""
    from wandb_logging import WandbSink

    prefix = args.eval_name
    sink = WandbSink(
        project=args.wandb_project,
        run_name=args.wandb_run_name or f"eval-{prefix}",
        config=vars(args),
        tags=("evaluation", prefix),
    )
    if not sink.enabled:
        return False
    sink.log(flatten_summary(summary, prefix))
    if strata is not None:
        sink.log_table(f"{prefix}/strata", strata_table_rows(strata))
    dice = [float(row["dice"]) for row in rows if not math.isnan(float(row["dice"]))]
    sink.log_histogram(f"{prefix}/dice", dice)
    sink.log_images(f"{prefix}/overlays", list(overlays), [Path(p).stem for p in overlays])
    sink.finish()
    return True
