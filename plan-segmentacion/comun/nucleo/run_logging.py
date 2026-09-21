"""JSONL metrics logging and throughput accounting for the pilot sizing."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path


class JsonlLogger:
    """Append one JSON object per line and echo it to stdout."""

    def __init__(self, path: Path, *, echo: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def log(self, record: dict) -> None:
        line = json.dumps(record, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if self.echo:
            print(line, flush=True)


class ThroughputMeter:
    """Wall-clock images per second, with an injectable clock for tests."""

    def __init__(self, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._start = 0.0
        self.images = 0

    def start(self) -> None:
        self._start = self._clock()
        self.images = 0

    def add(self, count: int) -> None:
        self.images += int(count)

    def snapshot(self) -> dict:
        seconds = self._clock() - self._start
        rate = self.images / seconds if seconds > 0 else 0.0
        return {"seconds": seconds, "images": self.images, "images_per_second": rate}


def timing_summary(epoch_records: Sequence[dict]) -> dict:
    """Aggregate per-epoch ``seconds`` and ``images`` (global, all ranks)."""
    epochs = len(epoch_records)
    total_seconds = float(sum(r["seconds"] for r in epoch_records))
    images = int(sum(r["images"] for r in epoch_records))
    return {
        "epochs": epochs,
        "total_seconds": total_seconds,
        "mean_seconds_per_epoch": total_seconds / epochs if epochs else 0.0,
        "images_processed": images,
        "images_per_second": images / total_seconds if total_seconds > 0 else 0.0,
    }
