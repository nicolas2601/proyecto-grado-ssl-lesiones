"""Every file the bench reads must be versioned, not merely present on the author's disk.

A global ``*.log`` ignore kept the ISIC 2020 epoch logs out of the repository while the tables and
their tests kept passing locally; a clean clone failed. This reads the same JSON the bench reads,
decides which logs it needs, and asks git (not the file system) whether they are tracked.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from dermssl.bench.ft_curve import LOG_DIR
from dermssl.bench.rows import DATASETS, MODELS

ROOT = Path(__file__).resolve().parents[1]
FT = DATASETS["isic2020"]


def _tracked(rel: str) -> bool:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel], capture_output=True)
    return out.returncode == 0


def _logs_the_bench_needs() -> list[str]:
    """Epoch logs read when the result JSON carries no ``curva_epocas`` (see ``ft_curve``)."""
    needed = []
    for model in MODELS:
        result = ROOT / FT.finetune / f"results_{model}_ft{FT.suffix}.json"
        if not result.is_file() or "curva_epocas" in json.loads(result.read_text(encoding="utf-8")):
            continue
        needed.append(f"{LOG_DIR}/ft_{model}.epochs.log")
    return needed


@pytest.mark.skipif(not (ROOT / ".git").exists(), reason="not a git checkout")
def test_the_epoch_logs_the_bench_reads_are_versioned():
    needed = _logs_the_bench_needs()
    assert needed, "at least the two models launched before the curve was stored in the JSON need their log"
    untracked = [rel for rel in needed if not _tracked(rel)]
    assert not untracked, f"read by the bench but not versioned (is a global ignore hiding them?): {untracked}"
