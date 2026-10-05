"""Shared fixtures of the E3b tests: a real E3 run (fake extractor) whose selection adopts F4."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import test_e3_run as t3  # noqa: E402

from dermssl import e3_store as st  # noqa: E402

SIZES, label_of = t3.SIZES, t3.label_of


def build_e3_dir(out: Path, *, winner: str = "F4") -> tuple[Path, str]:
    """E3 stages extract/select/final in ``out``; the selection is rewritten to adopt ``winner``."""
    path, _ = t3.selected(out)
    doc = json.loads(path.read_text())
    doc.update(decision="adopt", winner=winner, val_plan=["B71", winner])
    st.write_json_atomic(path, doc)
    sha = st.file_sha256(path)
    pin = ["--selection", str(path), "--selection-sha256", sha]
    assert t3.cli(out, "extract", "--splits", "train", "val", *pin) == 0
    assert t3.cli(out, "final", *pin, "--ci-resamples", "10") == 0
    return path, sha
