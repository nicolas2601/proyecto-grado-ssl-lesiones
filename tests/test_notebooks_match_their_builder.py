"""Every generated notebook must be what its builder produces today.

The repo treats notebooks as generated artefacts: "no editar notebooks a mano:
editar el builder y regenerar" (notebooks/finetune/README.md). Nothing enforced
it. On 2026-09-11 the fine-tuning builder gained the per-epoch history that
makes a truncated run visible, two banks were regenerated and a third was not,
so eight committed notebooks silently kept producing results with no
convergence evidence - the exact gap the change existed to close.

Running the builder is deterministic, so the check is simply: regenerate into a
scratch tree and diff.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDERS = {
    "build_finetune_tests.py": ("finetune", "finetune_isic2019", "finetune_isic2020"),
    "build_baseline_tests.py": ("baselines", "baselines_isic2019", "baselines_isic2020"),
}


def _cells(path: Path) -> list[str]:
    """Source of every cell, ignoring outputs and execution counts."""
    nb = json.loads(path.read_text())
    return ["".join(c.get("source", [])) for c in nb.get("cells", [])]


@pytest.mark.parametrize("builder", sorted(BUILDERS))
def test_the_committed_notebooks_are_what_the_builder_writes(builder, tmp_path):
    script = REPO / "scripts" / builder
    if not script.exists():
        pytest.skip(f"{builder} not present")

    work = tmp_path / "repo"
    work.mkdir()
    (work / "scripts").mkdir()
    shutil.copy2(script, work / "scripts" / builder)
    for helper in ("build_baseline_tests.py",):
        src = REPO / "scripts" / helper
        if src.exists() and helper != builder:
            shutil.copy2(src, work / "scripts" / helper)
    shutil.copytree(REPO / "src", work / "src")

    run = subprocess.run([sys.executable, str(work / "scripts" / builder)],
                         capture_output=True, text=True, cwd=work)
    assert run.returncode == 0, run.stderr[-2000:]

    stale = []
    for bank in BUILDERS[builder]:
        fresh_bank = work / "notebooks" / bank
        if not fresh_bank.is_dir():
            continue
        for fresh in sorted(fresh_bank.rglob("*.ipynb")):
            committed = REPO / fresh.relative_to(work)
            if not committed.exists():
                stale.append(f"{fresh.relative_to(work)} is missing from the repo")
            elif _cells(committed) != _cells(fresh):
                stale.append(str(fresh.relative_to(work)))
    assert not stale, (
        f"these notebooks differ from what {builder} writes - regenerate and "
        f"commit them: {stale}")
