"""Every thesis figure export is versioned (git, not the file system, answers): PDF + SVG + PNG per figure."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STEMS = ("fig1_ita", "fig2_models", "fig3_masks")
SUFFIXES = (".pdf", ".svg", ".png")


def _tracked(rel: str) -> bool:
    return subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0


@pytest.mark.skipif(not (REPO / ".git").exists(), reason="not a git checkout")
@pytest.mark.parametrize("stem", STEMS)
def test_each_figure_has_its_three_exports_tracked_by_git(stem):
    missing = [f"resultados/figuras/{stem}{suffix}" for suffix in SUFFIXES if not _tracked(f"resultados/figuras/{stem}{suffix}")]
    assert not missing, f"figure outputs not versioned (is an ignore rule hiding them?): {missing}"
