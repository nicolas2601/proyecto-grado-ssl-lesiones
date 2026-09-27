"""Provenance for results that must trace back to one commit.

A result records ``code_revision`` so it can be reproduced, but a revision
only describes the code that ran if the tree was clean. On 2026-09-27 the
selection ran from an untracked copy of its own script that shadowed the
commit; untracked files count as dirt here for that reason.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


class DirtyTreeError(RuntimeError):
    """The working tree does not match any commit."""


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError) as exc:
        raise DirtyTreeError(f"{repo} no es un repositorio git legible: {exc}") from exc


def clean_revision(repo: Path) -> str:
    """HEAD of ``repo``; refuses if anything, tracked or not, differs from it."""
    dirty = _git(repo, "status", "--porcelain")
    if dirty:
        raise DirtyTreeError(
            f"el arbol de {repo} tiene cambios sin commitear; commitealos o borralos:\n{dirty}"
        )
    return _git(repo, "rev-parse", "HEAD")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
