"""Checks ``scripts/pretrain.py`` makes before any GPU time.

They live outside the training loop on purpose: they decide whether a run may
start, not how it trains, so the loop and its numerics stay untouched.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from pathlib import Path

FULL_SHA = re.compile(r"[0-9a-f]{40}")


class LaunchRefused(RuntimeError):
    """A run that must not start."""


def require_code_revision(env: Mapping[str, str], repo_head: str | None) -> str:
    """The commit the run records as ``code_revision``; refuses anything unverifiable.

    ``run.json`` takes ``code_revision`` from ``GIT_COMMIT``. Unset, it used to
    become None, and two finals with None passed the evaluation's "same
    revision" guard as equal. A short sha or a ref name cannot be compared
    either, and an export that is not the checkout's HEAD records the wrong code.
    """
    sha = env.get("GIT_COMMIT", "")
    if not sha:
        raise LaunchRefused("GIT_COMMIT no esta exportado: export GIT_COMMIT=$(git rev-parse HEAD)")
    if not FULL_SHA.fullmatch(sha):
        raise LaunchRefused(f"GIT_COMMIT={sha!r} no es un sha completo de 40 hex en minuscula")
    if repo_head is not None and sha != repo_head:
        raise LaunchRefused(f"GIT_COMMIT={sha[:12]} no es el HEAD del clon ({repo_head[:12]})")
    return sha


def repo_head(repo: Path) -> str | None:
    """HEAD of ``repo``, or None when it is not a git checkout (a copied bundle)."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
