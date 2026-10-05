"""Fail closed unless the PAD-UFES prediction was committed before the code that reads the data.

docs/30 registers the protocol and the prediction between two HTML markers.
The evaluation runs only if all of this holds, checked against git and not
against the working tree:

1. The registration commit exists and is an ancestor of HEAD.
2. The registered block at HEAD is byte-identical to the block at that commit,
   so the prediction cannot be edited after the fact and still pass.
3. The evaluation script was first added in a strict descendant of the
   registration commit, so the prediction predates the code that reads PAD.

A shallow clone that lacks the history fails the check: without history the
order cannot be shown, and "cannot show" is treated as "did not happen".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REGISTRATION_DOC = "docs/30-pad-ufes-zero-shot.md"
BLOCK_START = "<!-- registro-pad:inicio -->"
BLOCK_END = "<!-- registro-pad:fin -->"
EVAL_SCRIPT = "scripts/pad_ufes_eval.py"
#: The commit that registered docs/30 (pushed on its own, before any PAD code).
REGISTRATION_COMMIT = "0c11d722e22fe3ad333cc8e3626ac6d55efcaebb"


class RegistrationError(RuntimeError):
    """The prediction cannot be shown to precede the evaluation code."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def _show(repo: Path, commit: str, path: str) -> str:
    result = _git(repo, "show", f"{commit}:{path}")
    if result.returncode != 0:
        raise RegistrationError(f"{path} no existe en {commit[:12]}: {result.stderr.strip()}")
    return result.stdout


def registered_block(text: str, start: str = BLOCK_START, end: str = BLOCK_END) -> str:
    first, last = text.find(start), text.find(end)
    if first < 0 or last < 0 or last < first or text.count(start) != 1:
        raise RegistrationError(f"el documento no tiene un unico bloque {start} ... {end}")
    return text[first:last + len(end)]


def _is_ancestor(repo: Path, older: str, newer: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", older, newer).returncode == 0


def _first_added(repo: Path, path: str) -> str:
    result = _git(repo, "log", "--diff-filter=A", "--format=%H", "--", path)
    commits = result.stdout.split()
    if result.returncode != 0 or not commits:
        raise RegistrationError(f"no se encuentra el commit que agrego {path} (clon superficial?)")
    return commits[-1]


def check_registration(repo: Path, commit: str = REGISTRATION_COMMIT, doc: str = REGISTRATION_DOC,
                       script: str = EVAL_SCRIPT, start: str = BLOCK_START, end: str = BLOCK_END) -> str:
    """Return the full registration commit, or raise :class:`RegistrationError`.

    ``start``/``end`` select the block, so a document can hold several
    registrations (docs/30 5 guards the FST validation on its own).
    """
    resolved = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved.returncode != 0:
        raise RegistrationError(f"el commit de registro {commit!r} no esta en el repositorio")
    full = resolved.stdout.strip()
    if not _is_ancestor(repo, full, "HEAD"):
        raise RegistrationError(f"el registro {full[:12]} no es ancestro de HEAD")
    block = lambda rev: registered_block(_show(repo, rev, doc), start, end)  # noqa: E731
    if block(full) != block("HEAD"):
        raise RegistrationError(f"el bloque registrado de {doc} cambio despues de {full[:12]}")
    added = _first_added(repo, script)
    if added == full or not _is_ancestor(repo, full, added):
        raise RegistrationError(f"{script} se agrego en {added[:12]}, que no es posterior al registro {full[:12]}")
    return full
