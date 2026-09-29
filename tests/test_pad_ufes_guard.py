"""docs/30: the PAD-UFES evaluation refuses to run unless its prediction came first.

Each case builds a throwaway git repository. The real repository is checked
in tests/test_pad_ufes_eval.py, once the evaluation script exists.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dermssl import pad_ufes_guard as pg

DOC = "docs/30.md"
SCRIPT = "scripts/eval.py"
BLOCK = f"{pg.BLOCK_START}\nprediccion: B* no mejora\n{pg.BLOCK_END}\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
         "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    return tmp_path


def _check(repo: Path, commit: str) -> str:
    return pg.check_registration(repo, commit=commit, doc=DOC, script=SCRIPT)


def test_prediction_then_code_passes(repo):
    registered = _commit(repo, {DOC: "# 30\n" + BLOCK}, "docs: register")
    _commit(repo, {SCRIPT: "print('eval')\n"}, "feat: eval")
    _commit(repo, {DOC: "# 30\n" + BLOCK + "\n## Resultado\nlisto\n"}, "docs: result")
    assert _check(repo, registered[:10]) == registered


def test_an_edited_prediction_is_refused(repo):
    registered = _commit(repo, {DOC: BLOCK}, "docs: register")
    _commit(repo, {SCRIPT: "x\n"}, "feat: eval")
    _commit(repo, {DOC: BLOCK.replace("no mejora", "mejora")}, "docs: rewrite")
    with pytest.raises(pg.RegistrationError, match="cambio despues"):
        _check(repo, registered)


def test_code_committed_with_the_prediction_is_refused(repo):
    registered = _commit(repo, {DOC: BLOCK, SCRIPT: "x\n"}, "both at once")
    with pytest.raises(pg.RegistrationError, match="no es posterior"):
        _check(repo, registered)


def test_code_before_the_prediction_is_refused(repo):
    _commit(repo, {SCRIPT: "x\n"}, "feat: eval first")
    registered = _commit(repo, {DOC: BLOCK}, "docs: register later")
    with pytest.raises(pg.RegistrationError, match="no es posterior"):
        _check(repo, registered)


def test_a_registration_off_the_history_is_refused(repo):
    base = _commit(repo, {"a.txt": "a\n"}, "base")
    _git(repo, "checkout", "-q", "-b", "side")
    registered = _commit(repo, {DOC: BLOCK}, "docs: register on a side branch")
    _git(repo, "checkout", "-q", "-")
    _commit(repo, {DOC: BLOCK, SCRIPT: "x\n"}, "main")
    assert base
    with pytest.raises(pg.RegistrationError, match="no es ancestro"):
        _check(repo, registered)


def test_an_unknown_commit_is_refused(repo):
    _commit(repo, {DOC: BLOCK}, "docs")
    with pytest.raises(pg.RegistrationError, match="no esta en el repositorio"):
        _check(repo, "0" * 40)


@pytest.mark.parametrize("text", ["sin marcadores", pg.BLOCK_END + pg.BLOCK_START, BLOCK + BLOCK])
def test_the_block_must_be_single_and_ordered(text):
    with pytest.raises(pg.RegistrationError, match="unico bloque"):
        pg.registered_block(text)



def test_a_second_block_is_guarded_on_its_own(repo):
    """docs/30 5 registers the FST validation in its own block, with its own script."""
    start, end = "<!-- otro:inicio -->", "<!-- otro:fin -->"
    second = f"{start}\nno supera la mayoria\n{end}\n"
    _commit(repo, {DOC: BLOCK}, "docs: first block")
    registered = _commit(repo, {DOC: BLOCK + second}, "docs: second block")
    _commit(repo, {"scripts/fst.py": "x\n"}, "feat: fst")
    kwargs = dict(commit=registered, doc=DOC, script="scripts/fst.py", start=start, end=end)
    assert pg.check_registration(repo, **kwargs) == registered
    _commit(repo, {DOC: BLOCK + second.replace("no supera", "supera")}, "docs: rewrite")
    with pytest.raises(pg.RegistrationError, match="cambio despues"):
        pg.check_registration(repo, **kwargs)
