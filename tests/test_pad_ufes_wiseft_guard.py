"""scripts/pad_ufes_wiseft.py: everything that must refuse before any PAD or HAM image is read.

No backbone, no real image. The docs/30 8 block guards the script the same way
the first pass and the addendum are guarded, with its own markers.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dermssl import pad_ufes_guard as pg
from scripts import pad_ufes_wiseft as ws

REPO = Path(__file__).resolve().parents[1]


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    paths = {"--manifest": "manifest.json", "--pad-root": "pad", "--ham-root": "ham",
             "--split-index": "index.json", "--holdout": "holdout.json", "--exclude-ids-file": "exclude.txt",
             "--weights-dir": "v7", "--out": "out.json"}
    return [*(token for flag, name in paths.items() for token in (flag, str(tmp_path / name))), *extra]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _check(repo: Path, commit: str) -> str:
    return pg.check_registration(repo, commit=commit, doc=pg.REGISTRATION_DOC, script=ws.THIS_SCRIPT,
                                 start=ws.BLOCK[0], end=ws.BLOCK[1])


WISEFT = f"{ws.BLOCK[0]}\nPAD-W1 y PAD-W2\n{ws.BLOCK[1]}\n"
FIRST = f"{pg.BLOCK_START}\nprediccion\n{pg.BLOCK_END}\n"


class TestTheBlockGuardsTheScript:
    def test_the_real_block_precedes_the_real_script(self):
        shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                                 capture_output=True, text=True).stdout.strip()
        if shallow != "false":
            pytest.skip("clon superficial: el orden no se puede mostrar sin historia")
        assert ws.check_registration() == ws.REGISTRATION_COMMIT

    def test_the_pinned_commit_is_the_one_that_added_the_block(self):
        before = _git(REPO, "show", f"{ws.REGISTRATION_COMMIT}^:{pg.REGISTRATION_DOC}")
        after = _git(REPO, "show", f"{ws.REGISTRATION_COMMIT}:{pg.REGISTRATION_DOC}")
        assert ws.BLOCK[0] not in before and after.count(ws.BLOCK[0]) == 1

    def test_the_script_before_the_block_is_refused(self, tmp_path):
        _git(tmp_path, "init", "-q")
        _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST, ws.THIS_SCRIPT: "x\n"}, "script first")
        registered = _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST + WISEFT}, "block later")
        with pytest.raises(pg.RegistrationError, match="no es posterior"):
            _check(tmp_path, registered)

    def test_a_block_changed_after_registration_is_refused(self, tmp_path):
        _git(tmp_path, "init", "-q")
        registered = _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST + WISEFT}, "block")
        _commit(tmp_path, {ws.THIS_SCRIPT: "x\n"}, "script")
        _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST + WISEFT.replace("PAD-W2", "PAD-W3")}, "edit")
        with pytest.raises(pg.RegistrationError, match="cambio despues"):
            _check(tmp_path, registered)

    def test_block_then_script_passes(self, tmp_path):
        _git(tmp_path, "init", "-q")
        registered = _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST + WISEFT}, "block")
        _commit(tmp_path, {ws.THIS_SCRIPT: "x\n"}, "script")
        assert _check(tmp_path, registered) == registered


class TestArguments:
    def test_dataloader_workers_default_to_zero(self, tmp_path):
        # /dev/shm on the pod is 64 MB: a run with --num-workers 8 crashed there.
        assert ws.parse_args(_argv(tmp_path)).num_workers == 0

    def test_the_ham_curve_is_the_committed_v71_holdout_grid(self, tmp_path):
        assert ws.parse_args(_argv(tmp_path)).ham_grid == REPO / "resultados/clasificacion/v71/holdout_grid.json"

    def test_the_weights_dir_is_required(self, tmp_path):
        argv = _argv(tmp_path)
        cut = argv.index("--weights-dir")
        with pytest.raises(SystemExit):
            ws.parse_args(argv[:cut] + argv[cut + 2:])


class TestItRefusesBeforeReadingData:
    def test_without_single_evaluation(self, tmp_path):
        with pytest.raises(SystemExit, match="single-evaluation"):
            ws.main(_argv(tmp_path))

    def test_when_the_output_exists(self, tmp_path):
        (tmp_path / "out.json").write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="ya existe"):
            ws.main(_argv(tmp_path, "--single-evaluation"))

    def test_when_the_block_is_not_registered_first(self, tmp_path, monkeypatch):
        def refuse(*args, **kwargs):
            raise pg.RegistrationError("el bloque registrado cambio")
        monkeypatch.setattr(pg, "check_registration", refuse)
        with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
            ws.main(_argv(tmp_path, "--single-evaluation"))

    def test_when_the_tree_is_dirty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ws, "check_registration", lambda: "r" * 40)

        def dirty(repo):
            raise ws.rp.DirtyTreeError("arbol sucio")
        monkeypatch.setattr(ws.rp, "clean_revision", dirty)
        with pytest.raises(SystemExit, match="ABORT: arbol sucio"):
            ws.main(_argv(tmp_path, "--single-evaluation"))


class TestTheFinalsSeeds:
    def _runs(self, **over):
        runs = {"A_s42": {"seed": 42}, "A_s43": {"seed": 43}, "B_s42": {"seed": 42}, "B_s43": {"seed": 43}}
        return {**runs, **over}

    def test_each_a_final_is_of_its_own_seed(self):
        ws.check_seeds(self._runs())

    @pytest.mark.parametrize("name, run", [("A_s43", {"seed": 42}), ("A_s42", {}), ("A_s42", {"seed": 44})])
    def test_a_final_of_another_seed_is_refused(self, name, run):
        with pytest.raises(SystemExit, match=name):
            ws.check_seeds(self._runs(**{name: run}))
