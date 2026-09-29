"""scripts/pad_ufes_addendum.py: everything that must refuse before any PAD data is read.

No backbone, no real image. The addendum block of docs/30 7 guards the script
the same way docs/30 2-3 guards the first pass, with its own markers.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dermssl import pad_ufes_addendum as pa
from dermssl import pad_ufes_guard as pg
from scripts import pad_ufes_addendum as ad

REPO = Path(__file__).resolve().parents[1]
SELECTION = REPO / "resultados" / "clasificacion" / "v71" / "selection.json"


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    paths = {"--manifest": "manifest.json", "--pad-root": "pad", "--ham-root": "ham",
             "--split-index": "index.json", "--holdout": "holdout.json", "--exclude-ids-file": "exclude.txt",
             "--weights-dir": "v7", "--v71-inputs": "inputs", "--pad-cache": "pad_cache", "--out": "out.json"}
    argv = [token for flag, name in paths.items() for token in (flag, str(tmp_path / name))]
    return [*argv, "--plain-s44", "dinov2_vitb14:a44", "--proposed-s44", "dinov2_vitb14:b44", *extra]


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
    return pg.check_registration(repo, commit=commit, doc=pg.REGISTRATION_DOC, script=ad.THIS_SCRIPT,
                                 start=ad.BLOCK[0], end=ad.BLOCK[1])


ADDENDUM = f"{ad.BLOCK[0]}\nA71 y s44, descriptivo\n{ad.BLOCK[1]}\n"
FIRST = f"{pg.BLOCK_START}\nprediccion\n{pg.BLOCK_END}\n"


class TestTheAddendumBlockGuardsTheScript:
    def test_the_real_addendum_precedes_the_real_script(self):
        shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"],
                                 capture_output=True, text=True).stdout.strip()
        if shallow != "false":
            pytest.skip("clon superficial: el orden no se puede mostrar sin historia")
        assert ad.check_registration() == ad.REGISTRATION_COMMIT

    def test_the_pinned_commit_is_the_one_that_added_the_block(self):
        before = _git(REPO, "show", f"{ad.REGISTRATION_COMMIT}^:{pg.REGISTRATION_DOC}")
        after = _git(REPO, "show", f"{ad.REGISTRATION_COMMIT}:{pg.REGISTRATION_DOC}")
        assert ad.BLOCK[0] not in before and after.count(ad.BLOCK[0]) == 1

    def test_the_script_before_the_addendum_is_refused(self, tmp_path):
        _git(tmp_path, "init", "-q")
        _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST, ad.THIS_SCRIPT: "x\n"}, "script first")
        registered = _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST + ADDENDUM}, "addendum later")
        with pytest.raises(pg.RegistrationError, match="no es posterior"):
            _check(tmp_path, registered)

    def test_an_addendum_changed_after_registration_is_refused(self, tmp_path):
        _git(tmp_path, "init", "-q")
        registered = _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST + ADDENDUM}, "addendum")
        _commit(tmp_path, {ad.THIS_SCRIPT: "x\n"}, "script")
        _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST + ADDENDUM.replace("descriptivo", "con veredicto")}, "edit")
        with pytest.raises(pg.RegistrationError, match="cambio despues"):
            _check(tmp_path, registered)

    def test_addendum_then_script_passes_even_if_the_first_block_is_elsewhere(self, tmp_path):
        _git(tmp_path, "init", "-q")
        registered = _commit(tmp_path, {pg.REGISTRATION_DOC: FIRST + ADDENDUM}, "addendum")
        _commit(tmp_path, {ad.THIS_SCRIPT: "x\n"}, "script")
        assert _check(tmp_path, registered) == registered


class TestArguments:
    def test_the_s44_finals_are_required(self, tmp_path):
        with pytest.raises(SystemExit):
            ad.parse_args(_argv(tmp_path)[:-2])

    def test_public_and_the_frozen_selection_are_the_defaults(self, tmp_path):
        args = ad.parse_args(_argv(tmp_path))
        assert args.public == "dinov2_vitb14:public"
        assert args.selection == SELECTION


class TestItRefusesBeforeReadingData:
    def test_without_single_evaluation(self, tmp_path):
        with pytest.raises(SystemExit, match="single-evaluation"):
            ad.main(_argv(tmp_path))

    def test_when_the_output_exists(self, tmp_path):
        (tmp_path / "out.json").write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="ya existe"):
            ad.main(_argv(tmp_path, "--single-evaluation"))

    def test_when_the_addendum_is_not_registered_first(self, tmp_path, monkeypatch):
        def refuse(*args, **kwargs):
            raise pg.RegistrationError("el bloque registrado cambio")
        monkeypatch.setattr(pg, "check_registration", refuse)
        with pytest.raises(SystemExit, match="ABORT: el bloque registrado cambio"):
            ad.main(_argv(tmp_path, "--single-evaluation"))

    def test_when_the_tree_is_dirty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ad, "check_registration", lambda: "r" * 40)

        def dirty(repo):
            raise ad.rp.DirtyTreeError("arbol sucio")
        monkeypatch.setattr(ad.rp, "clean_revision", dirty)
        with pytest.raises(SystemExit, match="ABORT: arbol sucio"):
            ad.main(_argv(tmp_path, "--single-evaluation"))


class TestTheFrozenSelection:
    def test_the_committed_selection_names_a71(self):
        assert ad.check_selection(SELECTION) == {pa.A71: pa.A71_CONFIG, pa.PUBLIC_A71: pa.PUBLIC_A71_CONFIG}

    def test_a_selection_with_other_bytes_is_refused(self, tmp_path):
        copy = tmp_path / "selection.json"
        shutil.copy(SELECTION, copy)
        copy.write_text(copy.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with pytest.raises(SystemExit, match="sha256"):
            ad.check_selection(copy)

    def test_a_selection_naming_another_winner_is_refused(self, tmp_path, monkeypatch):
        doc = json.loads(SELECTION.read_text(encoding="utf-8"))
        doc["winners"]["A"] = "A_s42_a1.0@224|cls|none|id"
        other = tmp_path / "selection.json"
        other.write_text(json.dumps(doc), encoding="utf-8")
        monkeypatch.setattr(pa, "SELECTION_SHA256", ad.rp.file_sha256(other))
        with pytest.raises(SystemExit, match="A71"):
            ad.check_selection(other)


class TestTheV71TrainInputs:
    def test_the_cached_train_array_of_v71_is_required(self, tmp_path):
        with pytest.raises(SystemExit, match="train_224_"):
            ad.train_inputs_path(tmp_path, ["ISIC_1", "ISIC_2"])

    def test_it_is_the_name_v71_wrote(self, tmp_path):
        from dermssl import feature_cache as fc

        ids = ["ISIC_1", "ISIC_2"]
        path = tmp_path / f"train_224_{fc.fingerprint_ids(ids)}.npy"
        path.write_bytes(b"x")
        assert ad.train_inputs_path(tmp_path, ids) == path


class TestTheS44Finals:
    def _runs(self, **over):
        base = {"code_revision": "c" * 40, "exclude_ids_sha256": "e" * 64, "seed": 44}
        return {"plain_s44": {**base, **over}, "proposed_s44": dict(base)}

    def test_consistent_s44_finals_pass(self):
        ad.check_s44_runs(self._runs(), "e" * 64)

    @pytest.mark.parametrize("over, match", [({"seed": 43}, "semilla"), ({"code_revision": "d" * 40}, "code_revision"),
                                             ({"exclude_ids_sha256": "f" * 64}, "exclude")])
    def test_a_mismatch_is_refused(self, over, match):
        with pytest.raises(SystemExit, match=match):
            ad.check_s44_runs(self._runs(**over), "e" * 64)

    def test_both_on_another_seed_are_refused(self):
        runs = {role: {**run, "seed": 43} for role, run in self._runs().items()}
        with pytest.raises(SystemExit, match="44"):
            ad.check_s44_runs(runs, "e" * 64)


def test_the_context_namespace_reuses_the_v71_loader_arguments(tmp_path):
    args = ad.parse_args(_argv(tmp_path))
    ns = ad.context_args(args)
    assert isinstance(ns, argparse.Namespace)
    assert (ns.image_dir, ns.limit) == (args.ham_root, None)
