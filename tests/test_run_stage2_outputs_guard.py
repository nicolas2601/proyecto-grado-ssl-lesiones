"""A wrong --outputs must stop the run, not restart it.

Completion is decided by "does this artefact exist". Point the orchestrator
at a path that is wrong, or at a volume that did not mount, and every stage
reads as pending: it would retrain forty-eight GPU-hours over work that is
already done, and look correct doing it.

From the inside, absence and never-ran are the same observation. The state
file is what separates them, because it names stages that did finish.
"""

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_stage2  # noqa: E402


def _settings(tmp_path: Path) -> run_stage2.Settings:
    return run_stage2.Settings(
        repo=tmp_path / "repo", outputs=tmp_path / "out", python="python",
        ham_root=tmp_path / "ham",
    )


def _state(done: list[str]) -> dict:
    return {"stages": {name: {"status": "ok"} for name in done}}


def test_a_missing_outputs_directory_stops_the_run(tmp_path):
    settings = _settings(tmp_path)          # nunca creado
    with pytest.raises(SystemExit, match="no existe el directorio de salidas"):
        run_stage2.check_outputs_tree(settings, {"stages": {}})


def test_a_fresh_tree_with_no_history_is_allowed(tmp_path):
    settings = _settings(tmp_path)
    settings.outputs.mkdir(parents=True)
    run_stage2.check_outputs_tree(settings, {"stages": {}})   # no levanta


def test_finished_stages_whose_artefact_vanished_stop_the_run(tmp_path):
    """The exact shape of a wrong path: history says done, nothing is there."""
    settings = _settings(tmp_path)
    settings.outputs.mkdir(parents=True)
    with pytest.raises(SystemExit, match="su artefacto no esta"):
        run_stage2.check_outputs_tree(settings, _state(["export", "tone_pairs"]))


def test_the_message_names_the_stages_and_the_path(tmp_path):
    settings = _settings(tmp_path)
    settings.outputs.mkdir(parents=True)
    with pytest.raises(SystemExit) as excinfo:
        run_stage2.check_outputs_tree(settings, _state(["export"]))
    message = str(excinfo.value)
    assert "export" in message
    assert str(settings.outputs) in message


def test_a_consistent_tree_passes(tmp_path):
    settings = _settings(tmp_path)
    export = next(s for s in run_stage2.all_stages(settings) if s.name == "export")
    export.done_when.parent.mkdir(parents=True)
    export.done_when.write_text("{}", encoding="utf-8")
    run_stage2.check_outputs_tree(settings, _state(["export"]))   # no levanta


def test_a_stage_that_failed_before_is_not_required_to_have_an_artefact(tmp_path):
    settings = _settings(tmp_path)
    settings.outputs.mkdir(parents=True)
    state = {"stages": {"export": {"status": "error"}}}
    run_stage2.check_outputs_tree(settings, state)   # no levanta


def test_run_refuses_before_launching_anything(tmp_path, monkeypatch):
    """The guard must fire before a single GPU-hour is spent."""
    settings = _settings(tmp_path)
    settings.outputs.mkdir(parents=True)
    (settings.outputs / run_stage2.STATE_FILE).write_text(
        json.dumps(_state(["export"])), encoding="utf-8"
    )
    launched = []
    monkeypatch.setattr(run_stage2, "_launch", lambda *a, **k: launched.append(a))

    with pytest.raises(SystemExit):
        run_stage2.run(settings)
    assert launched == [], "se lanzo una etapa pese al arbol inconsistente"


# ------------------------------------------------- existence is not correspondence
def test_a_subdirectory_of_the_real_root_is_refused(tmp_path):
    """The mistake that cost the morning: --outputs .../salidas/tone.

    That directory exists, so an existence check waves it through. Every
    path then becomes salidas/tone/tone/..., every stage reads as pending,
    and the run retrains twenty-four GPU-hours looking completely normal.
    """
    settings = _settings(tmp_path)
    tone = settings.outputs / "tone"
    for child in ("inputs", "shift2", "ssl_plain_vits14"):
        (tone / child).mkdir(parents=True)

    wrong = run_stage2.Settings(
        repo=settings.repo, outputs=tone, python="python", ham_root=settings.ham_root,
    )
    with pytest.raises(SystemExit, match="no parece la raiz"):
        run_stage2.check_outputs_tree(wrong, {"stages": {}})


def test_the_real_root_is_accepted(tmp_path):
    settings = _settings(tmp_path)
    (settings.outputs / "tone").mkdir(parents=True)
    run_stage2.check_outputs_tree(settings, {"stages": {}})   # no levanta


def test_an_empty_directory_is_accepted_as_a_fresh_run(tmp_path):
    settings = _settings(tmp_path)
    settings.outputs.mkdir(parents=True)
    run_stage2.check_outputs_tree(settings, {"stages": {}})   # no levanta


def test_a_root_with_only_the_state_file_is_accepted(tmp_path):
    settings = _settings(tmp_path)
    settings.outputs.mkdir(parents=True)
    (settings.outputs / run_stage2.STATE_FILE).write_text("{}", encoding="utf-8")
    run_stage2.check_outputs_tree(settings, {"stages": {}})


def test_the_message_names_what_it_found_instead(tmp_path):
    settings = _settings(tmp_path)
    (settings.outputs / "inputs").mkdir(parents=True)
    with pytest.raises(SystemExit) as excinfo:
        run_stage2.check_outputs_tree(settings, {"stages": {}})
    assert "inputs" in str(excinfo.value), "debe decir que encontro, no solo que falta"


# ------------------------------------------------- the other diagonal
def test_a_running_entry_with_no_artefact_is_called_out(tmp_path, capsys):
    """An orchestrator that died mid-stage leaves `running` forever.

    Nothing rewrites that entry: the process that would have closed it is
    gone. The run continues - the artefact decides completion, not this
    file - but the history says a stage is in flight that nobody is running.
    """
    settings = _settings(tmp_path)
    (settings.outputs / "tone").mkdir(parents=True)
    state = {"stages": {"ssl_plain": {"status": "running", "started_at": "2026-09-21T01:08:08"}}}

    run_stage2.check_outputs_tree(settings, state)

    printed = capsys.readouterr().out
    assert "ADVERTENCIA" in printed
    assert "2026-09-21T01:08:08" in printed
    assert "murio a mitad" in printed


def test_a_running_entry_whose_artefact_exists_is_reported_differently(tmp_path, capsys):
    """Finished but never closed: the same lie, the opposite cause."""
    settings = _settings(tmp_path)
    arm = next(s for s in run_stage2.all_stages(settings) if s.name == "ssl_proposed")
    arm.done_when.parent.mkdir(parents=True)
    arm.done_when.write_bytes(b"weights")
    state = {"stages": {"ssl_proposed": {"status": "running", "started_at": "07:26:09"}}}

    run_stage2.check_outputs_tree(settings, state)

    printed = capsys.readouterr().out
    assert "su artefacto existe" in printed
    assert "ADVERTENCIA" not in printed, "no es una alarma: termino, solo no se cerro"


def test_a_clean_state_says_nothing(tmp_path, capsys):
    settings = _settings(tmp_path)
    (settings.outputs / "tone").mkdir(parents=True)
    run_stage2.check_outputs_tree(settings, {"stages": {}})
    assert capsys.readouterr().out == ""


def test_the_warning_does_not_stop_the_run(tmp_path):
    """The artefact decides completion; a stale record is noise, not a blocker."""
    settings = _settings(tmp_path)
    (settings.outputs / "tone").mkdir(parents=True)
    state = {"stages": {"ssl_plain": {"status": "running", "started_at": "2026-09-21T01:08:08"}}}
    run_stage2.check_outputs_tree(settings, state)   # no levanta


def test_a_recent_running_entry_is_left_alone(tmp_path, capsys):
    """A live concurrent run must not be called a corpse."""
    settings = _settings(tmp_path)
    (settings.outputs / "tone").mkdir(parents=True)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_stage2.check_outputs_tree(
        settings, {"stages": {"ssl_plain": {"status": "running", "started_at": now}}}
    )
    assert "ADVERTENCIA" not in capsys.readouterr().out


def test_an_unreadable_timestamp_does_not_silence_the_warning(tmp_path, capsys):
    """Unknown age is not young."""
    settings = _settings(tmp_path)
    (settings.outputs / "tone").mkdir(parents=True)
    run_stage2.check_outputs_tree(
        settings, {"stages": {"ssl_plain": {"status": "running", "started_at": "ayer"}}}
    )
    assert "ADVERTENCIA" in capsys.readouterr().out
