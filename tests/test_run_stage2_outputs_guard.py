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
