"""Tests for the optional Weights & Biases sink.

The sink must be a silent no-op unless a project is given AND credentials are
present, so training on a machine without W&B never fails and never blocks.
"""

import json

import numpy as np
import pytest

import seg_paths  # noqa: F401
import wandb_logging as wl


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch, tmp_path):
    for name in ("WANDB_API_KEY", "WANDB_MODE", "WANDB_ENTITY", "WANDB_RUN_GROUP", "WANDB_PROJECT", "WANDB_NETRC_FILE"):
        monkeypatch.delenv(name, raising=False)
    # An isolated home so a developer .netrc never decides the outcome of a test.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _sink(**kwargs):
    return wl.WandbSink(project="p", run_name="r", config={"lr": 1e-3}, **kwargs)


# -------------------------------------------------------------- no-op path
def test_sink_is_disabled_without_credentials():
    assert _sink().enabled is False


def test_sink_is_disabled_without_a_project(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    assert wl.WandbSink(project="", run_name="r", config={}).enabled is False


def test_disabled_sink_accepts_every_call_without_raising():
    sink = _sink()
    sink.log({"epoch": 0, "loss": 1.0})
    sink.log_images("overlays", [np.zeros((2, 2, 3), np.uint8)], ["a"])
    sink.log_table("strata", [{"bucket": "1", "dice_mean": 0.5}])
    sink.log_histogram("dice", [0.1, 0.2])
    sink.finish()


def test_sink_is_disabled_when_the_wandb_import_fails(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setattr(wl, "_import_wandb", lambda: None)
    assert _sink().enabled is False


def test_credentials_accept_offline_mode_without_a_key(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert wl.credentials_available() is True


def test_credentials_require_a_key_or_offline_mode():
    assert wl.credentials_available() is False


def test_step_comes_from_the_epoch_then_the_step_field():
    assert wl.record_step({"epoch": 3, "step": 9}) == 3
    assert wl.record_step({"step": 9}) == 9
    assert wl.record_step({"loss": 1.0}) is None


def test_building_a_disabled_sink_never_prints_the_api_key(monkeypatch, capsys):
    monkeypatch.setenv("WANDB_API_KEY", "super-secret-value")
    monkeypatch.setattr(wl, "_import_wandb", lambda: None)
    sink = _sink()
    sink.log({"epoch": 0})
    sink.finish()
    assert "super-secret-value" not in capsys.readouterr().out


# --------------------------------------------------------------- real path
@pytest.mark.filterwarnings("ignore")
def test_offline_run_writes_a_local_directory(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    sink = wl.WandbSink(project="seg-test", run_name="unit", config={"lr": 1e-3}, tags=("test",))
    assert sink.enabled is True
    sink.log({"epoch": 0, "loss": 1.0})
    sink.log_table("strata", [{"bucket": "1", "dice_mean": 0.5}])
    sink.log_histogram("dice", [0.1, 0.9])
    sink.log_images("overlays", [np.zeros((4, 4, 3), np.uint8)], ["worst"])
    sink.finish()
    assert any(tmp_path.rglob("*.wandb"))


# ------------------------------------------------------- training wiring
def _args(**kwargs):
    import argparse

    base = {"wandb_project": "p", "wandb_run_name": "", "output_dir": "salidas/student_r1"}
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_default_run_name_joins_the_stage_and_the_output_folder():
    assert wl.default_run_name(_args(), "student") == "student-student_r1"


def test_an_explicit_run_name_wins_over_the_default():
    assert wl.default_run_name(_args(wandb_run_name="mine"), "student") == "mine"


def test_sink_from_args_is_disabled_on_secondary_ranks(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    monkeypatch.setattr(wl, "_import_wandb", lambda: None)
    assert wl.sink_from_args(_args(), "moco", is_main=False).enabled is False


def test_sink_from_args_returns_a_usable_no_op_without_credentials():
    sink = wl.sink_from_args(_args(), "moco", is_main=True)
    assert sink.enabled is False
    sink.log({"epoch": 0})
    sink.finish()


def test_tracking_arguments_are_added_with_an_empty_default():
    import argparse

    parser = argparse.ArgumentParser()
    wl.add_tracking_arguments(parser)
    args = parser.parse_args([])
    assert args.wandb_project == "" and args.wandb_run_name == ""


def test_every_training_entry_point_exposes_the_tracking_flags():
    import train_ccam
    import train_moco
    import train_student

    moco = train_moco.parse_args(["--cache-dir", "c", "--output-dir", "o"])
    ccam = train_ccam.parse_args(["--cache-dir", "c", "--backbone", "b", "--output-dir", "o"])
    student = train_student.parse_args(
        ["--cache-dir", "c", "--pseudo-masks", "p.npy", "--output-dir", "o"]
    )
    for args in (moco, ccam, student):
        assert args.wandb_project == "" and args.wandb_run_name == ""


# ------------------------------------------------- failures never kill training
class _FakeRun:
    def __init__(self, fail_log=False, fail_finish=False):
        self.fail_log = fail_log
        self.fail_finish = fail_finish
        self.logged = []
        self.finished = False

    def log(self, payload, step=None):
        if self.fail_log:
            raise RuntimeError("network down")
        self.logged.append((payload, step))

    def finish(self):
        if self.fail_finish:
            raise RuntimeError("flush failed")
        self.finished = True


class _FakeClient:
    def __init__(self, fail_init=False, fail_log=False, fail_finish=False):
        self.fail_init = fail_init
        self.fail_log = fail_log
        self.fail_finish = fail_finish
        self.init_kwargs = None
        self.run = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        if self.fail_init:
            raise RuntimeError("quota exceeded for key super-secret-value")
        self.run = _FakeRun(self.fail_log, self.fail_finish)
        return self.run

    def Image(self, item, caption=None):  # noqa: N802 - mirrors the client API
        return ("image", item, caption)

    def Table(self, columns, data):  # noqa: N802 - mirrors the client API
        return ("table", columns, data)

    def Histogram(self, values):  # noqa: N802 - mirrors the client API
        return ("histogram", values)


@pytest.fixture
def fake_client(monkeypatch):
    def _install(**kwargs):
        client = _FakeClient(**kwargs)
        monkeypatch.setenv("WANDB_API_KEY", "super-secret-value")
        monkeypatch.setattr(wl, "_import_wandb", lambda: client)
        return client

    return _install


def test_a_failing_init_disables_the_sink_instead_of_raising(fake_client, capsys):
    fake_client(fail_init=True)
    sink = _sink()
    assert sink.enabled is False
    assert "super-secret-value" not in capsys.readouterr().out


def test_a_failing_log_disables_the_sink_and_later_calls_are_no_ops(fake_client):
    client = fake_client(fail_log=True)
    sink = _sink()
    assert sink.enabled is True
    sink.log({"epoch": 0, "loss": 1.0})
    assert sink.enabled is False
    sink.log({"epoch": 1})
    sink.log_table("strata", [{"bucket": "1"}])
    sink.log_histogram("dice", [0.5])
    sink.log_images("overlays", [np.zeros((2, 2, 3), np.uint8)], ["a"])
    sink.finish()
    assert client.run.logged == []


def test_a_failing_finish_is_swallowed(fake_client):
    fake_client(fail_finish=True)
    sink = _sink()
    sink.finish()
    assert sink.enabled is False


def test_the_jsonl_line_survives_a_sink_that_raises(tmp_path, fake_client):
    from run_logging import JsonlLogger

    fake_client(fail_log=True)
    path = tmp_path / "metrics.jsonl"
    JsonlLogger(path, echo=False, sinks=(_sink(),)).log({"epoch": 3, "loss": 0.5})
    assert json.loads(path.read_text(encoding="utf-8"))["epoch"] == 3


def test_a_broken_sink_never_stops_the_jsonl_file(tmp_path):
    from run_logging import JsonlLogger

    class _Exploding:
        def log(self, record):
            raise RuntimeError("boom")

    path = tmp_path / "metrics.jsonl"
    JsonlLogger(path, echo=False, sinks=(_Exploding(),)).log({"epoch": 1})
    assert json.loads(path.read_text(encoding="utf-8"))["epoch"] == 1


# ------------------------------------------------------------- resumed runs
def test_the_run_id_is_stable_for_the_same_group_and_name():
    assert wl.run_id("piloto", "student-r1") == wl.run_id("piloto", "student-r1")
    assert wl.run_id("piloto", "student-r1") != wl.run_id("piloto", "student-r2")
    assert wl.run_id(None, "x") != wl.run_id("piloto", "x")


def test_the_run_id_is_twelve_lowercase_alphanumeric_characters():
    identifier = wl.run_id("piloto", "student-r1")
    assert len(identifier) == 12 and identifier.isalnum() and identifier.islower()


def test_a_resumed_run_reuses_the_same_identifier(fake_client, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_GROUP", "piloto")
    client = fake_client()
    _sink()
    assert client.init_kwargs["resume"] == "allow"
    assert client.init_kwargs["id"] == wl.run_id("piloto", "r")


# ------------------------------------------------------ captions and netrc
def test_extra_captions_are_truncated_instead_of_raising(fake_client):
    client = fake_client()
    sink = _sink()
    images = [np.zeros((2, 2, 3), np.uint8)]
    sink.log_images("overlays", images, ["one", "two", "three"])
    assert sink.enabled is True
    payload = client.run.logged[0][0]["overlays"]
    assert payload == [("image", images[0], "one")]


def test_missing_captions_are_padded_with_none(fake_client):
    client = fake_client()
    sink = _sink()
    images = [np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)]
    sink.log_images("overlays", images, ["only"])
    assert [caption for _, _, caption in client.run.logged[0][0]["overlays"]] == ["only", None]


def _write_netrc(home, machine="api.wandb.ai", password="netrc-secret"):
    path = home / ".netrc"
    path.write_text(f"machine {machine}\n  login user\n  password {password}\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def test_credentials_come_from_the_netrc_file(_clean_environment):
    _write_netrc(_clean_environment)
    assert wl.credentials_available() is True


def test_a_netrc_for_another_host_does_not_enable_tracking(_clean_environment):
    _write_netrc(_clean_environment, machine="api.github.com")
    assert wl.credentials_available() is False


def test_a_malformed_netrc_is_ignored_instead_of_raising(_clean_environment):
    (_clean_environment / ".netrc").write_text("this is not a netrc\n", encoding="utf-8")
    assert wl.credentials_available() is False


def test_a_requested_project_without_credentials_warns_once(capsys):
    sink = _sink()
    assert sink.enabled is False
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert len(lines) == 1 and "W&B" in lines[0]


def test_no_warning_when_tracking_was_never_requested(capsys):
    wl.WandbSink(project="", run_name="r", config={})
    assert capsys.readouterr().out == ""


def test_secondary_ranks_stay_silent(capsys):
    wl.sink_from_args(_args(), "moco", is_main=False)
    assert capsys.readouterr().out == ""


def test_custom_netrc_path_is_read_without_changing_environment(tmp_path, monkeypatch):
    path = _write_netrc(tmp_path)
    before = path.read_bytes()
    monkeypatch.setenv("WANDB_NETRC_FILE", str(path))
    assert wl.credentials_available()
    assert "WANDB_API_KEY" not in wl.os.environ
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["missing", "malformed", "wrong_host", "public"])
def test_invalid_explicit_file_does_not_fall_back_to_home(kind, tmp_path, monkeypatch, _clean_environment):
    _write_netrc(_clean_environment)
    path = tmp_path / "selected.netrc"
    if kind == "malformed":
        path.write_text("not a netrc\n")
        path.chmod(0o600)
    elif kind in ("wrong_host", "public"):
        path = _write_netrc(tmp_path, machine="other.example" if kind == "wrong_host" else "api.wandb.ai")
        if kind == "public":
            if wl.os.name != "posix": pytest.skip("POSIX permission check")
            path.chmod(0o644)
    monkeypatch.setenv("WANDB_NETRC_FILE", str(path))
    assert not wl.credentials_available()


@pytest.mark.parametrize("fail_init", [False, True])
def test_file_key_reaches_sdk_and_environment_is_restored(fail_init, tmp_path, monkeypatch, capsys):
    path = _write_netrc(tmp_path, password="super-secret-value")
    before = path.read_bytes()
    monkeypatch.setenv("WANDB_NETRC_FILE", str(path))
    client = _FakeClient(fail_init=fail_init)
    real_init = client.init
    observed = []
    def init(**kwargs):
        observed.append(wl.os.environ.get("WANDB_API_KEY"))
        return real_init(**kwargs)
    client.init = init
    monkeypatch.setattr(wl, "_import_wandb", lambda: client)
    sink = wl.WandbSink("p", "r", config={"nested": ["super-secret-value"]})
    assert sink.enabled is not fail_init
    assert observed == ["super-secret-value"]
    assert "WANDB_API_KEY" not in wl.os.environ
    assert client.init_kwargs["config"] == {"nested": ["***"]}
    assert path.read_bytes() == before
    assert "super-secret-value" not in capsys.readouterr().out


def test_environment_key_wins_over_file_and_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_NETRC_FILE", str(_write_netrc(tmp_path)))
    monkeypatch.setenv("WANDB_API_KEY", "environment-key")
    client = _FakeClient()
    def init(**kwargs):
        assert wl.os.environ["WANDB_API_KEY"] == "environment-key"
        return _FakeRun()
    client.init = init
    monkeypatch.setattr(wl, "_import_wandb", lambda: client)
    assert _sink().enabled
    assert wl.os.environ["WANDB_API_KEY"] == "environment-key"


@pytest.mark.parametrize("operation", ["log", "finish"])
def test_file_key_is_scrubbed_after_initialization(operation, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WANDB_NETRC_FILE", str(_write_netrc(tmp_path)))
    client = _FakeClient()
    monkeypatch.setattr(wl, "_import_wandb", lambda: client)
    sink = _sink()
    def fail(*args, **kwargs):
        raise RuntimeError("failure containing netrc-secret")
    setattr(client.run, operation, fail)
    if operation == "log": sink.log({"epoch": 1})
    else: sink.finish()
    assert not sink.enabled
    assert "netrc-secret" not in capsys.readouterr().out
    assert "WANDB_API_KEY" not in wl.os.environ


def test_file_key_is_removed_from_metric_records(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_NETRC_FILE", str(_write_netrc(tmp_path)))
    client = _FakeClient()
    monkeypatch.setattr(wl, "_import_wandb", lambda: client)
    sink = _sink()
    sink.log({"epoch": 1, "note": "value netrc-secret"})
    assert client.run.logged == [({"epoch": 1, "note": "value ***"}, 1)]

