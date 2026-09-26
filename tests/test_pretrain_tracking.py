"""The tracker must degrade loudly, never crash a run, and log through the
mlflow module it is handed - which is how we test it without mlflow."""


import re
import sys

import pytest

from dermssl.pretrain import tracking as tr


class FakeMlflow:
    """Records every call; mimics the tiny slice of mlflow's API we use."""

    def __init__(self):
        self.calls = []

    def set_tracking_uri(self, uri):
        self.calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name):
        self.calls.append(("set_experiment", name))

    def start_run(self, run_name):
        self.calls.append(("start_run", run_name))

    def log_params(self, params):
        self.calls.append(("log_params", params))

    def log_metric(self, key, value, step=None):
        self.calls.append(("log_metric", key, value, step))

    def log_artifact(self, path):
        self.calls.append(("log_artifact", path))

    def end_run(self):
        self.calls.append(("end_run",))


def test_run_name_is_deterministic_and_readable():
    assert tr.run_name("simsiam", "resnet50", 42) == "simsiam-resnet50-s42"


def test_mlflow_tracker_delegates_in_order():
    fake = FakeMlflow()
    tracker = tr.MlflowTracker(
        fake, experiment="exp", tracking_uri="file:/tmp/mlruns", run="r1"
    )
    tracker.start({"lr": 0.05})
    tracker.log_metric("loss", 1.5, step=10)
    tracker.log_artifact("/tmp/ckpt.pt")
    tracker.end()

    names = [c[0] for c in fake.calls]
    assert names == [
        "set_tracking_uri",
        "set_experiment",
        "start_run",
        "log_params",
        "log_metric",
        "log_artifact",
        "end_run",
    ]
    assert ("log_metric", "loss", 1.5, 10) in fake.calls


def test_empty_tracking_uri_keeps_mlflow_defaults():
    fake = FakeMlflow()
    tracker = tr.MlflowTracker(fake, experiment="exp", tracking_uri="", run="r1")
    tracker.start({})
    assert not any(c[0] == "set_tracking_uri" for c in fake.calls)


def test_null_tracker_accepts_everything_silently():
    tracker = tr.NullTracker()
    tracker.start({"a": 1})
    tracker.log_metric("loss", 1.0, step=1)
    tracker.log_artifact("x")
    tracker.end()  # nothing raised is the assertion


def test_build_tracker_falls_back_to_null_when_mlflow_missing():
    tracker = tr.build_tracker(
        None, experiment="exp", tracking_uri="", run="r1", rank=0
    )
    assert isinstance(tracker, tr.NullTracker)


def test_build_tracker_is_null_on_nonzero_ranks():
    """Only rank 0 logs under DDP - otherwise N GPUs create N duplicate runs."""
    fake = FakeMlflow()
    tracker = tr.build_tracker(
        fake, experiment="exp", tracking_uri="", run="r1", rank=3
    )
    assert isinstance(tracker, tr.NullTracker)


def test_build_tracker_uses_mlflow_on_rank_zero():
    fake = FakeMlflow()
    tracker = tr.build_tracker(
        fake, experiment="exp", tracking_uri="", run="r1", rank=0
    )
    assert isinstance(tracker, tr.MlflowTracker)


def test_file_store_optin_env_is_set_for_file_uri(monkeypatch):
    """MLflow 3.x puts the filesystem backend behind MLFLOW_ALLOW_FILE_STORE;
    without it, set_experiment raises on the cluster. The tracker must opt in
    when (and only when) the URI is a file store."""
    import os
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    fake = FakeMlflow()
    tr.MlflowTracker(fake, "exp", "file:/scratch/mlruns", "run").start({})
    assert os.environ.get("MLFLOW_ALLOW_FILE_STORE") == "true"

    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    tr.MlflowTracker(fake, "exp", "http://server:5000", "run").start({})
    assert "MLFLOW_ALLOW_FILE_STORE" not in os.environ


# -------------------------------------------------- Weights & Biases sink
class FakeWandbRun:
    def __init__(self, log):
        self.log_calls = log
        self.config = {}

    def log(self, payload, step=None):
        self.log_calls.append(("log", payload, step))


class FakeWandb:
    """Records the init/log/save/finish slice of the wandb API."""

    def __init__(self):
        self.calls = []
        self.run = None

    def init(self, **kwargs):
        self.calls.append(("init", kwargs))
        self.run = FakeWandbRun(self.calls)
        return self.run

    def save(self, path, base_path=None):
        self.calls.append(("save", path))

    def finish(self):
        self.calls.append(("finish",))


def test_wandb_tracker_delegates_in_order():
    fake = FakeWandb()
    tracker = tr.WandbTracker(fake, project="dermssl-pretrain", run="r1", group="etapa2")
    tracker.start({"lr": 0.05})
    tracker.log_metric("loss", 1.5, step=10)
    tracker.log_artifact("/tmp/ckpt.pt")
    tracker.end()

    names = [c[0] for c in fake.calls]
    assert names == ["init", "log", "save", "finish"]
    init_kwargs = fake.calls[0][1]
    assert init_kwargs["project"] == "dermssl-pretrain"
    assert init_kwargs["name"] == "r1"
    assert init_kwargs["group"] == "etapa2"
    assert init_kwargs["config"] == {"lr": 0.05}
    assert ("log", {"loss": 1.5}, 10) in fake.calls


def test_wandb_tracker_never_raises_when_the_sink_fails():
    class Broken:
        def init(self, **kwargs):
            raise RuntimeError("no network")

    tracker = tr.WandbTracker(Broken(), project="p", run="r", group="")
    tracker.start({})
    tracker.log_metric("loss", 1.0, step=1)
    tracker.log_artifact("/tmp/x")
    tracker.end()


def test_composite_tracker_fans_out_to_every_sink():
    mlflow, wandb = FakeMlflow(), FakeWandb()
    tracker = tr.CompositeTracker([
        tr.MlflowTracker(mlflow, experiment="e", tracking_uri="", run="r"),
        tr.WandbTracker(wandb, project="p", run="r", group=""),
    ])
    tracker.start({"a": 1})
    tracker.log_metric("loss", 2.0, step=3)
    tracker.end()
    assert ("log_metric", "loss", 2.0, 3) in mlflow.calls
    assert ("log", {"loss": 2.0}, 3) in wandb.calls
    assert ("finish",) in wandb.calls


def test_build_tracker_adds_wandb_only_with_a_project():
    mlflow, wandb = FakeMlflow(), FakeWandb()
    only_mlflow = tr.build_tracker(mlflow, experiment="e", tracking_uri="", run="r", rank=0,
                                   wandb_module=wandb, wandb_project="")
    assert isinstance(only_mlflow, tr.MlflowTracker)
    both = tr.build_tracker(mlflow, experiment="e", tracking_uri="", run="r", rank=0,
                            wandb_module=wandb, wandb_project="p")
    assert isinstance(both, tr.CompositeTracker)
    wandb_only = tr.build_tracker(None, experiment="e", tracking_uri="", run="r", rank=0,
                                  wandb_module=wandb, wandb_project="p")
    assert isinstance(wandb_only, tr.WandbTracker)
    assert isinstance(tr.build_tracker(None, experiment="e", tracking_uri="", run="r", rank=1,
                                       wandb_module=wandb, wandb_project="p"), tr.NullTracker)


# ------------------------------------------- resuming into the same W&B run
def test_wandb_tracker_passes_a_run_id_so_a_relaunch_resumes():
    """A pod restart relaunches the same command; without a stable id W&B
    opens a NEW run and the arm's curves end up split across two."""
    fake = FakeWandb()
    tracker = tr.WandbTracker(fake, project="p", run="r", group="g", run_id="abc123")
    tracker.start({})
    assert fake.calls[0][1]["id"] == "abc123"
    assert fake.calls[0][1]["resume"] == "allow"


def test_the_run_id_is_derived_from_the_output_directory():
    """Two arms share the method, backbone and seed, so the run NAME is
    identical; the output directory is what actually distinguishes them."""
    plain = tr.run_id_for("/work/salidas/tone/ssl_plain")
    proposed = tr.run_id_for("/work/salidas/tone/ssl_proposed")
    assert plain != proposed
    assert plain == tr.run_id_for("/work/salidas/tone/ssl_plain/")
    assert len(plain) == 12 and plain.isalnum()


def test_the_run_name_carries_the_arm_so_two_arms_are_told_apart():
    assert tr.run_name("simsiam_dinov2", "dinov2_vits14", 42, arm="ssl_plain") == (
        "simsiam_dinov2-dinov2_vits14-s42-ssl_plain"
    )
    assert tr.run_name("simsiam", "resnet50", 42) == "simsiam-resnet50-s42"


# ── W&B fail-closed: pedir seguimiento y no tenerlo es un error, no un aviso ──


class FakeWandbWithApi:
    """The slice of wandb this code touches."""

    class _Api:
        def __init__(self, api_key):
            self.api_key = api_key

    def __init__(self, api_key="k" * 40, init_returns="run"):
        self.api = self._Api(api_key)
        self._init_returns = init_returns
        self.calls = []

    def init(self, **kwargs):
        self.calls.append(("init", kwargs))
        if isinstance(self._init_returns, Exception):
            raise self._init_returns
        return self._init_returns


def _importer_for(module):
    def importer(name):
        assert name == "wandb"
        return module

    return importer


def _missing_importer(name):
    raise ImportError(f"No module named {name!r}")


class TestRequireWandb:
    """Every case passes ``environ`` explicitly: tests/conftest.py sets
    WANDB_MODE=disabled for the whole suite, so a case that omits it exercises
    the offline branch instead of the one it names.

    Engram #1590: wandb lived in ~/work/proyectonp/.venv and the run used
    /opt/conda/bin/python, so tracking was 'disabled' in silence and a whole
    GPU budget produced nothing anyone could link to."""

    def test_no_project_means_no_module_and_no_error(self):
        assert tr.require_wandb("", importer=_missing_importer) is None

    def test_a_missing_package_is_an_error_not_a_warning(self):
        with pytest.raises(tr.WandbUnavailable, match="wandb"):
            tr.require_wandb(
                "dermssl-segmentacion", importer=_missing_importer, environ={}
            )

    def test_the_error_names_the_interpreter(self):
        """The fix is always 'use the other python'; the message has to say which."""
        with pytest.raises(tr.WandbUnavailable, match=re.escape(sys.executable)):
            tr.require_wandb(
                "dermssl-segmentacion", importer=_missing_importer, environ={}
            )

    def test_an_importable_package_without_credentials_is_refused(self):
        """wandb imports fine and init then fails; the guard would swallow it."""
        fake = FakeWandbWithApi(api_key=None)
        with pytest.raises(tr.WandbUnavailable, match="credencial|login|WANDB_API_KEY"):
            tr.require_wandb("p", importer=_importer_for(fake), environ={})

    def test_the_environment_variable_counts_as_a_credential(self):
        fake = FakeWandbWithApi(api_key=None)
        module = tr.require_wandb(
            "p", importer=_importer_for(fake), environ={"WANDB_API_KEY": "x" * 40}
        )
        assert module is fake

    def test_a_netrc_credential_counts(self):
        fake = FakeWandbWithApi(api_key="k" * 40)
        assert tr.require_wandb("p", importer=_importer_for(fake), environ={}) is fake

    def test_offline_mode_is_refused_because_it_produces_no_link(self):
        """Pau is sent the run URL; an offline run has none."""
        fake = FakeWandbWithApi()
        with pytest.raises(tr.WandbUnavailable, match="offline|WANDB_MODE"):
            tr.require_wandb(
                "p", importer=_importer_for(fake), environ={"WANDB_MODE": "offline"}
            )


class TestStrictWandbTracker:
    def test_strict_start_propagates_instead_of_printing(self):
        fake = FakeWandbWithApi(init_returns=RuntimeError("no network"))
        tracker = tr.WandbTracker(fake, project="p", run="r", strict=True)
        with pytest.raises(RuntimeError, match="no network"):
            tracker.start({})

    def test_strict_refuses_an_init_that_returned_nothing(self):
        """wandb.init can return None; every later log would be a silent no-op."""
        fake = FakeWandbWithApi(init_returns=None)
        tracker = tr.WandbTracker(fake, project="p", run="r", strict=True)
        with pytest.raises(tr.WandbUnavailable, match="init"):
            tracker.start({})

    def test_lenient_start_still_swallows(self):
        """Default stays fail-safe: an old run must not die because of a sink."""
        fake = FakeWandbWithApi(init_returns=RuntimeError("no network"))
        tracker = tr.WandbTracker(fake, project="p", run="r")
        tracker.start({})

    def test_run_url_is_exposed_for_reporting(self):
        class Run:
            url = "https://wandb.ai/team/p/runs/abc"

        fake = FakeWandbWithApi(init_returns=Run())
        tracker = tr.WandbTracker(fake, project="p", run="r", strict=True)
        tracker.start({})
        assert tracker.run_url == "https://wandb.ai/team/p/runs/abc"

    def test_run_url_is_empty_before_start(self):
        assert tr.WandbTracker(FakeWandbWithApi(), project="p", run="r").run_url == ""


class TestNetrcPermissions:
    """A credential file others can read is a credential others have.

    The pod's ~/work is an NFS export mounted 777, with no sticky bit and
    sec=sys (the server authenticates by UID, and every JupyterHub pod runs as
    uid 1000). Putting a netrc there is the only way to survive a container
    restart -- $HOME is an ephemeral overlay -- so the file's own mode is the
    last line of defence and it gets checked rather than assumed.
    """

    def _netrc(self, tmp_path, mode):
        path = tmp_path / ".netrc"
        path.write_text("machine api.wandb.ai login user password secret\n")
        path.chmod(mode)
        return path

    def test_a_world_readable_netrc_is_refused(self, tmp_path):
        path = self._netrc(tmp_path, 0o644)
        with pytest.raises(tr.WandbUnavailable, match="permisos|600"):
            tr.require_wandb("p", environ={"NETRC": str(path)})

    def test_a_group_readable_netrc_is_refused(self, tmp_path):
        path = self._netrc(tmp_path, 0o640)
        with pytest.raises(tr.WandbUnavailable, match="permisos|600"):
            tr.require_wandb("p", environ={"NETRC": str(path)})

    def test_a_world_writable_netrc_is_refused(self, tmp_path):
        """Writable means replaceable: someone else's credential, silently."""
        path = self._netrc(tmp_path, 0o606)
        with pytest.raises(tr.WandbUnavailable, match="permisos|600"):
            tr.require_wandb("p", environ={"NETRC": str(path)})

    def test_a_missing_netrc_at_the_declared_path_is_refused(self, tmp_path):
        """Pointing NETRC at nothing means wandb silently falls back to ~/.netrc."""
        with pytest.raises(tr.WandbUnavailable, match="no existe|NETRC"):
            tr.require_wandb("p", environ={"NETRC": str(tmp_path / "ausente")})

    def test_a_600_netrc_is_accepted(self, tmp_path):
        path = self._netrc(tmp_path, 0o600)
        fake = FakeWandbWithApi(api_key=None)
        assert (
            tr.require_wandb(
                "p", importer=_importer_for(fake), environ={"NETRC": str(path)}
            )
            is fake
        )

    def test_a_400_netrc_is_accepted(self, tmp_path):
        """Stricter than 600 is fine; only MORE open is refused."""
        path = self._netrc(tmp_path, 0o400)
        fake = FakeWandbWithApi(api_key=None)
        assert (
            tr.require_wandb(
                "p", importer=_importer_for(fake), environ={"NETRC": str(path)}
            )
            is fake
        )

    def test_without_the_variable_nothing_is_checked(self, tmp_path):
        """No NETRC means the default path, which is not ours to police."""
        fake = FakeWandbWithApi()
        assert tr.require_wandb("p", importer=_importer_for(fake), environ={}) is fake
