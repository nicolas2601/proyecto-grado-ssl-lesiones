"""The tracker must degrade loudly, never crash a run, and log through the
mlflow module it is handed - which is how we test it without mlflow."""


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
