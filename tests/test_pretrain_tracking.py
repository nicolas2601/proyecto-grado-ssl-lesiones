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
