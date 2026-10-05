"""W&B + MLflow trackers must degrade gracefully and log through the modules
they are handed; no real wandb/mlflow needed to test them."""

from dermssl.finetune import tracking as tr
from dermssl.pretrain.tracking import NullTracker


class FakeWandb:
    def __init__(self):
        self.calls = []

    def init(self, **kw):
        self.calls.append(("init", kw))
        return "run"

    def log(self, payload, step=None):
        self.calls.append(("log", payload, step))

    def save(self, path):
        self.calls.append(("save", path))

    def finish(self):
        self.calls.append(("finish",))


class FakeMlflow:
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


def test_no_modules_means_null_composite():
    t = tr.build_trackers(wandb_module=None, mlflow_module=None, run="r",
                          wandb_project="p", wandb_mode="offline")
    assert t.is_null
    t.start({"a": 1}); t.log_metric("x", 1.0); t.log_artifact("f"); t.end()  # no crash


def test_wandb_disabled_mode_is_not_started():
    w = FakeWandb()
    t = tr.build_trackers(wandb_module=w, mlflow_module=None, run="r",
                          wandb_project="p", wandb_mode="disabled")
    assert t.is_null and w.calls == []


def test_both_backends_receive_every_call_in_order():
    w, m = FakeWandb(), FakeMlflow()
    t = tr.build_trackers(wandb_module=w, mlflow_module=m, run="ft-resnet50-ham10000-s42",
                          wandb_project="dermssl", wandb_mode="offline",
                          wandb_entity="lab", wandb_tags=["resnet50"], wandb_group="ham10000",
                          mlflow_experiment="finetune-ham10000",
                          mlflow_tracking_uri="file:/tmp/mlruns")
    assert not t.is_null and len(t.trackers) == 2
    t.start({"modelo": "resnet50"})
    t.log_metrics({"val/bal_acc": 0.5, "smoke": True, "name": "x"}, step=1)
    t.log_artifact("results.json")
    t.end()

    init = w.calls[0][1]
    assert init["project"] == "dermssl" and init["entity"] == "lab"
    assert init["mode"] == "offline" and init["group"] == "ham10000"
    assert ("log", {"val/bal_acc": 0.5}, 1) in w.calls
    assert not any(c[0] == "log" and "smoke" in c[1] for c in w.calls), "bools/str skipped"
    assert w.calls[-2:] == [("save", "results.json"), ("finish",)]

    assert m.calls[0] == ("set_tracking_uri", "file:/tmp/mlruns")
    assert ("set_experiment", "finetune-ham10000") in m.calls
    assert ("log_metric", "val/bal_acc", 0.5, 1) in m.calls
    assert m.calls[-1] == ("end_run",)


def test_null_trackers_are_dropped_from_composite():
    c = tr.CompositeTracker([NullTracker(), NullTracker()])
    assert c.is_null


class FakeSummaryWriter:
    """Stands in for torch.utils.tensorboard.SummaryWriter."""

    instances: list = []

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.calls = []
        FakeSummaryWriter.instances.append(self)

    def add_hparams(self, hparams, metrics, run_name=None):
        self.calls.append(("add_hparams", hparams, metrics, run_name))

    def add_scalar(self, tag, value, global_step=None):
        self.calls.append(("add_scalar", tag, value, global_step))

    def add_text(self, tag, text, global_step=None):
        self.calls.append(("add_text", tag, text, global_step))

    def flush(self):
        self.calls.append(("flush",))

    def close(self):
        self.calls.append(("close",))


def test_tensorboard_tracker_writes_under_run_named_dir():
    FakeSummaryWriter.instances.clear()
    t = tr.build_trackers(wandb_module=None, mlflow_module=None,
                          run="ft-resnet50-ham10000-s42",
                          wandb_project="p", wandb_mode="disabled",
                          summary_writer_cls=FakeSummaryWriter,
                          tensorboard_dir="/tmp/tb")
    assert not t.is_null and len(t.trackers) == 1
    t.start({"modelo": "resnet50", "lr": 1e-4, "smoke": True, "tags": ["a"]})
    t.log_metrics({"val/balanced_accuracy": 0.5}, step=3)
    t.log_artifact("results.json")
    t.end()

    w = FakeSummaryWriter.instances[0]
    assert w.log_dir == "/tmp/tb/ft-resnet50-ham10000-s42"
    assert ("add_scalar", "val/balanced_accuracy", 0.5, 3) in w.calls
    hparams = [c for c in w.calls if c[0] == "add_text" and c[1] == "hparams"]
    assert hparams and "resnet50" in hparams[0][2], "params land as text, not lost"
    assert w.calls[-1] == ("close",)


def test_tensorboard_needs_both_a_writer_class_and_a_dir():
    FakeSummaryWriter.instances.clear()
    only_dir = tr.build_trackers(wandb_module=None, mlflow_module=None, run="r",
                                 wandb_project="p", wandb_mode="disabled",
                                 tensorboard_dir="/tmp/tb")
    only_cls = tr.build_trackers(wandb_module=None, mlflow_module=None, run="r",
                                 wandb_project="p", wandb_mode="disabled",
                                 summary_writer_cls=FakeSummaryWriter)
    assert only_dir.is_null and only_cls.is_null
    assert FakeSummaryWriter.instances == [], "no writer opened without a dir"
