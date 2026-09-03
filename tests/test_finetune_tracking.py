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
