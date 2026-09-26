"""The bank gate has to leave a dashboard entry, including when it fails.

This gate's whole output is "condition (0) reproduces the published bank, or
it does not". That verdict is reported to Pau as a W&B link, so a run that
cannot produce one has to stop before it builds a backbone, not after -- and a
FAILING gate has to be logged too. A gate that only records its successes is
worse than no gate: it turns the dashboard into evidence for one side.

Engram #1590 is why this is tested rather than assumed: W&B was "disabled" in
silence once already because the package lived in a different interpreter.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dermssl.pretrain import tracking

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location(
        "c1_verify_bank_reproduction", REPO / "scripts" / "c1_verify_bank_reproduction.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RecordingTracker:
    def __init__(self):
        self.params = None
        self.metrics = {}
        self.ended = False

    def start(self, params):
        self.params = dict(params)

    def log_metric(self, key, value, step=None):
        self.metrics[key] = value

    def end(self):
        self.ended = True


class TestTrackingConfig:
    def test_the_config_carries_what_identifies_the_run(self, gate):
        config = gate.tracking_config(
            backbone="dinov2_public_vitb14",
            source="public",
            tolerance=1e-4,
            code_revision="abc1234",
            masks_sha256="a" * 64,
            n_images=64,
        )
        assert config["backbone"] == "dinov2_public_vitb14"
        assert config["source"] == "public"
        assert config["tolerance"] == 1e-4
        assert config["code_revision"] == "abc1234"
        assert config["masks_sha256"] == "a" * 64
        assert config["n_images"] == 64

    def test_the_condition_is_recorded_so_the_zero_fallbacks_are_readable(self, gate):
        """Without it, 'cls_fallbacks: 0' looks like a measurement of (c)."""
        config = gate.tracking_config(
            backbone="b", source="public", tolerance=1e-4,
            code_revision="r", masks_sha256="a" * 64, n_images=1,
        )
        assert config["condition"] == gate.BASELINE_CONDITION


class TestMetricsAreLoggedOnBothOutcomes:
    def _report(self, reproduces):
        return {
            "reproduces": reproduces,
            "max_abs_diff": 1.5e-5,
            "mean_abs_diff": 2.0e-6,
            "n_rows_over_tolerance": 0 if reproduces else 7,
            "n_rows": 64,
        }

    def test_a_passing_gate_is_logged(self, gate):
        tracker = RecordingTracker()
        gate.log_report(tracker, self._report(True))
        assert tracker.metrics["max_abs_diff"] == 1.5e-5
        assert tracker.metrics["mean_abs_diff"] == 2.0e-6
        assert tracker.metrics["n_rows_over_tolerance"] == 0
        assert tracker.metrics["reproduces"] == 1.0

    def test_a_FAILING_gate_is_logged_too(self, gate):
        """A dashboard that only holds successes is evidence for one side."""
        tracker = RecordingTracker()
        gate.log_report(tracker, self._report(False))
        assert tracker.metrics["reproduces"] == 0.0
        assert tracker.metrics["n_rows_over_tolerance"] == 7

    def test_reproduces_is_numeric_so_it_charts(self, gate):
        tracker = RecordingTracker()
        gate.log_report(tracker, self._report(True))
        assert isinstance(tracker.metrics["reproduces"], float)


class TestWandbIsStrictAndEarly:
    def test_no_project_means_no_tracker_and_no_error(self, gate):
        assert gate.build_gate_tracker("", "run", "group", environ={}) is None

    def test_a_missing_package_stops_the_gate(self, gate):
        def missing(_name):
            raise ImportError("No module named 'wandb'")

        with pytest.raises(tracking.WandbUnavailable):
            gate.build_gate_tracker("p", "run", "group", environ={}, importer=missing)

    def test_offline_mode_stops_the_gate(self, gate):
        """The gate's output IS the link; an offline run has none."""
        with pytest.raises(tracking.WandbUnavailable, match="offline|WANDB_MODE"):
            gate.build_gate_tracker("p", "run", "group", environ={"WANDB_MODE": "offline"})

    def test_the_tracker_is_strict(self, gate):
        class FakeWandb:
            class api:  # noqa: N801
                api_key = "k" * 40

            def init(self, **kwargs):
                return None  # the silent no-op wandb.init can return

        tracker = gate.build_gate_tracker(
            "p", "run", "group", environ={}, importer=lambda _n: FakeWandb()
        )
        with pytest.raises(tracking.WandbUnavailable, match="init"):
            tracker.start({})

    def test_a_credential_free_install_stops_the_gate(self, gate):
        """wandb imports, init would fail, and the tracker would swallow it."""
        class FakeWandb:
            class api:  # noqa: N801
                api_key = None

        with pytest.raises(tracking.WandbUnavailable, match="credencial"):
            gate.build_gate_tracker(
                "p", "run", "group", environ={}, importer=lambda _n: FakeWandb()
            )
