"""The cluster CLI must be launchable the way Dr. Karen's lab launches iBOT.

Her CT-Encoders runner takes `--data_path` and `--output_dir` and drops its
TensorBoard event files inside the output dir without being asked. Ours keeps
its own dash-style flags, but accepts her spelling and defaults TensorBoard the
same way, so the bank can be launched on her GPUs without learning a new CLI.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import finetune_baseline as fb  # noqa: E402
from dermssl.finetune.config import ConfigError  # noqa: E402


def _cfg(argv):
    return fb.build_config(fb.parse_args(argv))


BASE = ["--model", "resnet50", "--dataset", "ham10000"]


def test_underscore_flags_are_accepted_like_in_ct_encoders():
    cfg = _cfg([*BASE, "--data_path", "/data", "--output_dir", "/out"])
    assert cfg.data_root == "/data" and cfg.output_dir == "/out"


def test_dash_flags_still_work():
    cfg = _cfg([*BASE, "--data-root", "/data", "--output-dir", "/out"])
    assert cfg.data_root == "/data" and cfg.output_dir == "/out"


def test_tensorboard_defaults_into_the_output_dir():
    """Every run gets a dashboard even when nobody passes the flag."""
    cfg = _cfg([*BASE, "--data-root", "/data", "--output-dir", "/out"])
    assert cfg.tensorboard_dir == "/out/tb"


def test_an_explicit_tensorboard_dir_wins():
    cfg = _cfg([*BASE, "--data-root", "/d", "--output-dir", "/out",
                "--tensorboard-dir", "/shared/tb"])
    assert cfg.tensorboard_dir == "/shared/tb"


def test_tensorboard_can_be_turned_off():
    cfg = _cfg([*BASE, "--data-root", "/d", "--output-dir", "/out",
                "--tensorboard-dir", "none"])
    assert cfg.tensorboard_dir == ""


def test_missing_paths_still_fail_loudly():
    with pytest.raises(ConfigError):
        _cfg(BASE)
