"""Tests for safe inference from an existing student checkpoint."""

from argparse import Namespace

import numpy as np
import pytest

import seg_paths  # noqa: F401
import train_student


def test_predict_only_does_not_require_pseudo_masks():
    args = train_student.parse_args([
        "--cache-dir", "cache",
        "--output-dir", "student_r2",
        "--predict-only",
    ])

    assert args.predict_only
    assert args.pseudo_masks is None
    assert not args.overwrite_predictions


def test_predict_only_requires_existing_output_directory(tmp_path):
    args = train_student.parse_args([
        "--cache-dir", "cache",
        "--output-dir", str(tmp_path / "missing"),
        "--predict-only",
    ])

    with pytest.raises(SystemExit, match="carpeta de salida"):
        train_student.run(args)


def test_predict_only_requires_existing_checkpoint(tmp_path):
    args = train_student.parse_args([
        "--cache-dir", "cache",
        "--output-dir", str(tmp_path),
        "--predict-only",
    ])

    with pytest.raises(SystemExit, match=r"best\.pt"):
        train_student.run(args)


def test_predict_only_rejects_existing_predictions_without_overwrite(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"checkpoint")
    (tmp_path / "predictions.npy").write_bytes(b"existing")
    args = train_student.parse_args([
        "--cache-dir", "cache",
        "--output-dir", str(tmp_path),
        "--predict-only",
    ])

    with pytest.raises(SystemExit, match=r"predictions\.npy"):
        train_student.run(args)


def test_predict_only_writes_expected_path_without_training(tmp_path, monkeypatch):
    output_dir = tmp_path / "student_r2"
    output_dir.mkdir()
    (output_dir / "best.pt").write_bytes(b"checkpoint")
    args = train_student.parse_args([
        "--cache-dir", "cache",
        "--output-dir", str(output_dir),
        "--predict-only",
        "--preload",
        "--precision", "fp32",
        "--batch-size", "4",
        "--seed", "7",
    ])
    context = Namespace(is_main=True, device="cpu")
    calls = []

    class FakeDist:
        def init_distributed(self, *, force_cpu):
            calls.append(("init", force_cpu))
            return context

        def cleanup(self, received_context):
            calls.append(("cleanup", received_context))

    class FakeEngine:
        PREDICTIONS_FILE = "predictions.npy"

        def train(self, args, context):
            raise AssertionError("predict-only must not train")

        def write_predictions(self, args, context, images, output_path):
            calls.append(("predict", args.preload, args.precision, args.batch_size, args.seed, output_path))
            return output_path

    monkeypatch.setitem(__import__("sys").modules, "dist_utils", FakeDist())
    monkeypatch.setitem(__import__("sys").modules, "student_engine", FakeEngine())
    monkeypatch.setitem(
        __import__("sys").modules,
        "cache_io",
        Namespace(open_cache=lambda path, preload=False: np.zeros((2, 4, 4, 3), dtype=np.uint8)),
    )

    train_student.run(args)

    assert calls[1] == ("predict", True, "fp32", 4, 7, output_dir / "predictions.npy")


def test_predict_only_accepts_explicit_overwrite(tmp_path):
    (tmp_path / "best.pt").write_bytes(b"checkpoint")
    (tmp_path / "predictions.npy").write_bytes(b"existing")
    args = train_student.parse_args([
        "--cache-dir", "cache",
        "--output-dir", str(tmp_path),
        "--predict-only",
        "--overwrite-predictions",
    ])

    assert train_student._validate_predict_only(args).name == "predictions.npy"
