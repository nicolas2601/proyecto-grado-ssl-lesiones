"""Tests for scripts/run_circle_dinov2_adaptation.py."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.run_circle_dinov2_adaptation import evaluate_probe, parse_args, train_linear_probe


def test_parse_args_defaults() -> None:
    args = parse_args([
        "--segmenter-checkpoint", "fake_model.pt",
        "--images-dir", "fake_dir",
        "--out-dir", "output_dir",
    ])
    assert args.backbone == "dinov2_vitb14"
    assert args.epochs == 50
    assert args.reg_weight == 1.0
    assert args.shift_buckets == 2


def test_train_and_evaluate_probe_variants() -> None:
    import torch

    device = torch.device("cpu")
    n_samples = 32
    n_features = 64
    n_classes = 3

    rng = np.random.RandomState(42)
    x_orig = rng.randn(n_samples, n_features).astype(np.float32)
    x_shift = x_orig + 0.1 * rng.randn(n_samples, n_features).astype(np.float32)
    y = (rng.rand(n_samples) * n_classes).astype(np.int64)

    for variant in ("baseline", "regularization", "synthetic_blend"):
        probe = train_linear_probe(
            x_orig=x_orig,
            x_shift=x_shift,
            y=y,
            mode=variant,
            n_classes=n_classes,
            epochs=3,
            batch_size=8,
            lr=1e-2,
            reg_weight=1.0,
            synthetic_ratio=0.5,
            seed=42,
            device=device,
        )
        assert probe is not None

        metrics = evaluate_probe(probe, x_orig, y, n_classes, device)
        assert "macro_f1" in metrics
        assert "balanced_accuracy" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert len(metrics["per_class"]) == n_classes
