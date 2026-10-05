"""The release folds StandardScaler + multinomial logreg into one linear layer.

The fused layer must reproduce the evaluated probe, not an approximation of
it: the same probabilities to 1e-10 on float64 inputs, including features
whose training variance is zero (StandardScaler keeps scale_ = 1 there).
"""

import numpy as np
import pytest

pytest.importorskip("sklearn")

from dermssl import probes  # noqa: E402
from dermssl.release import fuse  # noqa: E402

N_CLASSES = 7
DIM = 12


def _data(seed: int, n: int = 350):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % N_CLASSES
    centres = rng.normal(0.0, 2.0, size=(N_CLASSES, DIM))
    x = centres[y] + rng.normal(0.0, 1.5, size=(n, DIM))
    x *= rng.uniform(0.1, 30.0, size=DIM)  # uneven feature scales, as in ViT features
    x += rng.uniform(-50.0, 50.0, size=DIM)
    return x, y


def test_fit_logreg_is_what_the_logreg_probe_scores_with():
    x, y = _data(0)
    x_val, _ = _data(1, n=40)
    scaler, model = probes.fit_logreg(x.astype(np.float32), y)
    expected = probes.logreg_predict_proba(x.astype(np.float32), y, x_val.astype(np.float32), N_CLASSES)
    got = model.predict_proba(scaler.transform(x_val.astype(np.float32)))
    np.testing.assert_array_equal(got, expected)


def test_fused_layer_equals_scaler_then_logreg_to_1e_10():
    x, y = _data(2)
    x_val, _ = _data(3, n=64)
    scaler, model = probes.fit_logreg(x, y)
    weight, bias = fuse.fuse_probe(scaler, model)
    assert weight.shape == (N_CLASSES, DIM) and bias.shape == (N_CLASSES,)
    expected = model.predict_proba(scaler.transform(x_val))
    np.testing.assert_allclose(fuse.fused_predict_proba(x_val, weight, bias), expected,
                               rtol=0, atol=1e-10)


def test_zero_variance_feature_is_handled_like_standard_scaler():
    x, y = _data(4)
    x[:, 5] = 3.25  # constant in train: StandardScaler sets scale_ = 1 there
    x_val, _ = _data(5, n=32)
    scaler, model = probes.fit_logreg(x, y)
    assert scaler.scale_[5] == 1.0
    weight, bias = fuse.fuse_probe(scaler, model)
    assert np.all(np.isfinite(weight)) and np.all(np.isfinite(bias))
    np.testing.assert_allclose(fuse.fused_predict_proba(x_val, weight, bias),
                               model.predict_proba(scaler.transform(x_val)), rtol=0, atol=1e-10)


def test_binary_logreg_is_refused():
    x, y = _data(6)
    y = (y > 3).astype(int)
    scaler, model = probes.fit_logreg(x, y)
    with pytest.raises(ValueError, match="multinomial"):
        fuse.fuse_probe(scaler, model)


def test_probe_missing_a_class_is_refused():
    x, y = _data(7)
    keep = y != 3
    scaler, model = probes.fit_logreg(x[keep], y[keep])
    with pytest.raises(ValueError, match="classes"):
        fuse.fuse_probe(scaler, model, n_classes=N_CLASSES)
