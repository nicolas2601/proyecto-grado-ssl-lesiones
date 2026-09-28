"""ONNX end-to-end export: preprocessed image in, class probabilities out.

Checked on a tiny random classifier built exactly like the release one
(plain DINOv2 layout + fused head), so CI needs no weights.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from dermssl.release import onnx_tools as ot  # noqa: E402
from dermssl.release import plain_dinov2 as plain  # noqa: E402
from dermssl.release import torch_export as te  # noqa: E402

N_CLASSES = 7


def _classifier(seed: int = 0):
    torch.manual_seed(seed)
    backbone = plain.PlainDinov2(dim=32, depth=2, heads=2, patch=14, image_size=224)
    with torch.no_grad():
        for param in backbone.parameters():
            param.copy_(torch.randn_like(param) * 0.2)
    weight, bias = torch.randn(N_CLASSES, 32, dtype=torch.float64), torch.randn(N_CLASSES)
    return plain.Classifier(backbone, weight, bias).eval()


def _batch(n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 3, 224, 224)).astype(np.float32)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    model = _classifier()
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    te.export_onnx(model, path)
    return model, path


@pytest.mark.parametrize("n", [1, 5])
def test_onnxruntime_matches_torch_with_a_dynamic_batch(exported, n):
    model, path = exported
    batch = _batch(n)
    with torch.no_grad():
        expected = model(torch.from_numpy(batch)).numpy()
    got = ot.predict(ot.session(path), batch)
    assert got.shape == (n, N_CLASSES)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-5)
    np.testing.assert_array_equal(got.argmax(1), expected.argmax(1))


def test_graph_names_are_the_documented_ones(exported):
    sess = ot.session(exported[1])
    assert [i.name for i in sess.get_inputs()] == [ot.INPUT_NAME]
    assert [o.name for o in sess.get_outputs()] == [ot.OUTPUT_NAME]


def test_fp16_copy_keeps_float32_io_and_stays_close(exported, tmp_path):
    fp16 = tmp_path / "model_fp16.onnx"
    ot.convert_fp16(exported[1], fp16)
    batch = _batch(4, seed=2)
    full = ot.predict(ot.session(exported[1]), batch)
    half = ot.predict(ot.session(fp16), batch)
    assert half.dtype == np.float64 and half.shape == full.shape
    np.testing.assert_allclose(half.sum(1), 1.0, atol=1e-3)
    np.testing.assert_allclose(half, full, rtol=0, atol=2e-2)
    assert fp16.stat().st_size < exported[1].stat().st_size


def test_predict_batches_long_inputs(exported):
    sess = ot.session(exported[1])
    batch = _batch(7, seed=3)
    np.testing.assert_allclose(ot.predict(sess, batch, batch_size=3), ot.predict(sess, batch),
                               rtol=0, atol=1e-6)
