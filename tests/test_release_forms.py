"""The three release forms of a tiny random model agree with the hub code path.

Same check the pod runs on the real weights (scripts/release_parity.py),
here on a tiny DinoVisionTransformer so CI needs no weights: hub fp32 ->
fused head is the reference; torch (plain), transformers and ONNX fp32 must
match it to atol 1e-4 with identical argmax.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxconverter_common")

import release_helpers as rh  # noqa: E402

from dermssl.release import package as pk  # noqa: E402
from dermssl.release import preprocess as pp  # noqa: E402
from dermssl.release import torch_forms as tf  # noqa: E402

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@pytest.fixture(scope="module")
def tiny_release(tmp_path_factory):
    hub = rh.tiny_hub_model(seed=11)
    rng = np.random.default_rng(0)
    weight, bias = rng.normal(size=(7, 32)), rng.normal(size=7)
    out = tmp_path_factory.mktemp("model")
    paths = tf.write_model_forms(hub, weight, bias, CLASSES, out, pp.PREPROCESSING)
    return hub, weight, bias, out, paths


def _batches(n_batches: int = 2, size: int = 3):
    rng = np.random.default_rng(7)
    return [rng.normal(size=(size, 3, 224, 224)).astype(np.float32) for _ in range(n_batches)]


def test_layout_is_the_documented_one(tiny_release):
    out = tiny_release[3]
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert names == {"hf/config.json", "hf/model.safetensors", "hf/head.safetensors",
                     "onnx/model.onnx", "onnx/model_fp16.onnx"}
    assert pk.read_head(out / "hf" / "head.safetensors")["classes"] == CLASSES


def test_all_forms_match_the_hub_reference(tiny_release):
    hub, weight, bias, out, _ = tiny_release
    batches = _batches()
    forms = tf.run_forms(tf.FormRunner(out), batches)
    reference = np.concatenate([tf.hub_probas(hub, weight, bias, b) for b in batches])
    fp32 = {"hub": reference, **{k: forms[k] for k in ("torch", "transformers", "onnx")}}
    verdict = pk.forms_agree(fp32)
    assert verdict["pass"], verdict
    np.testing.assert_allclose(forms["onnx_fp16"], reference, rtol=0, atol=5e-2)


def test_runner_without_fp16_skips_that_form(tiny_release):
    out = tiny_release[3]
    forms = tf.FormRunner(out, with_fp16=False).predict(_batches(1, 2)[0])
    assert set(forms) == {"torch", "transformers", "onnx"}
