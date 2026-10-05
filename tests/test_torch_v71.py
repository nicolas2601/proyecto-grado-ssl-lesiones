"""The torch half of v7.1: WiSE-FT weights, D4 views and the five feature parts.

Run on a tiny random DINOv2 (same class the hub builds) on CPU. The checks
that matter are the ones a wrong implementation still passes silently: alpha
endpoints that are not bit-exact, a CLS that is not the bank's CLS, an
attention pool that is not weighted by the CLS row.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from dermssl import torch_v71 as tv  # noqa: E402
from tests.release_helpers import tiny_hub_model  # noqa: E402


def _model(seed: int):
    return tiny_hub_model(seed=seed, depth=4)  # four blocks: the cls4 feature reads the last four


def _state(seed: int):
    return {k: v.clone() for k, v in _model(seed).state_dict().items()}


class TestWeights:
    def test_alpha_endpoints_are_bit_exact(self):
        public, adapted = _state(0), _state(1)
        assert tv.max_abs_diff(tv.interpolate_state(public, adapted, 0.0), public) == 0.0
        assert tv.max_abs_diff(tv.interpolate_state(public, adapted, 1.0), adapted) == 0.0

    def test_alpha_is_linear(self):
        public, adapted = _state(0), _state(1)
        mid = tv.interpolate_state(public, adapted, 0.4)
        key = next(iter(public))
        assert torch.allclose(mid[key], 0.6 * public[key] + 0.4 * adapted[key])

    def test_soup_is_the_mean(self):
        a, b = _state(1), _state(2)
        key = next(iter(a))
        assert torch.allclose(tv.average_states(a, b)[key], (a[key] + b[key]) / 2)

    def test_mismatched_keys_are_refused(self):
        public, adapted = _state(0), _state(1)
        adapted.pop(next(iter(adapted)))
        with pytest.raises(ValueError, match="claves"):
            tv.interpolate_state(public, adapted, 0.5)

    def test_non_float_buffers_must_agree(self):
        public = {"w": torch.ones(2), "n": torch.tensor([3])}
        adapted = {"w": torch.zeros(2), "n": torch.tensor([4])}
        with pytest.raises(ValueError, match="no flotante"):
            tv.interpolate_state(public, adapted, 0.5)


def _batch(n: int = 2, size: int = 56, seed: int = 0) -> np.ndarray:
    return np.random.RandomState(seed).randint(0, 256, (n, size, size, 3)).astype(np.uint8)


class TestFeatures:
    def test_cls_is_the_bank_forward(self):
        model = _model(seed=3)
        x = tv.normalize_uint8(_batch())
        parts = tv.extract_parts(model, x)
        with torch.no_grad():
            assert torch.allclose(parts["cls"], model(x), atol=1e-6)

    def test_last_of_the_four_cls_is_the_cls(self):
        model = _model(seed=3)
        parts = tv.extract_parts(model, tv.normalize_uint8(_batch()))
        dim = parts["cls"].shape[1]
        assert parts["cls4"].shape[1] == 4 * dim
        assert torch.allclose(parts["cls4"][:, -dim:], parts["cls"], atol=1e-6)

    def test_uniform_attention_pools_to_the_patch_mean(self):
        model = _model(seed=4)
        attn = tv.last_block(model).attn
        dim = attn.qkv.weight.shape[1]
        with torch.no_grad():  # q = 0 -> every CLS->token logit equal -> uniform weights
            attn.qkv.weight[:dim] = 0
            attn.qkv.bias[:dim] = 0
        parts = tv.extract_parts(model, tv.normalize_uint8(_batch()))
        assert torch.allclose(parts["attn"], parts["mean"], atol=1e-5)

    def test_attention_pool_differs_from_the_mean_in_general(self):
        parts = tv.extract_parts(_model(seed=4), tv.normalize_uint8(_batch()))
        assert not torch.allclose(parts["attn"], parts["mean"], atol=1e-4)

    def test_gem(self):
        tokens = torch.rand(2, 5, 3) + 0.1
        assert torch.allclose(tv.gem(tokens, p=1.0), tokens.mean(dim=1), atol=1e-6)
        assert torch.allclose(tv.gem(torch.full((1, 4, 2), 0.5)), torch.full((1, 2), 0.5), atol=1e-6)


class TestPreprocessing:
    def test_uint8_path_is_the_bank_transform(self):
        from PIL import Image

        from scripts import cluster_frozen_bank as bank

        image = Image.fromarray(_batch(1, 97)[0].repeat(4, axis=0)[:300, :97])
        mine = tv.normalize_uint8(tv.resize_crop_uint8(image, 224)[None])[0]
        assert torch.equal(mine, bank.eval_transform(224)(image))

    def test_d4_has_eight_distinct_views_starting_at_identity(self):
        x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
        views = tv.d4_views(x)
        assert len(views) == 8
        assert torch.equal(views[0], x)
        flat = {tuple(v[0, 0].flatten().tolist()) for v in views}
        assert len(flat) == 8
        assert all(torch.equal(v.sort(dim=-1)[0].sum((2, 3)), x.sort(dim=-1)[0].sum((2, 3))) for v in views)

    def test_extract_split_stacks_views(self):
        model = _model(seed=3)
        out = tv.extract_split(model, _batch(3), device=torch.device("cpu"), batch_size=2, n_views=8)
        assert out["cls"].shape[:2] == (8, 3)
        single = tv.extract_split(model, _batch(3), device=torch.device("cpu"), batch_size=3, n_views=1)
        assert np.allclose(single["cls"][0], out["cls"][0], atol=1e-5)


def test_attention_weights_stay_float32_under_autocast():
    """On the pod extraction runs under autocast; the recomputed CLS->patch
    softmax must not silently drop to half precision with it."""
    model = _model(seed=4)
    x = torch.randn(2, 17, 32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        weights = tv._cls_attention(tv.last_block(model).attn, x, 1)
    assert weights.dtype == torch.float32
    with torch.no_grad():
        exact = tv._cls_attention(tv.last_block(model).attn, x, 1)
    assert torch.allclose(weights, exact, atol=1e-6)
