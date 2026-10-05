"""Hub DINOv2 -> Hugging Face Dinov2Model keys, and the plain torch module.

One ``model.safetensors`` in HF layout serves the plain torch form and the
transformers form. The conversion must reproduce the hub CLS token (the
feature the probe was trained on) at 224, which needs the position grid
pre-interpolated with the hub's own rule: HF interpolates differently.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import release_helpers as rh  # noqa: E402

from dermssl.release import plain_dinov2 as plain  # noqa: E402
from dermssl.release import torch_convert as tc  # noqa: E402


def _images(n: int = 2, seed: int = 1, size: int = 224):
    return torch.from_numpy(np.random.default_rng(seed).normal(size=(n, 3, size, size)).astype("float32"))


def _hf_random(seed: int = 0):
    torch.manual_seed(seed)
    config = transformers.Dinov2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                                       image_size=224, patch_size=14, layerscale_value=0.7)
    return transformers.Dinov2Model(config).eval()


def test_plain_module_matches_hf_dinov2model():
    hf = _hf_random()
    module = plain.PlainDinov2.from_config(hf.config.to_dict())
    module.load_state_dict(hf.state_dict(), strict=True)
    x = _images()
    with torch.no_grad():
        np.testing.assert_allclose(module(x).numpy(), hf(pixel_values=x).pooler_output.numpy(),
                                   rtol=0, atol=1e-5)


def test_plain_module_refuses_another_resolution():
    module = plain.PlainDinov2.from_config(_hf_random().config.to_dict())
    with pytest.raises(ValueError, match="224"):
        module(_images(size=448))


def test_converted_keys_and_shapes_are_exactly_dinov2model():
    hub = rh.tiny_hub_model()
    state = tc.hub_to_hf_state(hub)
    config = transformers.Dinov2Config(**tc.hf_config(hub))
    expected = {k: tuple(v.shape) for k, v in transformers.Dinov2Model(config).state_dict().items()}
    assert {k: tuple(v.shape) for k, v in state.items()} == expected


def test_saved_backbone_reproduces_the_hub_cls_in_both_forms(tmp_path):
    hub = rh.tiny_hub_model(seed=3)
    tc.save_hf_backbone(hub, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} >= {"config.json", "model.safetensors"}
    x = _images(3, seed=5)
    with torch.no_grad():
        reference = hub(x).numpy()
        hf = transformers.AutoModel.from_pretrained(tmp_path).eval()
        np.testing.assert_allclose(hf(pixel_values=x).pooler_output.numpy(), reference,
                                   rtol=0, atol=1e-5)
        module = plain.load_backbone(tmp_path)
        np.testing.assert_allclose(module(x).numpy(), reference, rtol=0, atol=1e-5)


def test_hub_interpolation_is_not_what_hf_would_do():
    """Guard for the design: without the pre-interpolated grid HF drifts from the hub."""
    hub = rh.tiny_hub_model(seed=4)
    config = transformers.Dinov2Config(**{**tc.hf_config(hub), "image_size": 518})
    hf = transformers.Dinov2Model(config).eval()
    state = tc.hub_to_hf_state(hub)
    state["embeddings.position_embeddings"] = hub.pos_embed.detach().clone()
    hf.load_state_dict(state, strict=True)
    x = _images()
    with torch.no_grad():
        drift = np.abs(hf(pixel_values=x).pooler_output.numpy() - hub(x).numpy()).max()
    assert drift > 1e-4


@pytest.mark.parametrize("override", [dict(num_register_tokens=1), dict(ffn_layer="swiglu")])
def test_unsupported_hub_variants_are_refused(override):
    hub = rh.tiny_hub_model(**override)
    with pytest.raises(ValueError):
        tc.hf_config(hub)
