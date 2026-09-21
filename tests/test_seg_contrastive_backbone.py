"""Torch tests for the MoCo v2 backbone, split BatchNorm and the feature providers."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

pytestmark = pytest.mark.torch

import seg_paths  # noqa: F401,E402
import contrastive_backbone as cb  # noqa: E402


def test_split_batchnorm_differs_from_plain_in_train_and_matches_in_eval():
    torch.manual_seed(0)
    plain = torch.nn.BatchNorm2d(3)
    split = cb.SplitBatchNorm2d(3, num_splits=2)
    x = torch.randn(8, 3, 4, 4) * torch.arange(1, 9).view(8, 1, 1, 1)
    assert not torch.allclose(plain.train()(x), split.train()(x))
    split.load_state_dict(plain.state_dict())  # same running statistics
    plain.eval(), split.eval()
    assert torch.allclose(plain(x), split(x), atol=1e-6)


def test_split_batchnorm_updates_running_statistics_and_rejects_indivisible_batches():
    split = cb.SplitBatchNorm2d(2, num_splits=2).train()
    before = split.running_mean.clone()
    split(torch.randn(4, 2, 3, 3) + 5.0)
    assert not torch.equal(before, split.running_mean)
    with pytest.raises(ValueError):
        split(torch.randn(3, 2, 3, 3))


def test_convert_batchnorm_keeps_state_dict_keys_compatible():
    plain = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))
    keys = set(plain.state_dict())
    cb.convert_batchnorm_to_split(plain, 2)
    assert isinstance(plain[1], cb.SplitBatchNorm2d)
    assert set(plain.state_dict()) == keys


@pytest.mark.parametrize("stride,expected", [(2, 2), (1, 4)])
def test_trunk_layer4_stride_controls_spatial_size(stride, expected):
    trunk = cb.ResNet50Trunk(layer4_stride=stride).eval()
    layer3, layer4 = trunk.forward_maps(torch.randn(1, 3, 64, 64))
    assert layer3.shape[1:] == (1024, 4, 4)
    assert layer4.shape[1:] == (2048, expected, expected)


def test_moco_loss_is_finite_and_queue_wraps_around():
    torch.manual_seed(0)
    moco = cb.MoCoV2(embed_dim=8, queue_size=6, momentum=0.9, temperature=0.2, bn_splits=1)
    q = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    k = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    assert torch.isfinite(moco.loss(q, k))
    moco.enqueue(k)
    assert int(moco.queue_ptr) == 4
    moco.enqueue(k)
    assert int(moco.queue_ptr) == 2
    assert torch.allclose(moco.queue[0], k[2], atol=1e-6)  # wrapped write


def test_moco_loss_is_lower_when_query_matches_its_key():
    torch.manual_seed(1)
    moco = cb.MoCoV2(embed_dim=8, queue_size=32, bn_splits=1)
    k = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    other = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    assert moco.loss(k, k) < moco.loss(other, k)


def test_momentum_update_moves_key_encoder_toward_query_encoder():
    moco = cb.MoCoV2(embed_dim=8, queue_size=8, momentum=0.5, bn_splits=1)
    for parameter in moco.encoder_q.parameters():
        parameter.data.add_(1.0)
    before = next(moco.encoder_k.parameters()).clone()
    moco.momentum_update()
    after = next(moco.encoder_k.parameters())
    query = next(moco.encoder_q.parameters())
    assert torch.allclose(after, 0.5 * before + 0.5 * query, atol=1e-6)


def test_encode_key_preserves_input_order_after_internal_shuffle():
    moco = cb.MoCoV2(embed_dim=8, queue_size=8, bn_splits=1)
    moco.encoder_k.eval()  # eval BN is batch independent, so order is checkable
    x = torch.randn(4, 3, 32, 32)
    expected = moco.encoder_k(x)
    assert torch.allclose(moco.encode_key(x), expected, atol=1e-5)


def test_moco_key_encoder_has_no_trainable_parameters():
    moco = cb.MoCoV2(embed_dim=8, queue_size=8, bn_splits=1)
    assert not any(p.requires_grad for p in moco.encoder_k.parameters())


def test_feature_provider_concatenates_layer3_and_layer4(tmp_path):
    trunk = cb.ResNet50Trunk(layer4_stride=2, bn_splits=1)
    path = tmp_path / "backbone.pt"
    cb.save_backbone(path, trunk, {"resolution": 64})
    provider = cb.ResNet50MoCoFeatures.from_checkpoint(path)
    features = provider.extract(torch.randn(2, 3, 64, 64))
    assert provider.out_channels == 3072
    assert features.shape == (2, 3072, 4, 4)
    assert not any(p.requires_grad for p in provider.parameters())


def test_fused_provider_sums_channels_and_resizes_to_first_map():
    class _Fake(torch.nn.Module):
        def __init__(self, channels, size):
            super().__init__()
            self.out_channels, self.size = channels, size

        def extract(self, images):
            return torch.ones(images.shape[0], self.out_channels, self.size, self.size)

    fused = cb.FusedFeatureProvider([_Fake(5, 4), _Fake(3, 2)])
    assert fused.out_channels == 8
    assert fused.extract(torch.zeros(2, 3, 8, 8)).shape == (2, 8, 4, 4)


def test_external_weights_load_with_prefixes_and_report_mismatches(tmp_path):
    source = cb.ResNet50Trunk(layer4_stride=2, bn_splits=1)
    state = {f"module.encoder_q.{k}": v for k, v in source.state_dict().items()}
    state["module.encoder_q.fc.weight"] = torch.zeros(1)
    path = tmp_path / "external.pt"
    torch.save({"state_dict": state}, path)
    target = cb.ResNet50Trunk(layer4_stride=2, bn_splits=1)
    missing, unexpected = cb.load_external_weights(target, path)
    assert missing == [] and unexpected == []
    first = next(iter(source.state_dict()))
    assert torch.equal(source.state_dict()[first], target.state_dict()[first])
