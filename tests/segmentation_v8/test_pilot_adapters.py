"""Technical guards and offline adapters, without expert reference masks."""
import hashlib
import json

import numpy as np
import pytest
import torch

from dermssl import pilot_adapters as adapters


def test_reference_free_qa_keeps_cases_and_lesion_pixels():
    rgb = np.full((80, 100, 3), [180, 135, 110], dtype=np.uint8)
    lesion = np.zeros(rgb.shape[:2], dtype=bool)
    lesion[25:55, 35:65] = True
    fov = np.ones_like(lesion)
    result = adapters.reference_free_qa(rgb, lesion, fov)
    shifted, applied = adapters.shift_healthy_skin(rgb, result, shift_buckets=1)
    assert applied
    np.testing.assert_array_equal(shifted[lesion], rgb[lesion])
    assert not np.array_equal(shifted[~lesion], rgb[~lesion])
    empty = adapters.reference_free_qa(rgb, np.zeros_like(lesion), fov)
    original, applied = adapters.shift_healthy_skin(rgb, empty, shift_buckets=1)
    assert not applied and not empty.qa_passed
    np.testing.assert_array_equal(original, rgb)


def test_invalid_masks_and_empty_fov_fail_closed():
    rgb = np.full((20, 20, 3), 150, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    for bad in [np.ones((19, 20), dtype=bool), np.ones((20, 20), dtype=float)]:
        with pytest.raises(ValueError):
            adapters.reference_free_qa(rgb, bad, ~mask)
    result = adapters.reference_free_qa(rgb, mask, mask)
    assert not result.qa_passed and result.fallback_reason == "empty_valid_field"


def test_qa_keeps_raw_mask_and_never_shifts_outside_fov():
    rgb = np.full((80, 100, 3), [180, 135, 110], dtype=np.uint8)
    lesion = np.zeros(rgb.shape[:2], dtype=bool)
    lesion[0:25, 20:60] = True
    field = np.ones_like(lesion)
    field[:10] = False
    result = adapters.reference_free_qa(rgb, lesion, field)
    np.testing.assert_array_equal(result.lesion_mask, lesion)
    shifted, _ = adapters.shift_healthy_skin(rgb, result, shift_buckets=1)
    np.testing.assert_array_equal(shifted[~field], rgb[~field])


class TinyCLS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(768))

    def forward_features(self, batch):
        return {"x_norm_clstoken": self.weight[None].expand(len(batch), -1)}


def test_encoder_hash_strict_state_and_preprocessing(tmp_path):
    path = tmp_path / "encoder.pt"
    torch.save(TinyCLS().state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="SHA"):
        adapters.FrozenCLSAdapter(TinyCLS(), path, "0" * 64)
    adapter = adapters.FrozenCLSAdapter(TinyCLS(), path, digest)
    rgb = np.full((90, 180, 3), [128, 64, 32], dtype=np.uint8)
    batch = adapter.preprocess([rgb])
    assert batch.shape == (1, 3, 224, 224)
    expected = (torch.tensor([128, 64, 32]) / 255 - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
    torch.testing.assert_close(batch[0, :, 50, 50], expected)
    features = adapter.encode([rgb])
    assert features.shape == (1, 768) and torch.isfinite(features).all()
    assert not features.requires_grad and not adapter.model.training
    torch.save({"unexpected": torch.ones(2)}, path)
    with pytest.raises(RuntimeError):
        adapters.FrozenCLSAdapter(TinyCLS(), path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_segmenter_requires_matching_identity_and_backbone(tmp_path):
    path = tmp_path / "teacher.pt"
    identity = {"config": {"backbone": "resnet34"}}
    payload = {"run_id": "fixture", "identity": identity,
               "identity_fingerprint": hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
               "backbone": "resnet34", "checkpoint_component": "student", "teacher_state_dict": {}}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="component"):
        adapters.FrozenResNetSegmenter(path, hashlib.sha256(path.read_bytes()).hexdigest(), "fixture")


def test_offline_hub_source_is_pinned_before_execution(tmp_path):
    (tmp_path / "hubconf.py").write_text("raise RuntimeError('must not execute')\n")
    code = tmp_path / "dinov2"
    code.mkdir()
    model_file = code / "model.py"
    model_file.write_text("VERSION = 1\n")
    digest = adapters.hub_source_hash(tmp_path)
    model_file.write_text("VERSION = 2\n")
    assert adapters.hub_source_hash(tmp_path) != digest
    with pytest.raises(ValueError, match="source"):
        adapters.FrozenCLSAdapter.from_local_hub(
            tmp_path, tmp_path / "unused.pt", "a" * 64, expected_source_sha256=digest,
        )
