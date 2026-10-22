from __future__ import annotations

import base64
import json
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from demo.backend.f4_service import F4Service, F4ServiceError


def _artifact(feature_dim: int = 15) -> dict:
    return {
        "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "mean": np.zeros(feature_dim),
        "scale": np.ones(feature_dim),
        "coef": np.zeros((7, feature_dim)),
        "intercept": np.zeros(7),
        "feature_dim": feature_dim,
        "descriptor_mean": np.zeros(7),
        "descriptor_std": np.ones(7),
        "descriptor_names": ["d0", "d1", "d2", "d3", "d4", "d5", "d6"],
        "protocol": "B_s42_a1.0@224|cls4|balanced|tta",
    }


def _service(*, rendered=None, extracted=None, described=None, artifact=None, hair_fn=None):
    feature_dim = 4
    model_feature_dim = 2 * feature_dim + 7
    artifact = _artifact(model_feature_dim) if artifact is None else artifact
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    calls = []

    def segment(image_array):
        calls.append(("segment", image_array))
        return mask

    def render(image_array, mask_array, *, resize_crop, hair_fn):
        calls.append(("render", resize_crop, hair_fn))
        return rendered or ({"full": image_array, "crop_view": image_array.copy()}, frozenset())

    def describe(image_array, mask_array, hair_fn):
        calls.append(("describe", hair_fn))
        return np.arange(7, dtype=float) if described is None else described

    class DescriptorModule:
        @staticmethod
        def apply_standardizer(stats, values):
            values = np.asarray(values)
            return (values - np.asarray(stats["mean"])) / np.asarray(stats["std"]), [0] * 7

    def build_variant(name, features, descriptor_values):
        assert name == "F4"
        return np.concatenate(
            [features["full"], features["crop_view"],
             np.broadcast_to(descriptor_values[None], (8, 1, 7))], axis=-1
        )

    def extract(model, images, *, device, batch_size, n_views):
        calls.append(("extract", model, device, batch_size, n_views, len(images)))
        if extracted is not None:
            return extracted
        base = np.arange(8 * 4, dtype=np.float32).reshape(8, 1, 4)
        return {"cls4": base + (0 if np.array_equal(images[0], image) else 100)}

    service = F4Service(
        backbone=object(),
        segment_mask=segment,
        resize_crop=lambda im, size: im,
        extract_split=extract,
        artifact=artifact,
        device="cpu",
        batch_size=2,
        hair_fn=(lambda image_array: np.zeros(image_array.shape[:2], dtype=bool))
        if hair_fn is None else hair_fn,
        renderer=render,
        describer=describe,
        variant_builder=build_variant,
        descriptor_module=DescriptorModule,
        feature_dim=4,
    )
    return service, image, mask, calls


def test_predict_composes_two_cls4_conditions_and_averages_probabilities():
    artifact = _artifact(15)
    artifact["coef"][4, 0] = 1
    artifact["intercept"][4] = 2
    service, image, mask, calls = _service(artifact=artifact)

    result = service.predict(Image.fromarray(image))

    assert result["classes"] == artifact["classes"]
    assert result["feature_dim"] == 15
    assert sum(result["probs"].values()) == pytest.approx(1.0)
    assert result["fallback"] is False
    assert (result["mask_width"], result["mask_height"]) == (mask.shape[1], mask.shape[0])
    decoded_mask = np.asarray(Image.open(BytesIO(base64.b64decode(result["mask_png_base64"]))))
    assert decoded_mask.shape == mask.shape
    assert set(np.unique(decoded_mask)) <= {0, 255}
    json.dumps(result)
    assert [call[0] for call in calls] == ["segment", "render", "describe", "extract", "extract"]
    assert [call[4] for call in calls if call[0] == "extract"] == [8, 8]
    assert len(result["mask_png_base64"]) > 0
    logits = np.arange(8) * 4 + 2
    expected_mel = np.mean(np.exp(logits) / (np.exp(logits) + 6))
    assert result["probs"]["mel"] == pytest.approx(expected_mel)


def test_predict_reports_crop_fallback_without_substituting_baseline_silently():
    rendered = ({"full": np.zeros((224, 224, 3), dtype=np.uint8),
                 "crop_view": np.zeros((224, 224, 3), dtype=np.uint8)}, frozenset({"crop_view"}))
    service, image, _, _ = _service(rendered=rendered)

    result = service.predict(Image.fromarray(image))

    assert result["fallback"] is True
    assert result["fallback_conditions"] == ["crop_view"]


def test_predict_rejects_noncanonical_protocol_and_dimensions():
    artifact = _artifact(feature_dim=14)
    service, image, _, _ = _service(artifact=artifact)

    with pytest.raises(F4ServiceError, match="artifact"):
        service.predict(Image.fromarray(image))


def test_predict_rejects_invalid_segmentation_mask_shape():
    service, image, _, _ = _service()
    service.segment_mask = lambda image_array: np.zeros((2, 2), dtype=bool)

    with pytest.raises(F4ServiceError, match="mask"):
        service.predict(Image.fromarray(image))


def test_predict_rejects_missing_hair_detector_for_registered_f4():
    service, image, _, _ = _service(hair_fn=False)

    with pytest.raises(F4ServiceError, match="hair"):
        service.predict(Image.fromarray(image))


@pytest.mark.parametrize("key,bad", [("descriptor_mean", float("nan")), ("descriptor_std", float("nan")), ("scale", -1.0)])
def test_invalid_scaling_fails_closed(key, bad):
    artifact = _artifact(15)
    artifact[key][0] = bad
    service, image, _, _ = _service(artifact=artifact)
    with pytest.raises(F4ServiceError, match="invalid|finite"):
        service.predict(Image.fromarray(image))


def test_unused_condition_fallback_is_not_a_f4_crop_failure():
    rendered = ({"full": np.zeros((8, 8, 3), dtype=np.uint8),
                 "crop_view": np.zeros((8, 8, 3), dtype=np.uint8)}, frozenset({"crop_orig448"}))
    service, image, _, _ = _service(rendered=rendered)
    assert service.predict(Image.fromarray(image))["fallback"] is False
