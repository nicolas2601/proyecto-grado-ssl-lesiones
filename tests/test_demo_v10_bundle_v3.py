from __future__ import annotations

import pytest
from PIL import Image

from demo.backend.v10_artifacts import (
    BACKBONE_SHA256,
    SCHEMA_V2,
    SCHEMA_V3,
    V2_RECIPE,
    V3_RECIPE,
    V3_RELEASES,
    V10ArtifactError,
)
from demo.backend.v10_service import MoCoModelService
from tests.test_demo_v10_bundle_v2 import V2Session, _bundle


def _service(tmp_path, **kwargs):
    options = {"schema": SCHEMA_V3, "recipe": V3_RECIPE, "release": "v10.5.0", **kwargs}
    pin, files = _bundle(tmp_path, **options)
    session = V2Session()
    service = MoCoModelService(bundle_dir=tmp_path, expected_manifest_sha256=pin, session_factory=lambda _: session)
    return service, session, files


def test_v3_runs_one_448_view_per_image(tmp_path):
    service, session, files = _service(tmp_path)

    result = service.predict(Image.new("RGB", (32, 24)), model_key="MOCO_HAM7")

    assert session.feed_shapes and all(shape == (1, 3, 448, 448) for shape in session.feed_shapes)
    assert result["pred"] == "mel"
    assert result["views"] == 1
    assert result["inference_mode"] == "single_view_448"
    assert result["identity"]["head_sha256"] == files["heads/HAM-reader.json"]


def test_v3_registers_the_single_view_release_on_the_same_encoder():
    assert V3_RELEASES == {"v10.5.0": BACKBONE_SHA256}
    assert V3_RECIPE == {**V2_RECIPE, "tta": "none", "views": 1}


def test_v3_status_reports_the_single_view_recipe(tmp_path):
    service, _, _ = _service(tmp_path)

    status = service.status()

    assert status["classification_ready"] is True
    assert status["release"] == "v10.5.0"
    assert status["identity"]["recipe"] == V3_RECIPE


def test_v3_rejects_the_d4_recipe(tmp_path):
    service, _, _ = _service(tmp_path, recipe=V2_RECIPE)

    with pytest.raises(V10ArtifactError):
        service.predict(Image.new("RGB", (8, 8)), model_key="MOCO_HAM7")


def test_v3_rejects_a_d4_release(tmp_path):
    service, _, _ = _service(tmp_path, release="v10.4.0")

    with pytest.raises(V10ArtifactError):
        service.predict(Image.new("RGB", (8, 8)), model_key="MOCO_HAM7")


def test_v2_schema_rejects_the_single_view_release(tmp_path):
    service, _, _ = _service(tmp_path, schema=SCHEMA_V2, recipe=V2_RECIPE)

    with pytest.raises(V10ArtifactError):
        service.predict(Image.new("RGB", (8, 8)), model_key="MOCO_HAM7")
