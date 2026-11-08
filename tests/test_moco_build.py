from __future__ import annotations

import json

import numpy as np
import pytest

from dermssl.release import moco_build
from dermssl.release.moco_inputs import CLASSES, ENCODER_SHA256


def test_reader_loading_rejects_wrong_ontology_or_source(tmp_path):
    reader = {"classes": list(CLASSES["PAD"]), "feature_dim": 3072,
              "weight": np.zeros((6, 3072)).tolist(), "bias": [0.] * 6,
              "C": 30., "provenance": {"encoder_sha256": ENCODER_SHA256, "task": "PAD"}}
    path = tmp_path / "PAD.json"
    path.write_text(json.dumps(reader))
    assert moco_build.load_reader(path, "PAD")["C"] == 30.
    reader["classes"].reverse()
    path.write_text(json.dumps(reader))
    with pytest.raises(ValueError, match="classes"):
        moco_build.load_reader(path, "PAD")
    reader["classes"].reverse()
    reader["provenance"]["encoder_sha256"] = "0" * 64
    path.write_text(json.dumps(reader))
    with pytest.raises(ValueError, match="source"):
        moco_build.load_reader(path, "PAD")


def test_api_report_is_generated_from_aggregates_without_private_case_data():
    source = {"release_version": "10.0.0", "cohorts": {"HAM": {}, "PAD": {}},
              "scope": "development", "aggregation": "mean, not ensemble"}
    report = moco_build.api_report(source)
    assert report["release"] == "v10.0.0"
    assert report["method"]["family"] == "MoCo-v3-style"
    assert report["classification"] == source["cohorts"]
    assert "release" not in source
    source["cohorts"]["HAM"]["image_ids"] = ["private-id"]
    with pytest.raises(ValueError, match="private"):
        moco_build.api_report(source)


def test_probability_gate_is_explicit_and_never_adjusts_tolerance():
    expected = np.array([[.1, .2, .7], [.8, .1, .1]])
    assert moco_build.probability_gate(expected, expected)["passed"]
    wrong = expected[:, ::-1]
    assert not moco_build.probability_gate(wrong, expected)["passed"]
    invalid = expected.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        moco_build.probability_gate(invalid, expected)
    with pytest.raises(ValueError, match="shape"):
        moco_build.probability_gate(expected[:1], expected)
