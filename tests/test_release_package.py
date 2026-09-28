"""Release bookkeeping: head file, checksums, size limit and parity gates."""

import json

import numpy as np
import pytest

from dermssl import metrics
from dermssl.release import package as pk

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def _probas(n: int, seed: int) -> np.ndarray:
    logits = np.random.default_rng(seed).normal(size=(n, 7)) * 3
    exp = np.exp(logits - logits.max(1, keepdims=True))
    return exp / exp.sum(1, keepdims=True)


def test_head_roundtrip_keeps_float32_tensors_and_metadata(tmp_path):
    pytest.importorskip("safetensors")
    weight, bias = np.random.default_rng(0).normal(size=(7, 5)), np.arange(7.0)
    path = tmp_path / "head.safetensors"
    pk.write_head(path, weight, bias, CLASSES, {"center_crop": 224})
    head = pk.read_head(path)
    assert head["weight"].dtype == np.float32 and head["weight"].shape == (7, 5)
    np.testing.assert_array_equal(head["bias"], bias.astype(np.float32))
    assert head["classes"] == CLASSES and head["preprocessing"] == {"center_crop": 224}


def test_head_refuses_mismatched_class_count(tmp_path):
    with pytest.raises(ValueError, match="classes"):
        pk.write_head(tmp_path / "h.safetensors", np.zeros((6, 3)), np.zeros(6), CLASSES, {})


def test_sha256sums_cover_every_file_with_relative_paths(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"abc")
    (tmp_path / "y.txt").write_text("hi")
    lines = pk.write_sha256sums(tmp_path).read_text().splitlines()
    assert lines == [
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad  a/x.bin",
        "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4  y.txt",
    ]


def test_size_report_lists_files_and_fails_over_the_limit(tmp_path):
    (tmp_path / "small.bin").write_bytes(b"x" * 10)
    (tmp_path / "big.bin").write_bytes(b"x" * 30)
    report = pk.size_report(tmp_path, limit_bytes=100)
    assert [(r["path"], r["bytes"]) for r in report] == [("big.bin", 30), ("small.bin", 10)]
    with pytest.raises(pk.AssetTooLarge, match="big.bin"):
        pk.size_report(tmp_path, limit_bytes=20)


def test_forms_agree_passes_close_forms_and_names_the_bad_pair():
    base = _probas(100, 1)
    ok = pk.forms_agree({"plain": base, "hf": base + 1e-6, "onnx": base - 1e-6}, atol=1e-4)
    assert ok["pass"] and ok["pairs"]["hf-onnx"]["argmax_identical"]
    shifted = base.copy()
    shifted[0] = np.roll(shifted[0], 1)
    bad = pk.forms_agree({"plain": base, "onnx": shifted}, atol=1e-4)
    assert not bad["pass"] and bad["pairs"]["plain-onnx"]["argmax_mismatches"] == 1


def test_f1_against_json_reports_delta_and_changed_predictions():
    y = np.arange(200) % 7
    proba = _probas(200, 2)
    f1 = metrics.summarize(y, proba, 7)["macro_f1"]
    same = pk.f1_against_json(y, proba, f1, reference_proba=proba)
    assert same["equal"] and same["delta"] == 0.0 and same["argmax_changes_vs_reference"] == 0
    off = pk.f1_against_json(y, proba, f1 + 0.01)
    assert not off["equal"] and off["delta"] == pytest.approx(-0.01)


def test_fp16_rule_uses_the_registered_thresholds():
    y = np.arange(1000) % 7
    full = _probas(1000, 3)
    assert pk.fp16_rule(y, full, full)["publish"]
    flipped = full.copy()
    flipped[:6] = np.roll(flipped[:6], 1, axis=1)  # 0.6 % of argmax changed
    verdict = pk.fp16_rule(y, full, flipped)
    assert not verdict["publish"] and verdict["argmax_agreement"] == pytest.approx(0.994)
    assert verdict["rule"] == {"min_argmax_agreement": 0.995, "max_abs_delta_macro_f1": 0.002}


def test_json_writer_is_stable(tmp_path):
    path = pk.write_json(tmp_path / "r.json", {"b": np.float64(0.5), "a": np.int64(2)})
    assert json.loads(path.read_text()) == {"a": 2, "b": 0.5}


def _json_result(y, proba):
    s = metrics.summarize(y, proba, 7)
    return {k: s[k] for k in ("macro_f1", "balanced_accuracy", "ece")}


def test_deployment_metrics_report_every_metric_with_its_delta_to_the_evaluation():
    y = np.arange(700) % 7
    evaluated = _probas(700, 4)
    deployed = evaluated.copy()
    deployed[:5] = np.roll(deployed[:5], 1, axis=1)
    block = pk.deployment_metrics(y, deployed, _json_result(y, evaluated), evaluated)
    assert set(block["metrics"]) == {"macro_f1", "balanced_accuracy", "accuracy",
                                     "mel_sensitivity", "mel_auc", "ece"}
    for name, row in block["metrics"].items():
        assert row["delta"] == pytest.approx(row["value"] - row["evaluated"]), name
    assert block["n_predictions_changed"] == 5 and block["n_images"] == 700
    same = pk.deployment_metrics(y, evaluated, _json_result(y, evaluated), evaluated)
    assert same["n_predictions_changed"] == 0
    assert all(row["delta"] == 0 for row in same["metrics"].values())


def test_deployment_metrics_take_json_values_where_the_json_has_them():
    y = np.arange(70) % 7
    proba = _probas(70, 5)
    json_result = {**_json_result(y, proba), "macro_f1": 0.123}
    block = pk.deployment_metrics(y, proba, json_result, proba)
    assert block["metrics"]["macro_f1"]["evaluated"] == 0.123
    assert block["metrics"]["macro_f1"]["evaluated_source"] == "final_eval.json"
    assert block["metrics"]["mel_auc"]["evaluated_source"] == "final_eval.probas.npz"
