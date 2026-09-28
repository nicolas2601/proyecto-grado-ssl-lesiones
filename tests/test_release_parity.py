"""Parity gates of scripts/release_parity.py on a tiny random release."""

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxconverter_common")

import release_helpers as rh  # noqa: E402

from dermssl import metrics  # noqa: E402
from dermssl.release import preprocess as pp  # noqa: E402
from dermssl.release import torch_forms as tf  # noqa: E402
from scripts import release_parity as rpar  # noqa: E402

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    hub = rh.tiny_hub_model(seed=21)
    rng = np.random.default_rng(1)
    weight, bias = rng.normal(size=(7, 32)) * 3, rng.normal(size=7)
    out = tmp_path_factory.mktemp("m")
    tf.write_model_forms(hub, weight, bias, CLASSES, out, pp.PREPROCESSING)
    batches = [rng.normal(size=(4, 3, 224, 224)).astype(np.float32) for _ in range(3)]
    forms, reference = rpar.collect(tf.FormRunner(out),
                                    lambda b: tf.hub_probas(hub, weight, bias, b), batches, n_ref=6)
    y = np.arange(12) % 7
    return forms, reference, y


def test_collect_limits_the_reference_to_the_first_images(run):
    forms, reference, _ = run
    assert reference.shape == (6, 7)
    assert {k: v.shape for k, v in forms.items()} == {k: (12, 7) for k in
                                                       ("torch", "transformers", "onnx", "onnx_fp16")}


def _json(y, proba):
    summary = metrics.summarize(y, proba, 7)
    return {k: summary[k] for k in ("macro_f1", "balanced_accuracy", "ece")}


def _report(forms, reference, y, evaluated, head=None, json_result=None, on_cuda=True):
    head = evaluated if head is None else head
    return rpar.parity_report(forms, reference, y, json_result or _json(y, evaluated), evaluated,
                              evaluated, head, packaged_on_cuda=on_cuda)


def test_report_passes_when_forms_agree_and_the_packaged_head_is_the_evaluated_one(run):
    forms, reference, y = run
    report = _report(forms, reference, y, forms["torch"])
    assert report["pass"], report["gates"]
    assert report["evaluation_path"]["max_abs_diff_vs_committed_probas"] == 0.0


def test_deployment_metrics_cover_onnx_fp32_and_fp16_with_the_fp16_gate(run):
    forms, reference, y = run
    evaluated = forms["torch"].copy()
    evaluated[:2] = np.roll(evaluated[:2], 1, axis=1)  # the JSON predictions differ on 2 images
    block = _report(forms, reference, y, evaluated)["deployment_metrics"]
    assert set(block) == {"onnx_fp32", "onnx_fp16"}
    assert block["onnx_fp32"]["n_predictions_changed"] == 2
    assert set(block["onnx_fp32"]["metrics"]) == {"macro_f1", "balanced_accuracy", "accuracy",
                                                  "mel_sensitivity", "mel_auc", "ece"}
    assert set(block["onnx_fp16"]["gate"]["rule"]) == {"min_argmax_agreement",
                                                       "max_abs_delta_macro_f1"}


def test_deployment_drift_does_not_fail_but_a_head_that_misses_the_json_does(run):
    forms, reference, y = run
    evaluated = np.roll(forms["torch"], 1, axis=1)  # deployment differs from the evaluation
    assert _report(forms, reference, y, evaluated)["pass"]
    off = _report(forms, reference, y, evaluated, json_result={**_json(y, evaluated), "macro_f1": 0.9})
    assert not off["pass"] and not off["gates"]["evaluation_path_equals_json"]
    moved = _report(forms, reference, y, evaluated, head=forms["torch"])
    assert not moved["pass"] and not moved["gates"]["packaged_head_equals_json"]


def test_a_head_packaged_without_cuda_autocast_fails(run):
    forms, reference, y = run
    report = _report(forms, reference, y, forms["torch"], on_cuda=False)
    assert not report["pass"] and not report["gates"]["packaged_on_cuda"]


def test_report_fails_when_a_form_disagrees(run):
    forms, reference, y = run
    broken = {**forms, "onnx": np.roll(forms["onnx"], 1, axis=1)}
    report = _report(broken, reference, y, forms["torch"])
    assert not report["pass"] and not report["gates"]["forms_agree_first_images"]


def test_there_is_no_way_to_accept_json_drift():
    with pytest.raises(SystemExit):
        rpar.parse_args(["--ham-root", "x", "--model-dir", "y", "--final-eval", "z",
                         "--allow-json-drift"])
