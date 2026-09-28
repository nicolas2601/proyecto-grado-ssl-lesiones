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


def test_report_passes_when_forms_agree_and_reproduce_the_json(run):
    forms, reference, y = run
    f1 = metrics.summarize(y, forms["torch"], 7)["macro_f1"]
    report = rpar.parity_report(forms, reference, y, f1, forms["torch"], forms["torch"])
    assert report["pass_hard"] and report["json_reproduced"], report["gates"]
    assert report["evaluation_path"]["max_abs_diff_vs_committed_probas"] == 0.0
    assert set(report["fp16"]["rule"]) == {"min_argmax_agreement", "max_abs_delta_macro_f1"}


def test_report_flags_a_drifted_json_without_failing_the_hard_gates(run):
    forms, reference, y = run
    f1 = metrics.summarize(y, forms["torch"], 7)["macro_f1"]
    report = rpar.parity_report(forms, reference, y, f1 + 0.01, forms["torch"], forms["torch"])
    assert report["pass_hard"] and not report["json_reproduced"]
    assert report["val_macro_f1"]["onnx"]["delta"] == pytest.approx(-0.01)


def test_report_fails_hard_when_a_form_disagrees(run):
    forms, reference, y = run
    broken = {**forms, "onnx": np.roll(forms["onnx"], 1, axis=1)}
    report = rpar.parity_report(broken, reference, y, 0.5, forms["torch"], forms["torch"])
    assert not report["pass_hard"] and not report["gates"]["forms_agree_first_images"]
