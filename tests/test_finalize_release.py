"""Release assembly: parity verdicts decide, sizes and checksums close the folder."""

import json

import pytest

from scripts import finalize_release as fr

ROW = {"value": 0.6, "evaluated": 0.6, "delta": 0.0, "evaluated_source": "final_eval.json"}
FP32 = {"n_images": 3, "n_predictions_changed": 0,
        "metrics": {k: ROW for k in ("macro_f1", "balanced_accuracy", "accuracy",
                                     "mel_sensitivity", "mel_auc", "ece")}}


def _model(root, name, pass_hard=True, json_ok=True, fp16_ok=True):
    model = root / name
    (model / "onnx").mkdir(parents=True)
    (model / "onnx" / "model.onnx").write_bytes(b"onnx")
    (model / "onnx" / "model_fp16.onnx").write_bytes(b"fp16")
    parity = {"pass": pass_hard and json_ok, "gates": {"x": pass_hard and json_ok},
              "deployment_metrics": {"onnx_fp32": FP32, "onnx_fp16": {"gate": {"publish": fp16_ok},
                                                                       **FP32}}}
    (model / "parity.json").write_text(json.dumps(parity))
    return model


def test_every_copied_source_exists():
    missing = [name for name, src in fr.COPIES.items() if not src.is_file()]
    assert not missing


def test_failed_fp16_rule_removes_the_copy(tmp_path):
    model = _model(tmp_path, "B_s42", fp16_ok=False)
    assert fr.check_model(model)["fp16_published"] is False
    assert not (model / "onnx" / "model_fp16.onnx").exists()
    assert (model / "onnx" / "model.onnx").exists()


def test_models_without_parity_or_with_failed_gates_are_refused(tmp_path):
    (tmp_path / "A_s42").mkdir()
    with pytest.raises(fr.ReleaseRefused, match="parity"):
        fr.check_model(tmp_path / "A_s42")
    with pytest.raises(fr.ReleaseRefused, match="gate"):
        fr.check_model(_model(tmp_path, "A_s43", pass_hard=False))


def test_json_drift_cannot_be_accepted(tmp_path):
    with pytest.raises(fr.ReleaseRefused, match="gate"):
        fr.check_model(_model(tmp_path, "B_s43", json_ok=False))
    with pytest.raises(SystemExit):
        fr.main(["--release", str(tmp_path), "--allow-json-drift"])


def test_finalize_writes_sizes_and_checksums_and_enforces_the_limit(tmp_path):
    _model(tmp_path, "B_s42")
    result = fr.finalize(tmp_path, ["B_s42"])
    sums = (tmp_path / "SHA256SUMS").read_text()
    for name in ("B_s42/onnx/model.onnx", "README.md", "preprocess.py", "sizes.json", "LICENSE",
                 "LICENSE-WEIGHTS", "MODEL_CARD.md",
                 "results/s42/final_eval.json", "web/preprocess.js"):
        assert f"  {name}\n" in sums, name
    assert json.loads((tmp_path / "sizes.json").read_text())["files"] == result["sizes"]
    assert "| `B_s42` | ONNX fp32 |" in (tmp_path / "MODEL_CARD.md").read_text()
    with pytest.raises(fr.package.AssetTooLarge):
        fr.finalize(tmp_path, ["B_s42"], limit_bytes=100)
