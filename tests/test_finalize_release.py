"""Release assembly: parity verdicts decide, sizes and checksums close the folder."""

import json

import pytest

from scripts import finalize_release as fr


def _model(root, name, pass_hard=True, json_ok=True, fp16_ok=True):
    model = root / name
    (model / "onnx").mkdir(parents=True)
    (model / "onnx" / "model.onnx").write_bytes(b"onnx")
    (model / "onnx" / "model_fp16.onnx").write_bytes(b"fp16")
    parity = {"pass_hard": pass_hard, "json_reproduced": json_ok, "gates": {},
              "fp16": {"publish": fp16_ok}}
    (model / "parity.json").write_text(json.dumps(parity))
    return model


def test_every_copied_source_exists():
    missing = [name for name, src in fr.COPIES.items() if not src.is_file()]
    assert not missing


def test_failed_fp16_rule_removes_the_copy(tmp_path):
    model = _model(tmp_path, "B_s42", fp16_ok=False)
    assert fr.check_model(model, allow_json_drift=False)["fp16_published"] is False
    assert not (model / "onnx" / "model_fp16.onnx").exists()
    assert (model / "onnx" / "model.onnx").exists()


def test_models_without_parity_or_with_failed_gates_are_refused(tmp_path):
    (tmp_path / "A_s42").mkdir()
    with pytest.raises(fr.ReleaseRefused, match="parity"):
        fr.check_model(tmp_path / "A_s42", False)
    with pytest.raises(fr.ReleaseRefused, match="disagree"):
        fr.check_model(_model(tmp_path, "A_s43", pass_hard=False), False)


def test_json_drift_needs_the_explicit_flag(tmp_path):
    model = _model(tmp_path, "B_s43", json_ok=False)
    with pytest.raises(fr.ReleaseRefused, match="not reproduced"):
        fr.check_model(model, False)
    assert fr.check_model(model, True)["json_reproduced"] is False


def test_finalize_writes_sizes_and_checksums_and_enforces_the_limit(tmp_path):
    _model(tmp_path, "B_s42")
    result = fr.finalize(tmp_path, ["B_s42"])
    sums = (tmp_path / "SHA256SUMS").read_text()
    for name in ("B_s42/onnx/model.onnx", "README.md", "preprocess.py", "sizes.json",
                 "results/s42/final_eval.json", "web/preprocess.js"):
        assert f"  {name}\n" in sums, name
    assert json.loads((tmp_path / "sizes.json").read_text())["files"] == result["sizes"]
    with pytest.raises(fr.package.AssetTooLarge):
        fr.finalize(tmp_path, ["B_s42"], limit_bytes=100)
