"""The pod packaging script: guards and the refit that must equal the evaluation."""

import json

import numpy as np
import pytest

from scripts import build_release_package as brp

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def _features(dim: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_tr, y_va = np.arange(140) % 7, np.arange(35) % 7
    centres = rng.normal(size=(7, dim)) * 2
    make = lambda y: (centres[y] + rng.normal(size=(len(y), dim))).astype(np.float32)  # noqa: E731
    return {"x_tr": make(y_tr), "y_tr": y_tr, "x_va": make(y_va), "y_va": y_va}


def test_model_identity_maps_arm_to_role_and_label():
    assert brp.model_identity("B_s43") == {"model": "B_s43", "arm": "B", "arm_label": "B*",
                                           "role": "proposed", "seed": 43}
    assert brp.model_identity("A_s42")["role"] == "plain"
    with pytest.raises(ValueError):
        brp.model_identity("C_s42")


def test_packaged_weights_must_be_the_evaluated_ones(tmp_path):
    weights = tmp_path / "final_B" / "w.pt"
    doc = {"split": "val", "specs": {"proposed": f"dinov2_vitb14:{weights}"}}
    brp.check_evaluated_spec(doc, "proposed", weights)
    with pytest.raises(ValueError, match="evaluated from"):
        brp.check_evaluated_spec(doc, "proposed", tmp_path / "other.pt")
    with pytest.raises(ValueError, match="val"):
        brp.check_evaluated_spec({**doc, "split": "holdout"}, "proposed", weights)


def test_split_other_than_c1_is_refused():
    split = {"train_ids": ["a", "b"], "val_ids": ["c"]}
    with pytest.raises(ValueError, match="C1"):
        brp.check_split(split, {"eval_ids_sha256": "x"})


def test_refit_is_the_probe_c1_final_eval_scores_with():
    pytest.importorskip("sklearn")
    from dermssl import probes

    f = _features()
    out = brp.refit_and_fuse(f["x_tr"], f["y_tr"], f["x_va"], 7)
    expected = probes.run_probe("logreg", f["x_tr"], f["y_tr"], f["x_va"], 7)
    np.testing.assert_array_equal(out["eval_proba"], expected)
    assert out["fused_max_abs_diff"] < 1e-5  # float32 standardisation in the evaluated path


def test_package_model_writes_forms_release_json_and_work(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("onnxruntime")
    import release_helpers as rh

    hub = rh.tiny_hub_model(seed=2)
    meta = {**brp.model_identity("B_s42"), "json_macro_f1": 0.5}
    release = brp.package_model(hub, _features(), CLASSES, tmp_path / "B_s42", tmp_path / "w",
                                meta, fp16=False)
    written = json.loads((tmp_path / "B_s42" / "release.json").read_text())
    assert written["classes"] == CLASSES == release["classes"]
    assert written["probe_settings"]["max_iter"] == 2000
    assert (tmp_path / "B_s42" / "hf" / "model.safetensors").is_file()
    assert (tmp_path / "B_s42" / "onnx" / "model.onnx").is_file()
    with np.load(tmp_path / "w" / "eval.npz") as work:
        assert work["eval_proba"].shape == (35, 7) and work["weight"].shape == (7, 32)
        assert work["head_proba"].shape == (35, 7)


def test_packaging_only_runs_on_cuda():
    base = ["--ham-root", "h", "--model", "B_s42", "--weights", "w.pt", "--final-eval", "f.json"]
    assert brp.parse_args(base).device == "cuda"
    with pytest.raises(SystemExit):
        brp.parse_args([*base, "--device", "cpu"])
