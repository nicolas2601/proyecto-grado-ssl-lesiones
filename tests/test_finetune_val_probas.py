"""Per-image val probabilities from a saved FT checkpoint (inference only, no retraining)."""

import dataclasses
import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import ft_support  # noqa: E402

from dermssl.finetune import torch_run, val_probas  # noqa: E402
from dermssl.finetune.tracking import CompositeTracker  # noqa: E402

CORE = ("accuracy", "balanced_accuracy", "macro_f1", "auc_ovr_macro", "ece")


@pytest.fixture
def trained(tmp_path, monkeypatch):
    ft_support.patch_for_cpu(monkeypatch)
    root = ft_support.make_isic2020(tmp_path)
    cfg = ft_support.smoke_cfg(root, tmp_path)
    json_path = torch_run.run(cfg, CompositeTracker([]))
    return cfg, json_path, json_path.parent / "ft_alexnet_isic2020.pth"


def test_roundtrip_checkpoint_to_probas_reproduces_the_run_metrics(trained, tmp_path):
    cfg, json_path, ckpt = trained
    out = tmp_path / "probas" / "alexnet.probas.npz"
    record = val_probas.write_probas(cfg, ckpt, json_path, out)

    with np.load(out, allow_pickle=False) as z:
        data = torch_run.load_split(cfg)
        assert z["val_ids"].tolist() == data.va_ids
        assert z["y"].tolist() == data.y_va.tolist()
        assert z["proba"].shape == (len(data.va_ids), 2)
        assert np.allclose(z["proba"].sum(1), 1.0, atol=1e-5)
        assert z["groups"].dtype.kind == "U" and len(z["groups"]) == len(z["y"])
        assert str(z["checkpoint_sha256"]) == hashlib.sha256(ckpt.read_bytes()).hexdigest()
        recomputed = val_probas.core_metrics(z["y"], z["proba"])
    recorded = json.loads(json_path.read_text())["probes"]["finetune"]
    for key in CORE:
        assert abs(recomputed[key] - recorded[key]) <= 1e-4, key


def test_a_metric_that_differs_from_the_json_aborts_and_names_the_key(trained, tmp_path):
    cfg, json_path, ckpt = trained
    doc = json.loads(json_path.read_text())
    doc["probes"]["finetune"]["balanced_accuracy"] += 0.01
    json_path.write_text(json.dumps(doc))
    out = tmp_path / "probas" / "alexnet.probas.npz"
    with pytest.raises(val_probas.MetricMismatchError, match="balanced_accuracy"):
        val_probas.write_probas(cfg, ckpt, json_path, out)
    assert not out.exists()


def test_a_checkpoint_of_another_model_is_refused(trained, tmp_path):
    cfg, json_path, ckpt = trained
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    state["modelo"] = "resnet50"
    torch.save(state, ckpt)
    with pytest.raises(val_probas.CheckpointMismatchError):
        val_probas.write_probas(cfg, ckpt, json_path, tmp_path / "x.npz")



def test_ids_must_be_the_reference_set_and_the_record_says_so(trained, tmp_path):
    cfg, json_path, ckpt = trained
    ids = torch_run.load_split(cfg).va_ids
    out = tmp_path / "ok.npz"
    record = val_probas.write_probas(cfg, ckpt, json_path, out,
                                     reference_val_ids=list(reversed(ids)))
    assert record["status"] == "pareado" and record["ids_order_equal_reference"] is False
    assert record["mejor_epoca_json"] == json.loads(json_path.read_text())["mejor_epoca"]
    assert record["checkpoint_epoch"] is None  # the .pth carries no epoch
    with pytest.raises(val_probas.IdsMismatchError):
        val_probas.write_probas(cfg, ckpt, json_path, tmp_path / "bad.npz",
                                reference_val_ids=ids[:-1] + ["ISIC_9999999"])
    assert not (tmp_path / "bad.npz").exists()


def test_checkpoint_has_no_epoch_field(trained):
    state = torch.load(trained[2], map_location="cpu", weights_only=True)
    assert set(state) == {"modelo", "dataset", "smoke", "clases", "state_dict"}


def test_pair_models_records_a_failing_model_without_stopping_the_others(trained, tmp_path):
    cfg, json_path, ckpt = trained
    root = ckpt.parent.parent
    # layout <root>/<model>/...: expose the run output under two model names
    (root / "alexnet").mkdir(exist_ok=True)
    for f in ckpt.parent.iterdir():
        if f.is_file():
            (root / "alexnet" / f.name).write_bytes(f.read_bytes())
    sidecar = val_probas.pair_models(
        ["vgg19", "alexnet"], lambda m: dataclasses.replace(cfg, model=m),
        root, root, tmp_path / "probas", None)
    status = {r["model"]: r["status"] for r in sidecar["models"]}
    assert status == {"vgg19": "no pareado", "alexnet": "pareado"}
    assert "FileNotFoundError" in sidecar["models"][0]["reason"]
    assert (tmp_path / "probas" / "alexnet.probas.npz").exists()
    saved = json.loads((tmp_path / "probas" / "ft_isic2020_probas.sidecar.json").read_text())
    assert saved["label"] == val_probas.LABEL


def test_script_parses_defaults_and_points_at_the_committed_reference():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "ft_val_probas", Path(__file__).resolve().parents[1] / "scripts" / "ft_val_probas.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.parse_args(["--data-root", "d", "--ckpt-root", "c"])
    assert len(args.models) == 8
    assert Path(args.reference).is_file(), args.reference
    assert args.out_dir.endswith("resultados/banco_completo/ft_isic2020/probas")


def test_sidecar_states_the_checkpoint_identity_note_globally_and_per_model(trained, tmp_path):
    cfg, json_path, ckpt = trained
    root = ckpt.parent.parent
    (root / "alexnet").mkdir(exist_ok=True)
    for f in ckpt.parent.iterdir():
        if f.is_file():
            (root / "alexnet" / f.name).write_bytes(f.read_bytes())
    sidecar = val_probas.pair_models(
        ["vgg19", "alexnet"], lambda m: dataclasses.replace(cfg, model=m),
        root, root, tmp_path / "probas", None)
    note = "checkpoint_epoch=null; identidad del checkpoint inferida por coincidencia de métricas"
    assert val_probas.CHECKPOINT_IDENTITY_NOTE == note
    saved = json.loads((tmp_path / "probas" / "ft_isic2020_probas.sidecar.json").read_text())
    assert saved["checkpoint_identity_note"] == note
    assert [r["checkpoint_identity_note"] for r in saved["models"]] == [note, note]
    assert sidecar == saved
