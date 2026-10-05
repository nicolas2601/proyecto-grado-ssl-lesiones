"""The real extractor of ``isic_eval_run.py`` on a tiny DINOv2: weights, soup, public and a full cell."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import isic_eval_run as run  # noqa: E402
import v71_io  # noqa: E402
from dermssl import e3_store  # noqa: E402
from dermssl import torch_v71 as tv  # noqa: E402
from dermssl.isic_eval import runner as rn  # noqa: E402
from dermssl.isic_eval import splits as sp  # noqa: E402
from tests.release_helpers import tiny_hub_model  # noqa: E402


@pytest.fixture
def real(tmp_path, monkeypatch):
    monkeypatch.setattr(v71_io, "hub_model", lambda arch, pretrained: tiny_hub_model(seed=0 if pretrained else 5, depth=4))
    states = {}
    for seed, name in enumerate(("final_A_s42", "final_A_s43", "final_B_s42"), start=11):
        (tmp_path / "w" / name).mkdir(parents=True)
        model = tiny_hub_model(seed=seed, depth=4)
        torch.save(model.state_dict(), tmp_path / "w" / name / v71_io.WEIGHTS_FILE)
        states[name] = model.state_dict()
    n = 60
    ids = tuple(f"ISIC_{k:05d}" for k in range(n))
    rng = np.random.RandomState(0)
    (tmp_path / "img").mkdir()
    for k, i in enumerate(ids):
        Image.fromarray((rng.randint(0, 60, (40, 52, 3)) + 60 * (k % 3)).astype(np.uint8)).save(tmp_path / "img" / f"{i}.jpg")
    idx = np.arange(n)
    ds = sp.EvalDataset("isicfake", ("a", "b", "c"), ids, idx % 3, None, {i: tmp_path / "img" / f"{i}.jpg" for i in ids},
                        idx[idx % 5 != 0], idx[idx % 5 == 0],
                        {"val_ids_sha256": "v" * 64, "train_ids_sha256": "t" * 64})
    args = argparse.Namespace(out_dir=tmp_path / "out", device="cpu", batch_size=8, decode_workers=2)
    bank = run.LayoutBank(tmp_path / "w", True)
    return ds, args, bank, run.build_extractor(args, ds, bank), states


def test_alpha_one_b_weights_are_the_final_itself_and_cls4_has_four_blocks(real):
    ds, _, _, extract, states = real
    got = extract("B_s42_a1.0", "val", ds.val_ids[:4], 1, ("cls4",))["cls4"]
    direct = tiny_hub_model(seed=0, depth=4)
    direct.load_state_dict(states["final_B_s42"])
    images = v71_io.ImageStore(real[1].out_dir / "inputs", ds.paths, 1).array(ds.val_ids[:4], 224, "x")
    want = tv.extract_split(direct, images, device=torch.device("cpu"), batch_size=8)["cls4"]
    assert got.shape == (1, 4, 4 * direct.embed_dim) and np.allclose(got, want, atol=1e-6)


def test_a_soup_is_the_midpoint_of_the_two_finals_and_public_loads(real):
    _, _, bank, extract, states = real
    soup = bank.state("A_soup_a1.0")
    mean = tv.average_states(states["final_A_s42"], states["final_A_s43"])
    assert tv.max_abs_diff(soup, mean) < 1e-6
    assert bank.final_path("A", "s43").parent.name == "final_A_s43"
    assert set(extract("public", "val", ["ISIC_00000"], 1, ("cls", "attn"))) == {"cls", "attn"}


def test_full_cell_with_the_real_extractor_and_eight_views(real, tmp_path):
    ds, _, _, extract, _ = real
    store = e3_store.CellStore(tmp_path / "cells")
    status = rn.run_cell(store, ds, "A71", extract, pool_excludes_val=True, weights_sha="w", n_resamples=10)
    assert status == "done"
    doc = store.read("A71__clean")["result"]
    assert doc["metrics"]["matriz_confusion"] and 0.0 <= doc["metrics"]["macro_f1"] <= 1.0
