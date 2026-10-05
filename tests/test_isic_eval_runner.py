"""Resumable cells, retries, contrasts and pool-label guard of the ISIC evaluation, with a fake extractor."""

import zlib

import numpy as np
import pytest

from dermssl import e3_store
from dermssl.isic_eval import evaluate as ev
from dermssl.isic_eval import protocol as pr
from dermssl.isic_eval import runner as rn
from dermssl.isic_eval import splits as sp
from dermssl.isic_eval import weights as wt

IDENTITY = {"val_ids_sha256": "v" * 64, "train_ids_sha256": "t" * 64, "sklearn_version": "x"}
CLASSES = ("a", "b", "c")


def make_ds(grouped: bool, n: int = 240) -> sp.EvalDataset:
    ids = tuple(f"img{k:04d}" for k in range(n))
    labels = np.arange(n) % 3
    groups = np.array([f"p{k // 2}" for k in range(n)]) if grouped else None
    idx = np.arange(n)
    return sp.EvalDataset("isicfake", CLASSES, ids, labels, groups, {}, idx[idx % 5 != 0], idx[idx % 5 == 0], IDENTITY)


class FakeExtractor:
    """Separable class signal plus per-(id, view) noise; records every call."""

    def __init__(self, ds, fail_first: int = 0, always_fail: bool = False):
        self.label = dict(zip(ds.ids, ds.labels))
        self.calls, self.fail_first, self.always_fail = [], fail_first, always_fail

    def __call__(self, weights, split, ids, n_views, parts):
        self.calls.append((weights, split, tuple(ids), n_views, parts))
        if self.always_fail or len(self.calls) <= self.fail_first:
            raise RuntimeError("extractor down")
        out = {}
        for part in parts:
            views = [np.stack([np.eye(8)[self.label[i]] * 4 + np.random.RandomState(
                zlib.crc32(f"{i}{v}{part}".encode())).randn(8) for i in ids]) for v in range(n_views)]
            out[part] = np.stack(views)
        return out


def run(store, ds, config, extract, pool=True, **kw):
    return rn.run_cell(store, ds, config, extract, pool_excludes_val=pool, weights_sha="w" * 64, n_resamples=20, **kw)


def test_cell_fits_on_train_scores_val_and_labels_transductive(tmp_path):
    ds = make_ds(grouped=False)
    ex = FakeExtractor(ds)
    store = e3_store.CellStore(tmp_path)
    assert run(store, ds, "B71", ex, pool=False) == "done"
    assert [(c[1], c[3]) for c in ex.calls] == [("train", 1), ("val", 8)]
    assert ex.calls[0][2] == tuple(ds.train_ids) and ex.calls[1][2] == tuple(ds.val_ids)
    doc = store.read("B71__original")["result"]
    assert doc["transductivo"] is True and doc["metrics"]["macro_f1"] > 0.8 and doc["n_val"] == len(ds.val_idx)
    assert doc["metrics"]["bootstrap"]["unit"] == "image" and doc["metrics"]["matriz_confusion"]
    assert (tmp_path / "B71__original.probas.npz").is_file()
    assert not (tmp_path / "B71__original.features.tmp.npz").exists()


def test_clean_cell_is_not_transductive_and_public_never_is(tmp_path):
    ds = make_ds(grouped=True)
    store = e3_store.CellStore(tmp_path)
    run(store, ds, "A71", FakeExtractor(ds), pool=True)
    run(store, ds, "PUBLIC_A71", FakeExtractor(ds), pool=False)
    assert store.read("A71__clean")["result"]["transductivo"] is False
    public = store.read("PUBLIC_A71")["result"]
    assert public["transductivo"] is False and public["pool_excludes_val"] is None
    assert public["metrics"]["bootstrap"]["unit"] == "patient_id"


def test_rerun_with_the_same_key_skips_and_a_new_key_recomputes(tmp_path):
    ds = make_ds(False)
    ex, store = FakeExtractor(ds), e3_store.CellStore(tmp_path)
    run(store, ds, "B71", ex)
    n_calls = len(ex.calls)
    assert run(store, ds, "B71", ex) == "skipped" and len(ex.calls) == n_calls
    assert rn.run_cell(store, ds, "B71", ex, pool_excludes_val=True, weights_sha="other" * 8,
                       n_resamples=20) == "done"
    assert list(tmp_path.glob("B71__clean.json.stale-*"))


def test_two_retries_then_done_and_a_dead_cell_is_listed_while_the_rest_goes_on(tmp_path):
    ds = make_ds(False)
    store = e3_store.CellStore(tmp_path)
    assert run(store, ds, "B71", FakeExtractor(ds, fail_first=2)) == "done"
    assert run(store, ds, "A71", FakeExtractor(ds, always_fail=True)) == "failed"
    assert run(store, ds, "PUBLIC_B71", FakeExtractor(ds)) == "done"
    assert [name for name, _ in store.failures] == ["A71__clean"]


def test_features_are_cached_so_a_retry_does_not_extract_again(tmp_path):
    ds = make_ds(False)
    ex, cache = FakeExtractor(ds), tmp_path / "f.tmp.npz"
    first = ev.load_or_extract(ex, ds, "B71", cache)
    again = ev.load_or_extract(ex, ds, "B71", cache)
    assert len(ex.calls) == 2 and np.array_equal(first[1]["cls4"], again[1]["cls4"])


def test_paired_contrasts_need_both_cells_and_use_patient_groups(tmp_path):
    ds = make_ds(grouped=True)
    store, cstore = e3_store.CellStore(tmp_path), e3_store.CellStore(tmp_path / "contrasts")
    run(store, ds, "B71", FakeExtractor(ds), pool=True)
    assert rn.run_all_contrasts(store, cstore, n_resamples=20) == {}
    run(store, ds, "B71", FakeExtractor(ds), pool=False)
    run(store, ds, "PUBLIC_B71", FakeExtractor(ds))
    done = rn.run_all_contrasts(store, cstore, n_resamples=20)
    assert set(done) == {"B71__clean__vs__PUBLIC_B71", "B71__original__vs__PUBLIC_B71",
                         "B71__clean__vs__B71__original"}
    doc = cstore.read("B71__clean__vs__B71__original")["result"]
    assert doc["unit"] == "patient_id" and {r["metric"] for r in doc["paired"]} >= {"macro_f1", "auc"}
    assert set(rn.run_all_contrasts(store, cstore, n_resamples=20).values()) == {"skipped"}


def test_contrast_refuses_cells_scored_on_different_val_images(tmp_path):
    ds = make_ds(False)
    store, cstore = e3_store.CellStore(tmp_path), e3_store.CellStore(tmp_path / "c")
    run(store, ds, "B71", FakeExtractor(ds), pool=True)
    run(store, ds, "A71", FakeExtractor(ds), pool=True)
    path = tmp_path / "A71__clean.probas.npz"
    saved = dict(np.load(path))
    saved["val_ids"] = saved["val_ids"][::-1].copy()
    np.savez(path, **saved)
    with pytest.raises(ValueError, match="different val images"):
        rn.run_contrast(store, cstore, "B71__clean", "A71__clean", n_resamples=10)


def test_limit_keeps_a_class_balanced_subset_of_both_sides():
    small = sp.limited(make_ds(False), 30)
    assert len(small.train_idx) == len(small.val_idx) == 30 and set(small.train_idx).isdisjoint(small.val_idx)
    assert sp.limited(make_ds(False), None).train_idx.size == 192


def test_final_dir_names_follow_the_two_layouts():
    assert wt.final_dir_name("A", 42, False) == "final_A" and wt.final_dir_name("A", 43, False) == "final_A_s43"
    assert wt.final_dir_name("B", 42, True) == "final_B_s42"


def test_pool_label_is_checked_against_the_run_json_exclusion():
    ok = {k: {"exclude_ids_sha256": pr.POOL_EXCLUDE_SHA256["clean"]} for k in ("A_s42", "B_s42")}
    assert wt.check_pool_label(ok, True)["verified"]
    with pytest.raises(wt.PoolLabelError):
        wt.check_pool_label(ok, False)  # clean weights declared as original
    with pytest.raises(wt.PoolLabelError):
        wt.check_pool_label({"A_s42": {}}, True)  # unrecorded exclusion
    assert not wt.check_pool_label({"A_s42": {}}, True, allow_unverified=True)["verified"]


def test_logit_adjustment_and_tta_follow_the_registered_config():
    rng = np.random.RandomState(0)
    y = np.array([0] * 180 + [1] * 20)
    x = rng.randn(200, 3) + y[:, None] * 0.8
    views = np.stack([x + rng.randn(*x.shape) * 0.05 for _ in range(8)])
    tr, ev_parts = {"cls": x[None], "attn": x[None]}, {"cls": views, "attn": views}
    base = {"feature": "cls_attn", "weighting": "none", "tta": True}
    flat = ev.probe_proba(tr, y, ev_parts, base, 2)
    adjusted = ev.probe_proba(tr, y, ev_parts, {**base, "weighting": "la0.5"}, 2)
    single = ev.probe_proba(tr, y, ev_parts, {**base, "tta": False}, 2)
    assert adjusted.shape == flat.shape == (200, 2) and np.allclose(adjusted.sum(1), 1)
    assert adjusted[:, 1].mean() > flat[:, 1].mean()  # the rare class gains mass
    assert not np.allclose(single, flat)
