"""Contrasts-only rebuild of E3b: MCC is ADDED, every pre-existing record is reproduced identically."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from dermssl import e3_stages as stages
from dermssl import e3_store as st
from dermssl import paired_bootstrap as pb
from dermssl import run_environment as env
from dermssl.e3b import contrasts as c3
from dermssl.e3b import rebuild as rb

OLD = ("macro_f1", "balanced_accuracy", "mel_recall")
CLASSES = ("akiec", "mel", "nv", "vasc")
N_RES = 25


def _data():
    rng = np.random.RandomState(5)
    n = 240
    y = np.arange(n) % 4
    groups = np.array([f"L{i // 2}" for i in range(n)])

    def proba(signal):
        a = rng.dirichlet(np.ones(4), n) + np.eye(4)[y] * signal
        return a / a.sum(1, keepdims=True)

    arms = {"m": proba(0.2), "m+F4": proba(0.5), "B71": proba(0.7), "B71+F4": proba(0.8),
            "B71_tta": proba(0.75), "B71+F4_tta": proba(0.85)}
    return y, groups, arms


def _legacy_records(y, groups, arms, pairs):
    """What the run did before 1d7e447: the same bootstrap with the three original metrics only."""
    fns = {m: f for m, f in stages._metric_fns(y, 4, 1).items() if m != "mcc"}
    out = []
    for arm, control, family in pairs:
        for metric, fn in fns.items():
            rec = pb.contrast_record(arm=arm, control=control, metric=metric, probe=c3.PROBE_NAME, fraction=1.0,
                                     seed=42, y_true=y, arm_proba=arms[arm], control_proba=arms[control],
                                     n_resamples=N_RES, rng_seed=stages.BOOTSTRAP_SEED, metric_fn=fn, groups=groups)
            out.append({**rec, "family": family, "n_resamples": N_RES, "rng_seed": stages.BOOTSTRAP_SEED})
    return out


def _summary(old):
    return {"stage": "e3b", "requested_models": ["m"], "classes": list(CLASSES), "contrasts": old, "complete": True}


def test_rebuild_reproduces_the_legacy_records_and_adds_only_mcc():
    y, groups, arms = _data()
    pairs = c3.pairs_for("m")
    old = _legacy_records(y, groups, arms, pairs)
    new = rb.rebuild_records(y, arms, groups, ["m"], classes=CLASSES, n_resamples=N_RES)
    report = rb.verify_preserved(old, new)
    assert report == {"compared": len(old), "equal": len(old), "mcc_added": len(pairs)}
    assert [r["metric"] for r in new if r["metric"] == "mcc"] == ["mcc"] * len(pairs)
    assert [r for r in new if r["metric"] != "mcc"] == old


def test_verify_refuses_a_changed_ci_bound():
    y, groups, arms = _data()
    old = _legacy_records(y, groups, arms, c3.pairs_for("m"))
    new = rb.rebuild_records(y, arms, groups, ["m"], classes=CLASSES, n_resamples=N_RES)
    tampered = copy.deepcopy(old)
    tampered[3]["ci95_high"] += 1e-9
    with pytest.raises(rb.RebuildMismatchError, match="macro_f1|balanced_accuracy|mel_recall"):
        rb.verify_preserved(tampered, new)


def test_verify_refuses_wrong_groups():
    y, groups, arms = _data()
    old = _legacy_records(y, groups, arms, c3.pairs_for("m"))
    wrong = np.array([f"L{i}" for i in range(len(y))])  # each image its own group
    new = rb.rebuild_records(y, arms, wrong, ["m"], classes=CLASSES, n_resamples=N_RES)
    with pytest.raises(rb.RebuildMismatchError):
        rb.verify_preserved(old, new)


def test_verify_refuses_a_missing_or_extra_old_record():
    y, groups, arms = _data()
    old = _legacy_records(y, groups, arms, c3.pairs_for("m"))
    new = rb.rebuild_records(y, arms, groups, ["m"], classes=CLASSES, n_resamples=N_RES)
    with pytest.raises(rb.RebuildMismatchError):
        rb.verify_preserved(old[:-1], new)


def test_groups_come_from_the_metadata_and_a_missing_id_is_an_error(tmp_path):
    csv = tmp_path / "meta.csv"
    csv.write_text("lesion_id,image_id,dx\nHAM_1,ISIC_a,nv\nHAM_1,ISIC_b,nv\nHAM_2,ISIC_c,mel\n", encoding="utf-8")
    assert list(rb.groups_from_metadata(csv, ["ISIC_c", "ISIC_a"])) == ["HAM_2", "HAM_1"]
    with pytest.raises(rb.RebuildMismatchError, match="ISIC_zzz"):
        rb.groups_from_metadata(csv, ["ISIC_a", "ISIC_zzz"])


def _write_dir(tmp_path, old):
    y, groups, arms = _data()
    out = tmp_path / "e3b"
    out.mkdir()
    val_ids = np.array([f"ISIC_{i:04d}" for i in range(len(y))])
    np.savez(out / "e3b_probas.npz", y_val=y, val_ids=val_ids, **arms)
    manifest = {"file": "e3b_probas.npz", "sha256": st.file_sha256(out / "e3b_probas.npz"),
                "val_ids_sha256": rb.fc.fingerprint_ids(list(val_ids))}
    (out / "e3b_probas.json").write_text(json.dumps(manifest), encoding="utf-8")
    st.write_json_atomic(out / "e3b_summary.json", _summary(old))
    env.write_sidecar(out / "e3b_summary.json", environment={"schema": 1})
    meta = tmp_path / "meta.csv"
    rows = "\n".join(f"{g},{i},nv" for g, i in zip(groups, val_ids))
    meta.write_text("lesion_id,image_id,dx\n" + rows + "\n", encoding="utf-8")
    return out, meta


def test_rebuild_dir_rewrites_only_the_contrasts_with_a_fresh_sidecar(tmp_path):
    y, groups, arms = _data()
    old = _legacy_records(y, groups, arms, c3.pairs_for("m"))
    out, meta = _write_dir(tmp_path, old)
    before = json.loads((out / "e3b_summary.json").read_text())
    report = rb.rebuild_dir(out, meta, n_resamples=N_RES, mel_name="mel", models=["m"])
    after = json.loads((out / "e3b_summary.json").read_text())
    assert report["compared"] == report["equal"] == len(old) and report["mcc_added"] == 5
    assert [r for r in after["contrasts"] if r["metric"] != "mcc"] == old
    assert {k: v for k, v in after.items() if k not in ("contrasts", "contrasts_rebuild")} == \
        {k: v for k, v in before.items() if k != "contrasts"}
    assert env.sidecar_problems(env.sidecar_path(out / "e3b_summary.json")) == []
    assert hashlib.sha256((out / "e3b_summary.json").read_bytes()).hexdigest() == \
        json.loads(env.sidecar_path(out / "e3b_summary.json").read_text())["sha256"]


def test_rebuild_dir_leaves_everything_untouched_when_an_old_record_differs(tmp_path):
    y, groups, arms = _data()
    old = _legacy_records(y, groups, arms, c3.pairs_for("m"))
    old[0]["point_delta"] += 0.5
    out, meta = _write_dir(tmp_path, old)
    snapshot = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(rb.RebuildMismatchError):
        rb.rebuild_dir(out, meta, n_resamples=N_RES, mel_name="mel", models=["m"])
    assert {p.name: p.read_bytes() for p in out.iterdir()} == snapshot


def test_rebuild_dir_is_idempotent_on_an_already_rebuilt_summary(tmp_path):
    y, groups, arms = _data()
    out, meta = _write_dir(tmp_path, _legacy_records(y, groups, arms, c3.pairs_for("m")))
    rb.rebuild_dir(out, meta, n_resamples=N_RES, mel_name="mel", models=["m"])
    first = (out / "e3b_summary.json").read_bytes()
    rb.rebuild_dir(out, meta, n_resamples=N_RES, mel_name="mel", models=["m"])
    assert (out / "e3b_summary.json").read_bytes() == first


def test_the_script_aborts_with_rc2_and_writes_nothing_on_a_mismatch(tmp_path, monkeypatch, capsys):
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import e3b_contrasts_rebuild as script

    y, groups, arms = _data()
    old = _legacy_records(y, groups, arms, c3.pairs_for("m"))
    old[1]["ci95_low"] -= 0.01
    out, meta = _write_dir(tmp_path, old)
    before = (out / "e3b_summary.json").read_bytes()
    assert script.main(["--ham-metadata", str(meta), "--dir", str(out), "--n-resamples", str(N_RES)]) == 2
    assert "ABORT" in capsys.readouterr().out and (out / "e3b_summary.json").read_bytes() == before
