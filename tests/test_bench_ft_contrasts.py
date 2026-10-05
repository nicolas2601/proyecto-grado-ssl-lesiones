"""B71-limpio vs the ISIC 2020 fine-tuned baselines: id pairing, Holm, pending / unpaired models, labels."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dermssl.bench import ft_contrasts as F
from dermssl.bench import rows as R
from dermssl.bench import tables


def _arrays(n=120, seed=0, shuffle=False, offset=0.0):
    y = (np.random.RandomState(0).rand(n) < 0.3).astype(np.int64)  # same labels for every model
    score = np.clip(0.5 * y + np.random.RandomState(seed + 100).rand(n) * 0.6 + offset, 0.01, 0.99)
    proba = np.stack([1 - score, score], axis=1)
    ids = np.array([f"ISIC_{i:05d}" for i in range(n)])
    groups = np.array([f"P{i // 2}" for i in range(n)])
    out = {"val_ids": ids, "y": y, "proba": proba, "groups": groups}
    if shuffle:
        order = np.random.RandomState(9).permutation(n)
        out = {k: v[order] for k, v in out.items()}
    return out


def test_pairing_is_by_id_not_by_position():
    a, c = _arrays(), _arrays(seed=0, shuffle=True)
    first = F.compute(a, c, "ft_x", n_resamples=40)
    again = F.compute(a, _arrays(seed=0), "ft_x", n_resamples=40)
    assert first == again


def test_a_different_id_set_aborts_with_a_clear_message():
    a, c = _arrays(), _arrays()
    c["val_ids"] = np.array([f"OTHER_{i}" for i in range(len(c["y"]))])
    with pytest.raises(F.AlignmentError, match="no son el mismo conjunto"):
        F.compute(a, c, "ft_x", n_resamples=10)
    short = {k: v[:-1] for k, v in _arrays().items()}
    with pytest.raises(F.AlignmentError):
        F.align(_arrays(), short)


def test_label_patient_and_duplicate_mismatches_abort():
    bad_y = _arrays()
    bad_y["y"] = 1 - bad_y["y"]
    with pytest.raises(F.AlignmentError, match="etiquetas"):
        F.align(_arrays(), bad_y)
    bad_g = _arrays()
    bad_g["groups"] = np.array(["Z"] * len(bad_g["y"]))
    with pytest.raises(F.AlignmentError, match="pacientes"):
        F.align(_arrays(), bad_g)
    dup = _arrays()
    dup["val_ids"][1] = dup["val_ids"][0]
    with pytest.raises(F.AlignmentError, match="repetidos"):
        F.align(dup, _arrays())


def test_compute_keeps_the_four_metrics_with_the_isic_eval_schema_and_a_bounded_p():
    rec = F.compute(_arrays(), _arrays(seed=1, offset=-0.1), "ft_x", n_resamples=60)
    assert [r["metric"] for r in rec["paired"]] == list(F.METRICS) and rec["unit"] == "patient_id"
    for r in rec["paired"]:
        assert r["arm"] == "B71__clean" and r["control"] == "ft_x"
        assert 1 / 60 <= r["p_boot"] <= 1
        assert r["excludes_zero"] == (r["ci95_low"] > 0 or r["ci95_high"] < 0)


def test_p_bootstrap_floor_and_cap():
    assert F.p_bootstrap(np.ones(1000)) == pytest.approx(1 / 1000)
    assert F.p_bootstrap(np.array([-1.0, 1.0] * 500)) == 1.0


def test_holm_matches_the_textbook_and_treats_absent_models_as_p_one():
    assert F.holm([0.01, 0.04, 0.03], family=3) == pytest.approx([0.03, 0.06, 0.06])
    assert F.holm([None, 0.01], family=8) == [None, pytest.approx(0.08)]
    assert F.holm([0.5, 0.9], family=8) == pytest.approx([1.0, 1.0])


def _tmp_repo(tmp_path, models, unpaired=None):
    arm = _arrays()
    cell = tmp_path / F.isic_rows.cell_path(F.isic_rows.CANONICAL_DIR, "isic2020", F.ARM_CELL)
    cell.parent.mkdir(parents=True)
    cell.write_text(json.dumps({"result": {"probas_sha256": "x"}}), encoding="utf-8")
    np.savez(str(cell).replace(".json", ".probas.npz"), **arm)
    pdir = tmp_path / F.PROBAS_DIR
    pdir.mkdir(parents=True)
    for i, m in enumerate(models):
        np.savez(pdir / f"{m}.probas.npz", **_arrays(seed=i + 1, shuffle=True))
    for m, why in (unpaired or {}).items():
        (pdir / f"{m}.probas.json").write_text(json.dumps({"no_pareado": why}), encoding="utf-8")
    return tmp_path


def test_missing_npz_is_pending_with_reason_and_unpaired_model_is_labelled(tmp_path):
    repo = _tmp_repo(tmp_path, ["dinov2_vitb14", "resnet50"], {"resnet50": "ids no alineados"})
    assert F.write_all(repo) == ["dinov2_vitb14"] and F.stale_models(repo) == []
    by_metric, waiting, unpaired = F.entries(repo)
    assert len(waiting) == 6 and all("ETA" in w[1] for w in waiting)
    assert unpaired == [("ResNet-50 (fine-tuning)", "ids no alineados")]
    assert "no pareado: ids no alineados" in F.table(repo).note


def test_write_then_table_has_four_rows_per_paired_model_and_stale_detection(tmp_path):
    repo = _tmp_repo(tmp_path, ["dinov2_vitb14"])
    assert F.stale_models(repo) == ["dinov2_vitb14"]
    assert F.write_all(repo) == ["dinov2_vitb14"] and F.stale_models(repo) == []
    table = F.table(repo)
    assert [r[1] for r in table.rows] == [F.METRIC_ES[m] for m in F.METRICS]
    assert all(r[0] == "DINOv2 ViT-B/14 (fine-tuning)" for r in table.rows)
    np.savez(repo / F.probas_path("dinov2_vitb14"), **_arrays(seed=5))
    assert F.stale_models(repo) == ["dinov2_vitb14"]


def test_table_labels_post_hoc_optimism_reconstruction_and_p_definition():
    note = F.table(R.REPO).note
    for part in ("análisis posterior a la ejecución (post-hoc), no parte del registro original",
                 "reconstruidas por inferencia desde el checkpoint de la mejor época; verificadas contra las métricas "
                 "registradas", "mejor época elegida en val (optimista)", "B71-limpio no tiene fine-tuning",
                 "piso 1/1000", "Holm sobre los 8 modelos"):
        assert part in note, part
    md = "\n".join(tables.markdown(F.table(R.REPO)))
    assert "### B71-limpio contra el fine-tuning" in md
