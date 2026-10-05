"""ISIC 2020 fine-tuning rows: one per baseline, measured when its JSON exists, pending otherwise; epoch curves."""

from __future__ import annotations

import json

import pytest

from dermssl.bench import ft_curve as C
from dermssl.bench import rows as R

REPO = R.REPO
DS = R.DATASETS["isic2020"]
LOG = REPO / "resultados/banco_completo/ft_isic2020/logs/ft_dinov2_vitb14.epochs.log"


def _ft() -> dict[str, R.Row]:
    return {r.name: r for r in R.dataset_rows(REPO, DS) if r.block == R.FINETUNE}


def test_one_row_per_baseline_measured_or_pending_with_reason_and_eta():
    rows = _ft()
    assert len(rows) == 8
    for name, row in rows.items():
        if row.status == R.OK:
            assert row.finetune and "sesgo optimista" in row.note
        else:
            assert row.status == R.PENDING and not row.values and "ETA" in row.note and "GPU1" in row.note


def test_vitb14_row_is_the_json_number():
    row = _ft()["DINOv2 ViT-B/14"]
    doc = json.loads((REPO / row.sources[0]).read_text(encoding="utf-8"))
    probe = doc["probes"]["finetune"]
    assert row.status == R.OK and row.n_val == doc["n_val"] == 6660
    assert row.values["balanced_accuracy"].point == probe["balanced_accuracy"]
    assert row.values["macro_f1"].point == probe["macro_f1"]
    assert "mejor época 3" in row.note


def test_a_missing_json_turns_into_a_pending_row_and_back(tmp_path):
    rows = R.baseline_rows(tmp_path_repo(tmp_path), DS)
    ft = [r for r in rows if r.block == R.FINETUNE]
    assert len(ft) == 8 and all(r.status == R.PENDING for r in ft)


def tmp_path_repo(tmp_path):
    """A repo with only the frozen baselines of ISIC 2020 (no fine-tuning JSON)."""
    import shutil

    shutil.copytree(REPO / DS.baselines, tmp_path / DS.baselines)
    return tmp_path


REAL = [
    "epoca 1/8 loss=0.5195 val_bal_acc=0.5000 mejor=0.5000 (117s)",
    "epoca 3/8 loss=0.4442 val_bal_acc=0.6114 mejor=0.6114 (349s)",
    "epoca 8/8 loss=0.3829 val_bal_acc=0.5315 mejor=0.6114 (931s)",
]


def test_parser_reads_the_real_log_lines():
    got = C.parse_log("\n".join(["cabecera", *REAL, "dinov2 @ isic2020"]))
    assert [(c.epoch, c.train_loss, c.val_balanced_accuracy) for c in got] == [
        (1, 0.5195, 0.5), (3, 0.4442, 0.6114), (8, 0.3829, 0.5315)]


def test_committed_log_has_8_epochs_with_the_known_endpoints():
    got = C.parse_log(LOG.read_text(encoding="utf-8"))
    assert [c.epoch for c in got] == list(range(1, 9))
    assert (got[0].train_loss, got[0].val_balanced_accuracy) == (0.5195, 0.5)
    assert (got[2].train_loss, got[2].val_balanced_accuracy) == (0.4442, 0.6114)
    assert (got[7].train_loss, got[7].val_balanced_accuracy) == (0.3829, 0.5315)


def test_curve_prefers_the_json_then_the_log_then_says_not_recorded(tmp_path):
    doc = {"curva_epocas": [{"epoca": 1, "train_loss": 0.5, "val_balanced_accuracy": 0.6}]}
    curve, source = C.curve_for(tmp_path, "m", doc)
    assert source == "json" and curve[0].val_balanced_accuracy == 0.6
    assert C.curve_for(tmp_path, "m", {}) == ([], "no registrada")
    log = tmp_path / C.LOG_DIR / "ft_m.epochs.log"
    log.parent.mkdir(parents=True)
    log.write_text(REAL[0], encoding="utf-8")
    curve, source = C.curve_for(tmp_path, "m", {})
    assert source == "log" and curve[0].epoch == 1


def test_log_curve_matches_the_json_best_epoch():
    doc = json.loads((REPO / _ft()["DINOv2 ViT-B/14"].sources[0]).read_text(encoding="utf-8"))
    curve, _ = C.curve_for(REPO, "dinov2_vitb14", doc)
    best = max(curve, key=lambda c: c.val_balanced_accuracy)
    assert best.epoch == doc["mejor_epoca"]


def test_section_has_the_literal_notes_and_a_row_per_model():
    text = "\n".join(C.section(REPO))
    assert "Receta existente sin cambios; con 1,8 % de positivos la bal_acc en val oscila entre épocas; se reporta la mejor época en val (optimista)." in text
    assert "Esta ronda corrió en nuestro pod (GPU RTX PRO 4000), no en el banco de Karen." in text
    assert text.count("curva no registrada") + text.count("pendiente") >= 7
    assert "0,6114" in text


@pytest.mark.parametrize("bad", ["epoca x/8 loss=a", "", "epoca 1/8 loss=0.5"])
def test_parser_ignores_malformed_lines(bad):
    assert C.parse_log(bad) == []
