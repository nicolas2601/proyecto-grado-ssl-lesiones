"""Pure rules of the HAM10000 clean-weights sensitivity run (no torch, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import c1_split_identity as si
from dermssl import ham_clean as hc
from dermssl.isic_eval import protocol as pr

IDS = [f"ISIC_{k:07d}" for k in range(10)]
Y = np.arange(10) % 3
ENTRIES = {"B71": pr.B71, "A71": pr.A71, "public_B71": pr.CONFIGS["PUBLIC_B71"], "public_A71": pr.CONFIGS["PUBLIC_A71"]}


def _ref(split="val"):
    return {"split": split, "eval_ids_sha256": si.ids_sha256(IDS), "entries": dict(ENTRIES)}


def test_the_five_registered_contrasts_and_mcc_last():
    assert hc.CONTRASTS == (("B71_clean", "public_B71"), ("B71_clean", "B71"), ("A71_clean", "public_A71"),
                            ("A71_clean", "A71"), ("B71_clean", "A71_clean"))
    assert hc.CONTRAST_METRICS[-1] == "mcc" and "macro_f1" in hc.CONTRAST_METRICS


def test_clean_entries_are_the_registered_configs_of_the_original_selection():
    assert hc.clean_entries(ENTRIES) == {"B71_clean": pr.B71, "A71_clean": pr.A71}
    with pytest.raises(hc.HamCleanError, match="B71"):
        hc.clean_entries({**ENTRIES, "B71": "B_s42_a0.6@224|cls|none|id"})


def test_reference_must_share_split_ids_labels_and_configs():
    hc.check_reference(_ref(), Y, split="val", eval_ids=IDS, y=Y, entries=ENTRIES)
    with pytest.raises(hc.HamCleanError, match="split"):
        hc.check_reference(_ref("holdout"), Y, split="val", eval_ids=IDS, y=Y, entries=ENTRIES)
    with pytest.raises(hc.HamCleanError, match="ids"):
        hc.check_reference(_ref(), Y, split="val", eval_ids=IDS[::-1], y=Y, entries=ENTRIES)
    with pytest.raises(hc.HamCleanError, match="etiquetas"):
        hc.check_reference(_ref(), Y[::-1], split="val", eval_ids=IDS, y=Y, entries=ENTRIES)
    bad = _ref()
    bad["entries"]["public_B71"] = "public@224|cls|none|id"
    with pytest.raises(hc.HamCleanError, match="public_B71"):
        hc.check_reference(bad, Y, split="val", eval_ids=IDS, y=Y, entries=ENTRIES)


def test_real_val_demands_cuda_and_batch_64_and_the_rehearsal_does_not():
    assert hc.compute_block("cuda", 64, rehearsal=False)["canonical"] is True
    for device, batch in (("cpu", 64), ("cuda", 16)):
        with pytest.raises(hc.HamCleanError, match="canonico"):
            hc.compute_block(device, batch, rehearsal=False)
    assert hc.compute_block("cpu", 8, rehearsal=True)["canonical"] is False


def test_probas_for_the_contrasts_are_checked_for_shape():
    n, k = len(Y), 3
    clean = {"B71_clean": np.ones((n, k)), "A71_clean": np.ones((n, k))}
    ref = {name: np.ones((n, k)) for name in ("B71", "A71", "public_B71", "public_A71")}
    assert set(hc.contrast_probas(clean, ref)) == {"B71_clean", "A71_clean", "B71", "A71", "public_B71", "public_A71"}
    with pytest.raises(hc.HamCleanError, match="forma"):
        hc.contrast_probas({**clean, "A71_clean": np.ones((n - 1, k))}, ref)


def test_the_document_declares_what_it_is_and_what_it_assumes():
    doc = hc.declarations(rehearsal=False)
    assert doc["label"].startswith("analisis de sensibilidad posterior; no es seleccion")
    assert "val_read.lock de E3 no se reinterpreta" in doc["label"]
    assert "ORIGINALES" in doc["assumptions"][0] and "sin cambios" in doc["assumptions"][0]
    assert doc["rehearsal"] is False and "discard" not in doc
    rehearsal = hc.declarations(rehearsal=True)
    assert rehearsal["rehearsal"] is True and rehearsal["discard"] is True


def test_paths_live_under_the_committed_ham_clean_folder():
    assert hc.RESULT_JSON == "resultados/banco_completo/ham_clean/ham_clean_val.json"
    assert hc.RESULT_NPZ == "resultados/banco_completo/ham_clean/ham_clean_val.probas.npz"
