"""TABLA_ABLACION.md is generated from committed JSON, checked for freshness, honest about what is pending."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_bench_ablation as ba  # noqa: E402
from dermssl.display_nomenclature import display_text  # noqa: E402
from dermssl.bench import ablation_data as A  # noqa: E402
from dermssl.bench import cli  # noqa: E402

MD = REPO / cli.OUT / "TABLA_ABLACION.md"
TEX = REPO / cli.OUT / "latex" / "TABLA_ABLACION.tex"
POINT = re.compile(r"(?<![\w./@])\d+\.\d{4}(?![\w/])")


def _text() -> str:
    return MD.read_text(encoding="utf-8")


def test_committed_files_are_fresh():
    assert ba.main(["--check"]) == 0


def test_check_detects_a_stale_file(tmp_path):
    flags = ["--out-root", str(tmp_path)]
    assert ba.main(["--check", *flags]) == 1
    assert ba.main(flags) == 0 and ba.main(["--check", *flags]) == 0
    (tmp_path / "TABLA_ABLACION.md").write_text("edited", encoding="utf-8")
    assert ba.main(["--check", *flags]) == 1


def test_glossary_comes_first_and_numbers_use_decimal_comma():
    text = _text()
    assert text.index("## Glosario") < text.index("|")
    assert not POINT.findall(text) and not POINT.findall(TEX.read_text(encoding="utf-8"))


def test_every_pipe_table_row_has_the_header_column_count():
    for block in _text().split("\n\n"):
        rows = [ln for ln in block.splitlines() if ln.startswith("|")]
        if rows:
            cols = {len(re.split(r"(?<!\\)\|", ln)) for ln in rows}
            assert len(cols) == 1, rows[0]


def test_f4_sensitivity_and_seed_reading_follow_the_registered_rules():
    text = _text()
    assert "0,7106" in text and "0,7141" in text and "0,7089" in text
    assert display_text("Lo cumple en 2 de 3: sin diferencia demostrable de forma robusta; la dirección favorece a F4") in text
    assert "0 de 3 semillas: lectura «none»" in text and "+0,0217" in text


def test_e3_holdout_table_lists_f1_to_f5_and_val_only_b71_and_f4():
    text = _text()
    for variant in ("F1", "F2", "F2_448", "F3", "F4", "F5"):
        assert display_text(f"| {variant} |") in text.split("### F1-F5")[1].split(display_text("### F4 frente"))[0]
    assert "por diseño" in text


def test_e2_results_replace_the_pending_section_and_come_from_the_json():
    import json

    text = _text()
    assert "pendiente" not in text.split("### D0-D3")[1].split("###")[0] and "~10:30" not in text
    doc = json.loads((REPO / "resultados/banco_completo/pad_da/pad_e2_eval.json").read_text(encoding="utf-8"))
    block = text.split("### D0-D3 en PAD")[1]
    for arm in ("D0", "D1", "D2", "D3"):
        row = next(ln for ln in block.splitlines() if ln.startswith(display_text(f"| {arm} |")))
        for metric in ("macro_f1", "balanced_accuracy", "accuracy", "auc", "ece", "mcc"):
            assert f"{doc['arms'][arm][metric]:.4f}".replace(".", ",") in row
        assert f"{doc['outside_rate'][arm]:.4f}".replace(".", ",") in row
    assert "leído una sola vez (lock)" in block and "batch 64 + autocast" in block and "semilla 20260923" in block
    assert f"cumplida = {'sí' if doc['prediction_d3_minus_d2_met'] else 'no'}" in block
    p = doc["primary_contrast"]
    assert f"{p['point_delta']:+.4f}".replace(".", ",") in block
    assert f"{doc['n_images']}".replace(",", ".") in block.replace(".", "") or str(doc["n_patients"]) in block


def test_clean_table_does_not_compare_clean_with_the_batch16_original():
    clean = A.clean_table(REPO)
    assert "Predicción registrada" in clean.note and "mismo cómputo canónico" in clean.note
    assert "pendiente" not in clean.note
    assert not any(r[1].startswith("B71-limpio") and "batch 16" in r[-1] for r in clean.rows)
