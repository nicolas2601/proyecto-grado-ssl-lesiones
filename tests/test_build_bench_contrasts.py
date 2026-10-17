"""CONTRASTES_SSL.md lists every paired contrast of the bench from committed JSON, with the honesty rules."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_bench_contrasts as bc  # noqa: E402
from dermssl.display_nomenclature import display_text  # noqa: E402
from dermssl.bench import cli, contrasts_doc as D, contrasts_load as L, e2_tables as E2, ft_contrasts as F  # noqa: E402

MD = REPO / cli.OUT / "CONTRASTES_SSL.md"
POINT = re.compile(r"(?<![\w./@])\d+\.\d{4}(?![\w/])")


def _text() -> str:
    return MD.read_text(encoding="utf-8")


def test_committed_files_are_fresh_and_staleness_is_detected(tmp_path):
    assert bc.main(["--check"]) == 0
    flags = ["--out-root", str(tmp_path)]
    assert bc.main(["--check", *flags]) == 1 and bc.main(flags) == 0 and bc.main(["--check", *flags]) == 0
    (tmp_path / "CONTRASTES_SSL.md").write_text("edited", encoding="utf-8")
    assert bc.main(["--check", *flags]) == 1


def test_every_contrast_family_of_the_bench_is_present():
    tables, items = D.build(REPO)
    titles = " ".join(t.title for t in tables)
    for part in ("B71 frente a los 8 baselines", "F4 sobre cada baseline", "cómputo S0", "cómputo S1", "cómputo S2",
                 "semilla de B* 42", "semilla de B* 43", "semilla de B* 44", "v7 y v7.1", "ISIC 2019: canónico",
                 "ISIC 2020: sensibilidad", "contra el fine-tuning", "solo intervalos", "Contrastes pendientes", "MCC puntual"):
        assert part in titles, part
    fixed = 160 + 6 * 8 + 48  # E3b: 120 contrasts of 3 metrics + 40 of MCC; E3 final: 6 evals x (6 + 2 MCC refit on the pod)
    isic = sum(len(c) for ds in ('isic2019', 'isic2020') for c in L.isic_all(REPO, ds).values())
    import json

    clean = len(json.loads((REPO / "resultados/banco_completo/ham_clean/ham_clean_val.json").read_text(encoding="utf-8"))["paired"])
    assert len(items) == fixed + isic + len(F.contrasts(REPO)) + len(E2.contrasts(REPO)) + clean


def test_multiplicity_warning_states_the_counts_from_the_data():
    _, items = D.build(REPO)
    text = _text()
    assert f"lista {len(items)} contrastes" in text and "sin corrección por multiplicidad" in text
    assert text.index("## Advertencia de multiplicidad") < text.index("### ")


def test_isic_contrasts_include_mcc_and_the_baselines_note_says_interval_only():
    text = _text()
    isic = text.split("### ISIC 2019: canónico")[1].split("###")[0]
    assert "| MCC |" in isic and "sin probabilidades" not in isic
    assert "solo se comparan intervalos" in text and "no es un test pareado" in text


def test_e3b_mcc_has_real_paired_contrasts_and_the_note_says_what_exists():
    text = _text().split("### HAM10000: MCC puntual")[1].split("###")[0]
    assert "NO es un contraste" not in text and "no traen contrastes" not in text
    assert "contraste pareado" in text and "IC 95 %" in text and "40 contrastes" in text
    assert "los 120 registros previos" in text and "idénticos" in text  # the count comes from the summary's rebuild block
    assert re.search(r"Prueba con segmentación final: \d+ contrastes de MCC en los JSON, 0 no disponibles", text)  # refit on the pod, all reproduced


def test_an_e3_mcc_cell_stored_as_unavailable_renders_unavailable_from_the_json(tmp_path):
    import json

    reason = "no disponible: el refit no reproduce los puntos registrados"
    rec = {"arm": "F4", "control": "B71", "metric": "mcc", "subset": "dark_tercile", "available": False, "reason": reason}
    (tmp_path / "e3.json").write_text(json.dumps({"contrasts": [rec]}), encoding="utf-8")
    [c] = L.f4_vs_b71(tmp_path, "e3.json")
    table = D._table("t", [c], "n", "l")
    assert c.metric == "mcc" and c.unavailable == reason
    assert table.rows[0][3] == reason and table.rows[0][4] == "—" and table.rows[0][0].startswith("F4")


def test_e3b_tables_carry_mcc_like_any_other_metric():
    text = _text()
    above = text.split("### HAM10000: Proposed sin segmentación frente a los 8 baselines")[1].split("###")[0]
    assert "(4 variantes × 4 métricas)" in text and above.count("| MCC |") == 32
    raised = text.split(display_text("### HAM10000: F4 sobre cada baseline congelado"))[1].split("###")[0]
    assert raised.count("| MCC |") == 8


def test_no_contrast_is_pending_once_every_result_is_committed():
    block = _text().split("### Contrastes pendientes")[1]
    assert "D3 − D2" not in block and "~10:30" not in block and "~15:00" not in block
    assert "MCC en HAM10000" not in block  # E3 and E3b both carry their MCC contrasts now
    assert "ham_clean_val.json" not in block and block.count("sin ETA") == 0  # HAM clean is measured too
    assert "No queda ningún contraste pendiente" in block


def test_e2_contrasts_are_listed_with_the_primary_one_first():
    text = _text().split(display_text("### D0-D3 (E2): contrastes pareados", context="bench", code_labels=True))[1]
    assert "predicción primaria registrada" in text and display_text("| D3 | D2 | F1 macro |") in text


def test_interval_only_table_never_claims_a_pairwise_win():
    for ds in ("isic2019", "isic2020"):
        table = D.interval_table(REPO, ds)
        assert len(table.rows) == 8
        assert all("no es un test pareado" in r[2] or "no se afirma" in r[2] for r in table.rows)


def test_numbers_use_decimal_comma():
    assert not POINT.findall(_text())


def test_the_e3_f4_mcc_reading_is_generated_from_the_cells_and_keeps_f4_exploratory():
    import json

    near_zero = 0.0025  # the bound documented in contrasts_doc.NEAR_ZERO, restated so a silent change fails here
    rels = dict.fromkeys([*(rel for _, _, rel in D.COMPUTES), *L.SEED_FILES.values()])
    cells = [c for rel in rels for c in json.loads((REPO / rel).read_text(encoding="utf-8"))["contrasts"]
             if c["metric"] == "mcc"]
    hit = [c for c in cells if c["ci95_low"] > 0 or c["ci95_high"] < 0]
    near = [c for c in hit if min(abs(c["ci95_low"]), abs(c["ci95_high"])) < near_zero]
    text = _text().split("### HAM10000: MCC puntual")[1].split("###")[0]
    assert f"IC que excluye 0 en {len(hit)} de {len(cells)} celdas; {len(near)} con límite inferior ≈ 0" in text
    assert display_text("F4 sigue siendo exploratorio") in text and "regla de lectura de la robustez" in text
