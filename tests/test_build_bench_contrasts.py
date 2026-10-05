"""CONTRASTES_SSL.md lists every paired contrast of the bench from committed JSON, with the honesty rules."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_bench_contrasts as bc  # noqa: E402
from dermssl.bench import cli, contrasts_doc as D, contrasts_load as L, ft_contrasts as F  # noqa: E402

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
    fixed = 120 + 6 * 6 + 48
    isic = sum(len(c) for ds in ('isic2019', 'isic2020') for c in L.isic_all(REPO, ds).values())
    assert len(items) == fixed + isic + len(F.contrasts(REPO))


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


def test_ham_mcc_pairs_are_declared_unavailable_not_invented():
    text = _text().split("### HAM10000: MCC puntual")[1].split("###")[0]
    assert "NO es un contraste" in text and "SIN IC" in text and "1d7e447" in text


def test_pending_contrasts_carry_reason_and_eta():
    block = _text().split("### Contrastes pendientes")[1]
    for part in ("D3 − D2", "A71-limpio", "B71-limpio contra B71 original", "MCC en HAM10000"):
        assert part in block
    assert "~10:30 UTC" in block and "~15:00 UTC" in block


def test_interval_only_table_never_claims_a_pairwise_win():
    for ds in ("isic2019", "isic2020"):
        table = D.interval_table(REPO, ds)
        assert len(table.rows) == 8
        assert all("no es un test pareado" in r[2] or "no se afirma" in r[2] for r in table.rows)


def test_numbers_use_decimal_comma():
    assert not POINT.findall(_text())
