"""The published tables must still match the bank they came from.

build_results_readme.py generates the classification and tone tables in
resultados/README.md. Generating them once is not enough: a rerun changes
bank.json and the committed document keeps the old figure, which is exactly
the drift the generator exists to prevent. This test is that guard in CI.

It skips while the bank has not been committed yet - there is nothing to
compare against - and starts failing the moment the data lands.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_results_readme as gen  # noqa: E402

BANK_DIR = REPO / "resultados" / "clasificacion"
README = REPO / "resultados" / "README.md"


def test_the_published_tables_match_the_committed_bank():
    bank_file = BANK_DIR / "bank.json"
    if not bank_file.is_file():
        pytest.skip(f"todavia no hay banco commiteado en {bank_file.relative_to(REPO)}")

    import json

    bank = json.loads(bank_file.read_text(encoding="utf-8"))
    strata_file = BANK_DIR / "strata.json"
    strata = json.loads(strata_file.read_text(encoding="utf-8")) if strata_file.is_file() else {}

    current = README.read_text(encoding="utf-8")
    expected = gen.replace_block(current, gen.render(bank, strata))

    assert current == expected, (
        "resultados/README.md quedo desactualizado respecto de bank.json.\n"
        "Corre: python scripts/build_results_readme.py "
        "--bank resultados/clasificacion --readme resultados/README.md"
    )


def test_the_document_still_has_the_markers_the_generator_needs():
    current = README.read_text(encoding="utf-8")
    assert gen.BEGIN in current and gen.END in current, (
        "alguien borro los marcadores; el generador no puede escribir las tablas "
        "y el documento se vuelve transcripcion a mano"
    )
