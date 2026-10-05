"""The bench tables are generated from committed JSON and checked for freshness."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_bench_tables as bb  # noqa: E402

OUT = REPO / bb.OUT
TABLES = ["TABLA_HAM10000", "TABLA_ISIC2019", "TABLA_ISIC2020"]
POINT = re.compile(r"(?<![\w./])\d+\.\d{4}(?![\w/])")


def _md(name: str) -> str:
    return (OUT / f"{name}.md").read_text(encoding="utf-8")


def test_committed_tables_are_fresh():
    assert bb.main(["--check"]) == 0


def test_check_detects_a_stale_table(tmp_path):
    flags = ["--out-root", str(tmp_path), "--skip-png"]
    assert bb.main(["--check", *flags]) == 1
    assert bb.main(flags) == 0
    assert bb.main(["--check", *flags]) == 0
    (tmp_path / "TABLA_HAM10000.md").write_text("edited by hand", encoding="utf-8")
    assert bb.main(["--check", *flags]) == 1


def test_check_requires_the_figures_unless_skipped(tmp_path):
    assert bb.main(["--out-root", str(tmp_path), "--skip-png"]) == 0
    assert bb.main(["--check", "--out-root", str(tmp_path)]) == 1
    assert bb.main(["--check", "--out-root", str(tmp_path), "--skip-png"]) == 0


def test_every_expected_file_exists():
    for name in TABLES:
        assert (OUT / f"{name}.md").is_file() and (OUT / "latex" / f"{name}.tex").is_file()


@pytest.mark.parametrize("name", TABLES)
def test_numbers_use_decimal_comma(name):
    for text in (_md(name), (OUT / "latex" / f"{name}.tex").read_text(encoding="utf-8")):
        assert not POINT.findall(text), POINT.findall(text)[:3]
        assert re.search(r"\b0,\d{4}\b", text)


@pytest.mark.parametrize("name", TABLES)
def test_glossary_comes_before_any_table_and_splits_are_stated(name):
    text = _md(name)
    assert text.index("## Glosario") < text.index("|")
    assert "MCC (coeficiente de correlación de Matthews)" in text
    assert "solo de intervalos" in text and "n = 2.004" in text and "n = 2.014" in text
    assert "elige su mejor época sobre val" in text and "optimista" in text


def _section(text: str, title: str) -> list[str]:
    """Table lines of every ``### <title>`` subsection."""
    out = []
    for part in text.split(f"### {title}")[1:]:
        out += [ln for ln in part.split("\n##")[0].splitlines() if re.match(r"\| (\d+|—) \|", ln)]
    return out


@pytest.mark.parametrize("name", TABLES)
def test_every_data_row_names_an_existing_source_path(name):
    rows = _section(_md(name), "Condición")
    assert len(rows) >= 13
    for line in rows:
        paths = re.findall(r"`([^`]+)`", line.split("|")[-2])
        assert paths and all((REPO / p).is_file() for p in paths), line


@pytest.mark.parametrize("name", TABLES)
def test_metric_tables_are_sorted_by_macro_f1_inside_each_block(name):
    for section in _md(name).split("\n## ")[1:]:
        if "### Métricas" not in section:
            continue
        lines = _section("### Métricas" + section.split("### Métricas")[1], "Métricas")
        scores = [float(ln.split("|")[7].split("[")[0].strip().replace(",", "."))
                  for ln in lines if re.match(r"\| \d+ \|", ln)]
        assert scores == sorted(scores, reverse=True)


def test_isic2020_finetune_block_says_not_run_and_pending_rows_are_listed():
    text = _md("TABLA_ISIC2020")
    assert "no ejecutado todavia" in text.split("## Con fine-tuning")[1]
    for name in TABLES:
        assert "B71-limpio" in _md(name) and "pendiente" in _md(name)
    assert "B71 original" in _md("TABLA_ISIC2019") and "B71 original" not in _md("TABLA_HAM10000")


def test_ham_table_has_the_proposed_rows_and_all_eight_baselines_in_both_blocks():
    text = _md("TABLA_HAM10000")
    for label in ("B71 (propuesto)", "A71 (ablación", "DINOv2 público, protocolo de B71"):
        assert label in text
    frozen, tuned = text.split("## Con fine-tuning")
    for model in ("AlexNet", "VGG19", "ResNet-50", "EfficientNet-B0", "ViT-B/16", "SimSiam ResNet-50",
                  "DINOv2 ViT-S/14", "DINOv2 ViT-B/14"):
        assert f"| {model} |" in frozen.split("### Métricas")[1] and f"| {model} |" in tuned


def test_latex_is_booktabs_and_escapes_percent():
    tex = (OUT / "latex" / "TABLA_HAM10000.tex").read_text(encoding="utf-8")
    assert r"\toprule" in tex and r"\bottomrule" in tex and r"\midrule" in tex
    assert not re.search(r"(?<!\\)%(?! Generado)", tex.replace("% Generado", "")), "unescaped percent"
