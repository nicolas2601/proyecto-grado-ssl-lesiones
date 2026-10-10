"""Delivery package: every listed figure has its formats, caption and note; checksums and the SVG zip match."""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import package_delivery as package  # noqa: E402


def test_package_is_complete_and_consistent(tmp_path):
    out = tmp_path / "entrega"
    summary = package.build(REPO, out)
    assert summary["figures"] >= 31 and summary["svg"] == summary["figures"] - 1  # T2 is a table, not an SVG
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    with zipfile.ZipFile(out / "entrega_pau_svg.zip") as zf:
        assert all(n.endswith(".svg") for n in zf.namelist()) and len(zf.namelist()) == summary["svg"]
    index = (out / "INDICE.md").read_text(encoding="utf-8")
    assert "Pendientes" in index and "Requieren experimentos nuevos" in index and "ITA no equivale" in index


def test_missing_export_aborts(tmp_path):
    import pytest  # noqa: PLC0415

    (tmp_path / package.FIGURES).mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        package.collect(tmp_path)


def test_latex_escaping():
    assert package.latex("100 % & a_b") == r"100 \% \& a\_b"
