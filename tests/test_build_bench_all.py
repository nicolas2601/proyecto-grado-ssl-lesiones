"""One entry point regenerates and checks the whole bench; a stale file anywhere fails it."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_bench_all as ba  # noqa: E402

EXPECTED = ("TABLA_HAM10000.md", "TABLA_ISIC2019.md", "TABLA_ISIC2020.md", "TABLA_ABLACION.md", "CONTRASTES_SSL.md",
            "latex/TABLA_HAM10000.tex", "latex/TABLA_ISIC2019.tex", "latex/TABLA_ISIC2020.tex",
            "latex/TABLA_ABLACION.tex", "latex/CONTRASTES_SSL.tex", "confusion/manifest.json", "confusion/README.md")


def test_every_generator_is_wired():
    assert [n for n, _ in ba.GENERATORS] == ["build_bench_tables.py", "build_bench_ablation.py",
                                             "build_bench_contrasts.py"]


def test_committed_bench_is_fresh(capsys):
    assert ba.main(["--check"]) == 0
    assert "stale" not in capsys.readouterr().err


def test_write_then_check_roundtrip_and_every_expected_file(tmp_path):
    flags = ["--out-root", str(tmp_path), "--skip-png"]
    assert ba.main(["--check", *flags]) == 1
    assert ba.main(flags) == 0 and ba.main(["--check", *flags]) == 0
    assert all((tmp_path / name).is_file() for name in EXPECTED)


def test_a_stale_file_in_any_generator_fails_the_check(tmp_path, capsys):
    flags = ["--out-root", str(tmp_path), "--skip-png"]
    assert ba.main(flags) == 0
    for name in ("TABLA_ISIC2019.md", "TABLA_ABLACION.md", "CONTRASTES_SSL.md"):
        path = tmp_path / name
        original = path.read_text(encoding="utf-8")
        path.write_text(original + "\nedited by hand\n", encoding="utf-8")
        assert ba.main(["--check", *flags]) == 1
        assert name in capsys.readouterr().err
        path.write_text(original, encoding="utf-8")
    assert ba.main(["--check", *flags]) == 0
