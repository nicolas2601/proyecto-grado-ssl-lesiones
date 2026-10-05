"""Shared ``--check`` / write driver of the bench generators (each one owns a set of text files)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

OUT = Path("resultados") / "banco_completo"


def stale(outputs: dict[Path, str], root: Path) -> list[Path]:
    """Generated files that are missing or differ from ``outputs``."""
    return [rel for rel, text in outputs.items()
            if not (root / rel).is_file() or (root / rel).read_text(encoding="utf-8") != text]


def write(outputs: dict[Path, str], root: Path) -> None:
    for rel, text in outputs.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")


def main(build: Callable[[], dict[Path, str]], script: str, repo: Path, doc: str, argv: list[str] | None) -> int:
    """Write the outputs of ``build`` or, with ``--check``, fail (exit 1) if any committed file is stale."""
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--check", action="store_true", help="fail if a generated file is stale")
    parser.add_argument("--out-root", type=Path, default=repo / OUT, help="output directory (tests use a tmp dir)")
    args = parser.parse_args(argv)
    outputs = build()
    if args.check:
        bad = stale(outputs, args.out_root)
        for rel in bad:
            print(f"{OUT / rel} is stale: run scripts/{script}", file=sys.stderr)
        return 1 if bad else 0
    write(outputs, args.out_root)
    print(f"wrote {len(outputs)} files under {OUT}")
    return 0
