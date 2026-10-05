"""Build resultados/banco_completo/ (tables, LaTeX, confusion matrices) from committed JSON.

The figures need matplotlib to be drawn; ``--check`` only needs them to exist.

The proposed model and the 8 baselines, frozen and fine-tuned, on HAM10000, ISIC 2019 and
ISIC 2020 (docs/33, sections 3-6). Nothing is transcribed: every number is read from the
JSON under ``results/`` and ``resultados/clasificacion/v71``. ``--check`` fails if any
generated text differs from what the JSON produce.

    python scripts/build_bench_tables.py            # write
    python scripts/build_bench_tables.py --check    # CI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.bench import confusion  # noqa: E402
from dermssl.bench.confusion_render import readme, write_png  # noqa: E402
from dermssl.bench.latex import render_latex  # noqa: E402
from dermssl.bench.render import render_markdown  # noqa: E402
from dermssl.bench.rows import DATASETS, Dataset  # noqa: E402

OUT = Path("resultados") / "banco_completo"


def stem(ds: Dataset) -> str:
    return "TABLA_" + ds.title.replace(" ", "")


def text_outputs(repo: Path = REPO) -> dict[Path, str]:
    """Relative path (under OUT) -> exact text of every generated table."""
    out: dict[Path, str] = {}
    for ds in DATASETS.values():
        out[Path(f"{stem(ds)}.md")] = render_markdown(repo, ds)
        out[Path("latex") / f"{stem(ds)}.tex"] = render_latex(repo, ds)
    arts = confusion.artifacts(repo)
    for art in arts:
        out[confusion.rel_json(art)] = confusion.dump(art)
    out[confusion.OUT / "manifest.json"] = confusion.dump(confusion.manifest(arts))
    out[confusion.OUT / "README.md"] = readme(arts, confusion.pending())
    return out


def png_paths(repo: Path = REPO) -> list[Path]:
    """Relative path (under OUT) of every confusion figure; checked for existence only (bytes vary by version)."""
    return [confusion.rel_json(a).with_suffix(".png") for a in confusion.artifacts(repo)]


def stale(outputs: dict[Path, str], root: Path) -> list[Path]:
    """Generated files that are missing or differ from ``outputs``."""
    return [rel for rel, text in outputs.items()
            if not (root / rel).is_file() or (root / rel).read_text(encoding="utf-8") != text]


def write(outputs: dict[Path, str], root: Path, png: bool = True) -> None:
    for rel, text in outputs.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    if not png:
        return
    for art in confusion.artifacts(REPO):
        write_png(art, root / confusion.rel_json(art).with_suffix(".png"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if a generated file is stale")
    parser.add_argument("--skip-png", action="store_true", help="do not draw or require the PNG figures")
    parser.add_argument("--out-root", type=Path, default=REPO / OUT, help="output directory (tests use a tmp dir)")
    args = parser.parse_args(argv)
    outputs = text_outputs()
    if args.check:
        bad = stale(outputs, args.out_root)
        bad += [] if args.skip_png else [p for p in png_paths() if not (args.out_root / p).is_file()]
        for rel in bad:
            print(f"{OUT / rel} is stale: run scripts/build_bench_tables.py", file=sys.stderr)
        return 1 if bad else 0
    write(outputs, args.out_root, png=not args.skip_png)
    print(f"wrote {len(outputs)} files under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
