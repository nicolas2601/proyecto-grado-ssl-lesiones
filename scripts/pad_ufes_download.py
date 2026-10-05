"""Download the official PAD-UFES-20 files and verify them against Mendeley (docs/30 2.1).

~3.6 GB (3,592,712,972 bytes) in four files. Meant for the pod, not a laptop:

    python scripts/pad_ufes_download.py --dest ~/work/datos/pad_ufes_20 --extract

A file already present is verified, not downloaded again. A file whose size
or sha256 differs is deleted and the script exits non-zero. ``--extract``
unpacks the three image archives into ``<dest>/images`` after all four verify.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import pad_ufes_files as pf  # noqa: E402

IMAGES_DIR = "images"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--extract", action="store_true",
                        help=f"unpack the image archives into <dest>/{IMAGES_DIR}")
    return parser.parse_args(argv)


def download_all(dest: Path, opener=None) -> list[Path]:
    kwargs = {} if opener is None else {"opener": opener}
    paths = []
    for official in pf.OFFICIAL_FILES:
        print(f"{official.name}: {official.size / 1e9:.2f} GB")
        paths.append(pf.fetch(official, dest, **kwargs))
    return paths


def extract_all(dest: Path, paths: list[Path]) -> int:
    archives = [p for p in paths if p.suffix == ".zip"]
    return sum(pf.safe_extract(p, dest / IMAGES_DIR) for p in archives)


def main(argv: list[str] | None = None, opener=None) -> int:
    args = parse_args(argv)
    dest = args.dest.expanduser()
    try:
        paths = download_all(dest, opener)
        if args.extract:
            print(f"{extract_all(dest, paths)} miembros extraidos en {dest / IMAGES_DIR}")
    except pf.IntegrityError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    print(f"PAD-UFES-20 ({pf.DATASET_DOI}, {pf.LICENSE}) verificado en {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
