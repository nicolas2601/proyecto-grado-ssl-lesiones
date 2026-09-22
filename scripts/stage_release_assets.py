"""Lay out the release assets so each file says what it is.

The four arms all write a checkpoint under the same file name. Copied into
one directory as-is they overwrite each other and the release ends up with
whichever one landed last - silently, and with the right file count. Here
every asset is renamed after the arm that produced it, and the copy refuses
to clobber a name that is already taken.

    python scripts/stage_release_assets.py --src <descarga> --staging <dir>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_release_notes import MissingEvidence, discover_arms

BANK_FILES = ("bank.json", "bank_rows.csv", "strata.json", "bank.md")


def _copy(src: Path, dest: Path) -> None:
    if dest.exists():
        raise MissingEvidence(f"{dest.name} ya existe en el staging; dos arms colisionan")
    shutil.copy2(src, dest)


def stage_arms(arms: dict[str, dict[str, Path]], staging: Path) -> list[Path]:
    written: list[Path] = []
    for arm in sorted(arms):
        weights = staging / f"{arm}_backbone.pt"
        manifest = staging / f"{arm}_run.json"
        _copy(arms[arm]["weights"], weights)
        _copy(arms[arm]["run"], manifest)
        written += [weights, manifest]
    return written


def stage_bank(src: Path, staging: Path) -> list[Path]:
    """Copy the bank outputs, from the root or from a bank/ subdirectory."""
    written: list[Path] = []
    for name in BANK_FILES:
        for candidate in (src / name, src / "bank" / name):
            if candidate.is_file():
                dest = staging / name
                if not dest.exists():
                    shutil.copy2(candidate, dest)
                    written.append(dest)
                break
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", type=Path, required=True, help="Downloaded or trained outputs")
    parser.add_argument("--staging", type=Path, required=True, help="Directory to fill")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.staging.mkdir(parents=True, exist_ok=True)
    try:
        arms = discover_arms(args.src)
        written = stage_arms(arms, args.staging)
    except MissingEvidence as exc:
        raise SystemExit(f"no publicable: {exc}")

    written += stage_bank(args.src, args.staging)
    print(f"staging listo: {len(arms)} checkpoints, {len(written)} archivos")


if __name__ == "__main__":
    main()
