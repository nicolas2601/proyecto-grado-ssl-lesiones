"""argparse types shared by the C1 scripts.

Small on purpose. It exists because the same validator was about to be written
twice: `--limit 0` reading as "no limit" was a real defect in one script, and
copying the fix into the second one is how the two drift apart later. A CLI
guard duplicated is a CLI guard that will disagree with itself.
"""

from __future__ import annotations

import argparse

from dermssl.c1_mask_export import is_sha256


def positive_int(raw: str) -> int:
    """A count that must be at least 1.

    ``--limit 0`` used to be falsy and therefore mean "no limit" -- the exact
    opposite of what the operator typed.
    """
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def sha256_arg(raw: str) -> str:
    """The same digest validator the manifests use, so the CLI cannot accept
    a value the artifact layer would later reject."""
    if not is_sha256(raw):
        raise argparse.ArgumentTypeError(f"not a lowercase sha256 hex digest: {raw!r}")
    return raw
