"""Build resultados/banco_completo/CONTRASTES_SSL.md (+ latex/) from committed JSON.

    python scripts/build_bench_contrasts.py            # write
    python scripts/build_bench_contrasts.py --check    # CI: exit 1 if a committed file is stale
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.bench import cli, contrasts_doc  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return cli.main(lambda: contrasts_doc.outputs(REPO), "build_bench_contrasts.py", REPO, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
