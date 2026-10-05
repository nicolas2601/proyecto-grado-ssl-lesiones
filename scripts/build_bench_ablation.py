"""Build resultados/banco_completo/TABLA_ABLACION.md (+ latex/) from committed JSON.

    python scripts/build_bench_ablation.py            # write
    python scripts/build_bench_ablation.py --check    # CI: exit 1 if a committed file is stale
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl.bench import ablation, cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return cli.main(lambda: ablation.outputs(REPO), "build_bench_ablation.py", REPO, __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
