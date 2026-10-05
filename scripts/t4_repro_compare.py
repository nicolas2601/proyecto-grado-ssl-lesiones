"""Compare two identity-only T4 grids with each other and with the published v7.1 grid.

Paula's diagnosis (2026-09-29): the same identity grid is run twice with the
same flags on a GPU with no other process. If the two runs are identical
(feature hashes and probe outputs bit for bit), the gap to v7.1 comes from the
environment; if they differ, the pipeline is not deterministic and the size of
that difference is the measurement noise the errata would cite.

    python scripts/t4_repro_compare.py --run-a $P/salidas/t4_diag_1 --run-b $P/salidas/t4_diag_2 \
      --out $P/salidas/t4_diag_compare.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from dermssl import run_environment  # noqa: E402

V71_GRID = REPO / "resultados/clasificacion/v71/holdout_grid.json"
INPUT = "identity"


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _units(run: Path) -> dict:
    root = Path(run) / INPUT / "units"
    return {d.name: d for d in sorted(root.glob("*")) if (d / "result.json").exists()} if root.exists() else {}


def _cells(a_dir: Path, b_dir: Path) -> dict:
    a_res, b_res = _read(a_dir / "result.json"), _read(b_dir / "result.json")
    a_p, b_p = np.load(a_dir / "probas.npz"), np.load(b_dir / "probas.npz")
    out = {}
    for k, cid in enumerate(a_res["cells"]):
        pa, pb = a_p[f"p{k}"], b_p[f"p{b_res['cells'].index(cid)}"]
        out[cid] = {"max_abs_proba_delta": float(np.abs(pa.astype(float) - pb).max()),
                    "argmax_flips": int((pa.argmax(1) != pb.argmax(1)).sum()),
                    "bit_equal": bool(np.array_equal(pa, pb))}
    return out


def _unit(a_dir: Path, b_dir: Path) -> dict:
    da = _read(a_dir / "result.json").get("feature_sha256", {})
    db = _read(b_dir / "result.json").get("feature_sha256", {})
    differing = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
    cells = _cells(a_dir, b_dir)
    return {"features_equal": bool(da) and not differing, "differing_parts": differing, "cells": cells,
            "cells_bit_equal": all(c["bit_equal"] for c in cells.values()),
            "max_argmax_flips": max(c["argmax_flips"] for c in cells.values())}


def row_deltas(rows_a: dict, rows_b: dict) -> dict:
    """Max |a - b| per metric over the cells both grids have."""
    common = sorted(set(rows_a) & set(rows_b))
    metrics = sorted({m for c in common for m, v in rows_a[c].items() if isinstance(v, (int, float))})
    out = {}
    for m in metrics:
        deltas = [abs(float(rows_a[c][m]) - float(rows_b[c][m])) for c in common
                  if m in rows_b[c] and not (np.isnan(rows_a[c][m]) and np.isnan(rows_b[c][m]))]
        out[m] = max(deltas) if deltas else 0.0
    return {"n_cells": len(common), "max_abs_delta": out}


def compare_runs(run_a: Path, run_b: Path) -> dict:
    units_a, units_b = _units(run_a), _units(run_b)
    if not units_a or set(units_a) != set(units_b):
        raise SystemExit(f"ABORT: las unidades no coinciden: {sorted(units_a)} contra {sorted(units_b)}")
    units = {name: _unit(units_a[name], units_b[name]) for name in units_a}
    rows = row_deltas(_read(Path(run_a) / "t4_holdout_grid.json")["inputs"][INPUT],
                      _read(Path(run_b) / "t4_holdout_grid.json")["inputs"][INPUT])
    identical = (all(u["features_equal"] and u["cells_bit_equal"] for u in units.values())
                 and not any(rows["max_abs_delta"].values()))
    return {"identical": identical, "units": units, **rows}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-a", type=Path, required=True)
    p.add_argument("--run-b", type=Path, required=True)
    p.add_argument("--v71-grid", type=Path, default=V71_GRID)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    published = _read(args.v71_grid)["rows"]
    doc = {"a_vs_b": compare_runs(args.run_a, args.run_b)}
    for name, run in (("a", args.run_a), ("b", args.run_b)):
        doc[f"{name}_vs_published"] = row_deltas(_read(run / "t4_holdout_grid.json")["inputs"][INPUT], published)
        doc[f"{name}_environment"] = run_environment.read_sidecar(run / "t4_holdout_grid.json")
    args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    run_environment.write_sidecar(args.out, args)
    verdict = "IDENTICAS: el origen es el entorno" if doc["a_vs_b"]["identical"] else "DIFIEREN: no determinismo"
    print(f"a vs b: {verdict}; max |d macro_f1| a-b {doc['a_vs_b']['max_abs_delta'].get('macro_f1', 0):.2e}, "
          f"a-publicada {doc['a_vs_published']['max_abs_delta'].get('macro_f1', 0):.2e}, "
          f"b-publicada {doc['b_vs_published']['max_abs_delta'].get('macro_f1', 0):.2e} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
