"""Build the docs/28 selection holdout and the pretraining exclusion list.

Reads the committed C1 split index, rebuilds the HAM metadata with the bank's
own loader (for dx and lesion_id), verifies the index against it, and writes:

* ``selection_holdout.json``: ``fit_ids`` (80 % of train) and ``holdout_ids``
  (20 %), grouped by lesion, stratified by dx, seed 42, with their digests;
* ``pretrain_exclude_ids.txt``: validation + holdout, one id per line, for
  ``pretrain.py --exclude-ids-file``. Neither may shape the SSL features that
  are later scored on them.

Never overwrites an existing output: a holdout that silently changes between
the pilots and the final runs would select on one set and report another.

    python scripts/c1_selection_holdout.py --ham-root ~/work/datos/ham10000 \
        --split-index $S/c1_split_index.json --out-dir $S/selection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from c1_build_val_index import rebuild_split  # noqa: E402

from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import selection_holdout as sh  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ham-root", type=Path, required=True)
    parser.add_argument("--split-index", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser()
    holdout_path = out_dir / "selection_holdout.json"
    exclude_path = out_dir / "pretrain_exclude_ids.txt"
    existing = [p for p in (holdout_path, exclude_path) if p.exists()]
    if existing:
        raise SystemExit(f"ABORT: ya existen {existing}; no se sobrescribe un holdout")

    index = json.loads(args.split_index.expanduser().read_text(encoding="utf-8"))
    train_ids, val_ids = list(index["train_ids"]), list(index["val_ids"])
    split, meta = rebuild_split(args.ham_root.expanduser())
    if list(split["train_ids"]) != train_ids or list(split["val_ids"]) != val_ids:
        raise SystemExit("ABORT: el indice de C1 no coincide con el split reconstruido")

    row = meta.set_index("image_id")
    labels = [split["cls2idx"][d] for d in row.loc[train_ids, "dx"]]
    groups = list(row.loc[train_ids, "lesion_id"])
    fit_ids, holdout_ids = sh.grouped_holdout(train_ids, labels, groups, forbidden=val_ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text(
        json.dumps(
            {
                "seed": sh.HOLDOUT_SEED,
                "n_splits": sh.HOLDOUT_N_SPLITS,
                "source_train_ids_sha256": si.ids_sha256(train_ids),
                "fit_ids_sha256": si.ids_sha256(fit_ids),
                "holdout_ids_sha256": si.ids_sha256(holdout_ids),
                "fit_ids": fit_ids,
                "holdout_ids": holdout_ids,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    exclude_path.write_text("\n".join(sorted(set(val_ids) | set(holdout_ids))) + "\n", encoding="utf-8")
    print(
        f"fit {len(fit_ids)} / holdout {len(holdout_ids)} de {len(train_ids)} train; "
        f"excluidos del pool: {len(val_ids)} val + {len(holdout_ids)} holdout -> {exclude_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
