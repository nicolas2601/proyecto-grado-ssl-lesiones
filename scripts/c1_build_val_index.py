"""Rebuild the HAM fold-0 split and verify it against the index C1 already uses.

Why this script exists
----------------------
``val_index.json`` has no producer in this repo. The bank gate reads it
(``--val-index``) and nothing writes it, so the file the gate just passed
against came from somewhere unversioned. That is a provenance hole: the whole
C1 comparison rests on "the same images as the v6 bank", and right now nothing
in the repo can show that.

So this rebuilds the split from its real source -- ``build_split`` in
``cluster_frozen_bank.py``, ``StratifiedGroupKFold(5, shuffle=True,
random_state=42)``, fold 0 -- and compares it to the stored file id by id, in
order, on BOTH sides.

It FAILS on any difference rather than overwriting. If the rebuilt split does
not match, the meaning is not "the file is stale": it is that the v6 bank and
C1 are not looking at the same images, and every contrast between them would
be comparing two different experiments.

Usage
-----
    python scripts/c1_build_val_index.py \\
        --ham-root ~/work/proyectonp/datos/ham10000 \\
        --val-index ~/work/proyectonp/salidas/tone/bank_repro/predictions/val_index.json \\
        --identity-out ~/work/proyectonp/salidas/c1/split_identity.json

Add ``--write`` only to CREATE the index when it does not exist yet. It never
overwrites an existing one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import c1_split_identity as si  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ham-root", type=Path, required=True)
    parser.add_argument("--val-index", type=Path, required=True)
    parser.add_argument("--identity-out", type=Path, default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help="crear el indice si NO existe; nunca sobrescribe uno existente",
    )
    return parser.parse_args()


def rebuild_split(ham_root: Path) -> dict:
    """The bank's own split function, not a reimplementation of it.

    Rebuilding StratifiedGroupKFold by hand here is exactly the mistake that
    produced the centre-crop bug: a second implementation that looks right and
    disagrees in one detail.
    """
    import cluster_frozen_bank as bank  # noqa: PLC0415

    meta = bank.read_ham_metadata_csv(ham_root)
    image_index = bank.index_images([ham_root])
    meta = bank.filter_metadata_to_available(meta, image_index)
    return bank.build_split(meta), meta


def main() -> int:
    args = parse_args()
    split, meta = rebuild_split(args.ham_root.expanduser())
    train_ids = list(split["train_ids"])
    val_ids = list(split["val_ids"])

    print(f"split reconstruido: {len(train_ids)} train / {len(val_ids)} val")

    lesion_by_id = dict(zip(meta["image_id"], meta["lesion_id"], strict=True))
    shared = si.shared_lesions(
        train_ids=train_ids, val_ids=val_ids, lesion_by_id=lesion_by_id
    )
    if shared:
        print(f"ABORT: {len(shared)} lesiones en ambos lados, p.ej. {shared[:5]}")
        return 1
    print("lesiones compartidas entre train y val: 0")

    identity = si.split_identity(train_ids=train_ids, val_ids=val_ids)
    for key in ("c1_train_ids_sha256", "c1_val_ids_sha256", "feature_cache_ids_fingerprint"):
        print(f"{key}: {identity[key]}")

    index_path = args.val_index.expanduser()
    if not index_path.exists():
        if not args.write:
            print(f"ABORT: {index_path} no existe. Usa --write para crearlo.")
            return 1
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            json.dumps({"train_ids": train_ids, "val_ids": val_ids}, indent=1),
            encoding="utf-8",
        )
        print(f"indice CREADO en {index_path}")
    else:
        stored = json.loads(index_path.read_text(encoding="utf-8"))
        agrees, reasons = si.compare_to_index(stored, train_ids=train_ids, val_ids=val_ids)
        if not agrees:
            print("\n=== EL INDICE NO COINCIDE CON EL SPLIT RECONSTRUIDO ===")
            for reason in reasons:
                print(f"  - {reason}")
            print(
                "\nEsto NO es un archivo desactualizado: significa que el banco v6 y C1 no "
                "estan mirando las mismas imagenes. No se sobrescribe nada."
            )
            return 1
        print(f"indice verificado contra {index_path}: identico en train y val")

    if args.identity_out is not None:
        out = args.identity_out.expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(identity, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"identidad del split escrita en {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
