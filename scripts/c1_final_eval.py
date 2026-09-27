"""The v7 single evaluation: A*, B* and public DINOv2, one command.

``--split holdout`` fits the C1 logreg probe on 80 % of train and scores the
internal holdout; it exists to prove the pipeline reproduces the pilot
numbers. ``--split val`` fits on the whole fold-0 train and scores the 2004
validation images. Validation is touched once: that mode needs
``--single-evaluation`` and refuses to run if its output already exists.

    python scripts/c1_final_eval.py --split val --single-evaluation \
        --ham-root ~/work/datos/ham10000 --holdout $S/selection/selection_holdout.json \
        --plain dinov2_vitb14:$V/final_A/simsiam_dinov2_backbone_final.pt \
        --proposed dinov2_vitb14:$V/final_B/simsiam_dinov2_backbone_final.pt \
        --exclude-ids-file $S/selection/pretrain_exclude_ids_finals.txt \
        --out resultados/clasificacion/v7/final_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import c1_select_variant as sel  # noqa: E402
import cluster_frozen_bank as bank  # noqa: E402
from c1_build_val_index import rebuild_split  # noqa: E402

from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import final_eval as fe  # noqa: E402
from dermssl import paired_bootstrap as pb  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402

ARM_LABELS = {"plain": "A*", "proposed": "B*", "public": "DINOv2 público"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=("holdout", "val"), required=True)
    parser.add_argument("--single-evaluation", action="store_true",
                        help="required for --split val: the validation set is read once")
    parser.add_argument("--ham-root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--plain", required=True, help="ARCH:SOURCE of A*")
    parser.add_argument("--proposed", required=True, help="ARCH:SOURCE of B*")
    parser.add_argument("--public", default="dinov2_vitb14:public")
    parser.add_argument("--exclude-ids-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def evaluation_ids(split_name: str, split: dict, holdout: dict) -> dict:
    if split_name == "holdout":
        return {"fit_ids": holdout["fit_ids"], "holdout_ids": holdout["holdout_ids"]}
    return {"fit_ids": list(split["train_ids"]), "holdout_ids": list(split["val_ids"])}


def paired_rows(probas: dict, y_true: np.ndarray) -> list[dict]:
    return [
        pb.contrast_record(arm=arm, control=control, metric=metric, probe=sel.PROBE,
                           fraction=1.0, seed=0, y_true=y_true,
                           arm_proba=probas[arm], control_proba=probas[control])
        for arm, control in fe.CONTRASTS
        for metric in fe.METRICS
    ]


def main() -> int:
    args = parse_args()
    out = args.out.expanduser()
    if args.split == "val" and not args.single_evaluation:
        raise SystemExit("ABORT: --split val exige --single-evaluation (la validacion se toca una vez)")
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe; la evaluacion no se repite")
    revision = rp.clean_revision(REPO)
    import torch  # noqa: PLC0415

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    split, _ = rebuild_split(args.ham_root.expanduser())
    holdout = json.loads(args.holdout.expanduser().read_text(encoding="utf-8"))
    ids = evaluation_ids(args.split, split, holdout)
    image_index = bank.index_images([args.ham_root.expanduser()])
    specs = {"plain": args.plain, "proposed": args.proposed, "public": args.public}
    results, probas = {}, {}
    for role, spec in specs.items():
        results[role], probas[role], y_eval = sel.score(
            f"{role}={spec}", ids, image_index, split["id2label"], device, args
        )
    rows = paired_rows(probas, y_eval)
    document = {
        "split": args.split,
        "code_revision": revision,
        "probe": sel.PROBE,
        "n_eval": int(len(y_eval)),
        "eval_ids_sha256": si.ids_sha256(ids["holdout_ids"]),
        "exclude_ids_sha256": rp.file_sha256(args.exclude_ids_file.expanduser()),
        "arms": ARM_LABELS,
        "specs": specs,
        "results": results,
        "paired": rows,
        "predictions": fe.predictions(fe.paired_from_rows(rows)),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out.with_suffix(".probas.npz"), y_true=y_eval, **probas)
    for name, pred in document["predictions"].items():
        print(f"{name}: {pred['verdict']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
