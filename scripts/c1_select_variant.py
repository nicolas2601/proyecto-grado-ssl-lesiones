"""Score pilot variants on the docs/28 internal holdout and pick the winner.

Each variant's backbone is frozen, the bank eval geometry extracts features
for the fit (80 % of train) and holdout (20 %) ids, the C1 ``logreg`` probe is
fit on the first and scored on the second. Validation never enters. The
reference (public DINOv2) is scored the same way but is never a candidate.

    python scripts/c1_select_variant.py --ham-root ~/work/datos/ham10000 \
        --holdout $S/selection/selection_holdout.json \
        --variant A1=dinov2_vitb14:$T/pilot_A1/simsiam_dinov2_backbone_final.pt \
        --variant A2=dinov2_vitb14:$T/pilot_A2/simsiam_dinov2_backbone_final.pt \
        --reference public=dinov2_vitb14:public --out $S/selection/wave1.json \
        --exclude-ids-file $S/selection/pretrain_exclude_ids.txt

Refuses to run from a dirty tree (untracked files included), so the recorded
``code_revision`` is the code that ran.
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

import cluster_frozen_bank as bank  # noqa: E402
from c1_build_val_index import rebuild_split  # noqa: E402

from dermssl import c1_split_identity as si  # noqa: E402
from dermssl import metrics as _metrics  # noqa: E402
from dermssl import paired_bootstrap as pb  # noqa: E402
from dermssl import probe_threads, probes  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import selection_holdout as sh  # noqa: E402

RESOLUTION = 224
PROBE = "logreg"
REPORTED = ("macro_f1", "balanced_accuracy", "auc", "ece")
CI_METRICS = REPORTED  # Reported beside the rule, never used by it.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ham-root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True, help="NAME=ARCH:SOURCE")
    parser.add_argument("--reference", action="append", default=[], help="NAME=ARCH:SOURCE, never a candidate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--exclude-ids-file", type=Path, required=True,
        help="the pretraining exclusion list the variants ran with; its sha256 is recorded",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    # 0: the pod caps /dev/shm, and worker processes die on it (RUNBOOK_CLUSTER 12.3).
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def score(spec: str, holdout: dict, image_index, id2label, device, args) -> dict:
    name, arch, source = bank.parse_backbone_spec(spec)
    model = bank.build_frozen_backbone(arch, source, RESOLUTION, device)["model"]
    extract = lambda ids: bank.extract_features(  # noqa: E731
        model, ids, image_index, id2label, RESOLUTION, device, args.batch_size, args.num_workers
    )
    x_fit, y_fit = extract(holdout["fit_ids"])
    x_hold, y_hold = extract(holdout["holdout_ids"])
    n_classes = int(max(y_fit.max(), y_hold.max())) + 1
    proba = probe_threads.call_pinned(probes.run_probe, PROBE, x_fit, y_fit, x_hold, n_classes)
    summary = _metrics.summarize(y_hold, proba, n_classes)
    print(f"{name}: " + " | ".join(f"{k} {summary[k]:.4f}" for k in REPORTED))
    return {
        "spec": f"{arch}:{source}",
        **{k: float(summary[k]) for k in REPORTED},
        "ci95": {k: sh.bootstrap_ci(y_hold, proba, pb.METRIC_FUNCTIONS[k]) for k in CI_METRICS},
    }, proba, y_hold


def main() -> int:
    args = parse_args()
    out = args.out.expanduser()
    if out.exists():
        raise SystemExit(f"ABORT: {out} ya existe; una seleccion no se reescribe")
    try:
        revision = rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    exclude_sha = rp.file_sha256(args.exclude_ids_file.expanduser())
    holdout = json.loads(args.holdout.expanduser().read_text(encoding="utf-8"))
    for key in ("fit_ids", "holdout_ids"):
        if si.ids_sha256(holdout[key]) != holdout[f"{key}_sha256"]:
            raise SystemExit(f"ABORT: {key} no coincide con su digest")

    split, _ = rebuild_split(args.ham_root.expanduser())
    leaked = set(split["val_ids"]) & (set(holdout["fit_ids"]) | set(holdout["holdout_ids"]))
    if leaked:
        raise SystemExit(f"ABORT: {len(leaked)} ids de validacion en el holdout")
    image_index = bank.index_images([args.ham_root.expanduser()])

    import torch  # noqa: PLC0415

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    names = [bank.parse_backbone_spec(s)[0] for s in args.variant]
    results, probas = {}, {}
    for spec in args.variant + args.reference:
        name = bank.parse_backbone_spec(spec)[0]
        results[name], probas[name], y_hold = score(
            spec, holdout, image_index, split["id2label"], device, args
        )
    winner = sh.select_winner(results, candidates=names)
    paired = {
        other: {
            metric: pb.contrast_record(
                arm=winner, control=other, metric=metric, probe=PROBE, fraction=1.0, seed=0,
                y_true=y_hold, arm_proba=probas[winner], control_proba=probas[other],
            )
            for metric in CI_METRICS
        }
        for other in results
        if other != winner
    }
    document = {
        "rule": f"max macro_f1; within {sh.TIE_MACRO_F1} of the best, lowest ece (docs/28 4.1)",
        "probe": PROBE,
        "code_revision": revision,
        "exclude_ids_file": str(args.exclude_ids_file),
        "exclude_ids_sha256": exclude_sha,
        "holdout_ids_sha256": holdout["holdout_ids_sha256"],
        "candidates": names,
        "results": results,
        "winner": winner,
        "paired_vs_winner": paired,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1), encoding="utf-8")
    run_environment.write_sidecar(out, args, loky_inner_threads=probe_threads.PROBE_BLAS_THREADS)
    # Probabilities kept so later contrasts need no second feature extraction.
    np.savez_compressed(out.with_suffix(".probas.npz"), y_true=y_hold, **probas)
    print(f"ganadora: {winner} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
