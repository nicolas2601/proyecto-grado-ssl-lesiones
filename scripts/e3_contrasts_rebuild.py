r"""E3 final: MCC paired contrasts from a probe refit on the cached features (CPU, contrasts only).

E3 stored no probabilities. The final stage is deterministic given the cached ``train`` / ``val``
feature arrays, so this refits the probe (the stage's own path), checks that it reproduces every
stored point value of ``e3_final_eval.json`` within 1e-4 (macro-F1, balanced accuracy, melanoma
recall; plus every pre-existing contrast record) and then recomputes the contrasts with the stage's
own bootstrap (seed 20260923, 1000 resamples, grouped by lesion_id). A variant that does not
reproduce gets its MCC record written as "no disponible: el refit no reproduce los puntos
registrados" and NO number. The stored records are kept as they are.

Validation lock: this script never reads, writes or deletes ``val_read.lock`` and never calls the
final stage. It takes no val label from the dataset: ``y_val`` comes from the committed
``e3b_probas.npz`` (the artifact that already holds it; its ``val_ids`` must equal the split's), the
train labels from the training split. It writes ONLY to its own ``--out-dir`` (refused if that is
the folder of the original final eval or holds a lock). Weights, images and GPU are not used.

Pod, one run per final eval (CPU; est. 5-15 min each, unmeasured)::

    P=$HOME/work/proyectonp; C1=$P/salidas/c1; R=resultados/banco_completo
    python scripts/e3_contrasts_rebuild.py \
      --final-eval $R/e3/e3_final_eval.json --features-dir $P/salidas/e3/features \
      --e3b-probas $R/e3b/e3b_probas.npz --out-dir $P/salidas/e3_mcc_rebuild \
      --ham-root $HOME/work/datos/ham10000 --image-dir $P/eval_ham10000_images \
      --split-index $C1/c1_split_index.json --holdout $C1/selection/selection_holdout.json \
      --tone-manifest $P/salidas/final/fst/fst_pseudo_manifest.csv
    # same for $R/e3/robustez/e3_s1_bs64_autocast/e3_final_eval.json with $P/salidas/e3_s1_bs64_autocast/features
    # (and its own --out-dir); then copy <out-dir>/e3_final_eval.json + .environment.json over the repo files
    # ONLY if the pre-existing contrast records are unchanged (git diff shows only added mcc records).

Exit codes: 0 every variant reproduced; 3 written but some MCC unavailable; 2 refused (nothing written).
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

from dermssl import e3_mcc_rebuild as mr  # noqa: E402
from dermssl import e3_selection as sel  # noqa: E402
from dermssl import e3_store as st  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for flag in ("--final-eval", "--features-dir", "--e3b-probas", "--out-dir", "--ham-root", "--image-dir",
                 "--split-index", "--holdout", "--tone-manifest"):
        p.add_argument(flag, type=Path, required=True)
    p.add_argument("--n-resamples", type=int, default=None, help="default: the one stored in the final eval")
    args = p.parse_args(argv)
    args.limit = None  # v71_io.load_context reads it; a subsample would break every check
    return args


def _val_labels(path: Path, val_ids: list[str]) -> np.ndarray:
    """``y_val`` from the committed artifact, after proving it is the labels of THIS split's val ids."""
    with np.load(path) as data:
        if [str(i) for i in data["val_ids"]] != list(val_ids):
            raise mr.RebuildInputError(f"{path.name}: val_ids differ from the reconstructed split")
        return np.asarray(data["y_val"])


def _inputs(args) -> dict:
    import e3_io  # noqa: PLC0415
    import v71_io  # noqa: PLC0415

    ctx = v71_io.load_context(args)  # metadata and ids only; no val label is taken from it
    ids = ctx["ids"]
    return {"y_train": v71_io.labels_of(ctx, ids["train"]), "y_val": _val_labels(args.e3b_probas.expanduser(), ids["val"]),
            "groups": e3_io._lesion_ids(args.ham_root, ids)["val"],
            "codes": sel.tercile_codes(ids["val"], args.tone_manifest.expanduser()),
            "n_classes": len(ctx["classes"]), "mel": ctx["mel"]}


def main(argv=None) -> int:
    args = parse_args(argv)
    final_path, out_dir = args.final_eval.expanduser(), args.out_dir.expanduser()
    try:
        mr.check_out_dir(out_dir, final_path)
        final_doc = json.loads(final_path.read_text(encoding="utf-8"))
        feats = args.features_dir.expanduser()
        mr.verify_feature_hashes(final_doc, feats)
        records, report = mr.rebuild(final_doc, feats, n_resamples=args.n_resamples, **_inputs(args))
    except st.E3Error as exc:
        print(f"ABORT: {exc}")
        return 2
    out = mr.write_output(out_dir, final_doc, records, report)
    print(json.dumps(report, indent=1))
    print(f"-> {out}")
    return 0 if report["mcc_unavailable"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
