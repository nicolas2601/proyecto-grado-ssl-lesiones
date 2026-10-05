r"""E3b: the E3-adopted variant F4 on the eight frozen public baselines, HAM10000 v7 split (docs/33 section 3).

Each baseline (alexnet, vgg19, resnet50, efficientnet_b0, vit_b16, dinov2_vits14, dinov2_vitb14,
simsiam_r50) is re-extracted on the v7 split, so it is PAIRED with B71 on the same validation
set (the JSONs of ``results/baselines`` use the Kaggle split and are not comparable), WITH and
WITHOUT F4 = concat(full image, lesion-box crop with 10 % margin) + standardised mask descriptors,
always with the ``student_r3`` masks of E3. Protocol: one view (resize 255 + centre crop 224, no
TTA), ImageNet normalisation, GAP of the last conv block (CNN) or CLS (ViT), probe logreg C=1
standardised ``class_weight=balanced`` fitted on the v7 train split and scored on val.

The stage reads validation labels ONLY because F4 was selected on the holdout and pinned: it
REFUSES to run unless ``--selection`` has the pinned sha256 and names F4 as the adopted winner,
and it chooses nothing on val. Outputs in ``--out-dir``: ``e3b_summary.json`` (rows, CIs,
paired contrasts: what the table builder reads), ``e3b_probas.npz``, ``cells/e3b__*.json``,
each with its ``.environment.json`` sidecar. Resumable: a cell is skipped when it exists with
the same key sha256; two retries; failures are listed and the exit code is 1.

B71 references come from E3's features (``--e3-dir``, train/val of the v7 run, checked against
this run's ids, masks and bytes, and against ``e3_final_eval.json``): ``B71`` / ``B71+F4`` with
the baselines' single view and probe, ``B71_tta`` / ``B71+F4_tta`` with its registered protocol.

Pod (RUNBOOK_CLUSTER; ``PYTHONPATH=$HOME/work/pylibs/timm_pin`` for timm 1.0.30, isolated)::

    P=$HOME/work/proyectonp; C1=$P/salidas/c1; E3=$P/salidas/e3; E3B=$P/salidas/e3b
    COMMON="--ham-root $HOME/work/datos/ham10000 --image-dir $P/eval_ham10000_images \
      --split-index $C1/c1_split_index.json --holdout $C1/selection/selection_holdout.json \
      --masks $C1/masks/c1_pseudo_masks.npy --masks-manifest $C1/masks/c1_pseudo_masks_manifest.json \
      --e3-dir $E3 --out-dir $E3B"
    PYTHONPATH=$HOME/work/pylibs/timm_pin python scripts/e3b_run.py $COMMON --limit 70 --allow-dirty \
      --models alexnet dinov2_vits14                       # smoke: no B71 rows, numbers NOT valid
    PYTHONPATH=$HOME/work/pylibs/timm_pin PROBE_BLAS_THREADS=4 python scripts/e3b_run.py $COMMON \
      --batch-size 32 --device cuda                        # the real run (all 8 models)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import e3_store as st  # noqa: E402
from dermssl import probe_threads, run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl.e3b import protocol as proto  # noqa: E402
from dermssl.e3b import runner  # noqa: E402
from dermssl.e3b.runtime import Settings  # noqa: E402

CI_RESAMPLES = proto.PROTOCOL["bootstrap"]["n_resamples"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, required=True)
    for flag in ("--ham-root", "--image-dir", "--split-index", "--holdout", "--masks", "--masks-manifest"):
        p.add_argument(flag, type=Path, default=None)
    for flag in ("--weights-dir", "--tone-manifest"):  # E3's COMMON flags are accepted; E3b does not use them
        p.add_argument(flag, type=Path, default=None, help="accepted for E3's COMMON; unused")
    p.add_argument("--e3-dir", type=Path, default=None, help="E3 output dir (features/ and cells/); needed unless --limit")
    p.add_argument("--selection", type=Path, default=REPO / proto.SELECTION_FILE)
    p.add_argument("--selection-sha256", default=proto.PINNED_SELECTION_SHA256)
    p.add_argument("--expected-winner", default=proto.ADOPTED_VARIANT)
    p.add_argument("--models", nargs="+", choices=proto.MODELS, default=list(proto.MODELS))
    p.add_argument("--ci-resamples", type=int, default=CI_RESAMPLES)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--decode-workers", type=int, default=4)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="smoke: stratified subset; numbers are NOT valid")
    p.add_argument("--allow-dirty", action="store_true", help="smoke (--limit) only")
    return p.parse_args(argv)


def _revision(args) -> str:
    try:
        return rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        if args.allow_dirty and args.limit:
            return "dirty"
        raise st.E3Error(str(exc)) from exc


def _settings(args, revision: str) -> Settings:
    if args.e3_dir is None and not args.limit:
        raise st.E3Error("falta --e3-dir: las filas B71 salen de las features de E3 (solo --limit lo omite)")
    return Settings(out_dir=args.out_dir, models=tuple(args.models), batch_size=args.batch_size,
                    workers=args.decode_workers, n_resamples=args.ci_resamples,
                    e3_dir=args.e3_dir.expanduser() if args.e3_dir else None, limit=args.limit,
                    selection_sha256=args.selection_sha256, revision=revision)


def main(argv=None, *, runtime=None, revision: str | None = None) -> int:
    args = parse_args(argv)
    args.out_dir = args.out_dir.expanduser()
    try:  # the gate comes before any weight, image or label is touched
        proto.require_pinned_selection(args.selection, args.selection_sha256, args.expected_winner)
        settings = _settings(args, revision or _revision(args))
        if runtime is None:
            import e3b_io  # noqa: PLC0415

            runtime = e3b_io.build_runtime(args)
        runtime.fingerprint = {**runtime.fingerprint, "selection_sha256": args.selection_sha256}

        def sidecar(path):
            run_environment.write_sidecar(path, args, probe_blas_threads=probe_threads.PROBE_BLAS_THREADS)

        return runner.run(runtime, settings, sidecar)
    except st.E3Error as exc:
        print(f"ABORT: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
