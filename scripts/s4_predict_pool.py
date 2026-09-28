r"""Run a segmenter checkpoint over the pool cache, for the FST rerun (docs/27).

docs/27 registered that if S4-UNet A2 met its prediction it becomes the
candidate segmenter "and the FST is re-run with its masks". The published FST
manifest (``salidas/final/fst``) did NOT threshold a segmenter at 0.5: it read
``pseudo_r2``, i.e. ``student_r2/predictions.npy`` (float16 sigmoid over the
cache, ``student_engine.write_predictions``) turned into 0/1/255 pseudo-masks
by ``make_pseudo_masks.py --source student`` (uncertainty band, central region,
field, colour artefacts). So this script writes that same ``predictions.npy``
for any checkpoint ``eval_student.load_student`` accepts (S4-UNet or student
U-Net), and A2 then takes the student's exact downstream path: only the
segmenter changes. The probabilities come from ``eval_student.predict_masks``,
the evaluator's inference (fp32, ImageNet normalisation, no TTA), i.e. the
path behind the Dice that met the prediction; the student's own predictions
ran under bf16 autocast, a difference of rounding, not of model.

``--threshold-masks`` also writes the evaluator's binarisation (``> 0.5``,
uint8 0/1, no ignore band) with a ``pseudo_masks_stats.json``-style sidecar
that ``fst_pseudo_manifest.py --stats-json`` reads. It is a secondary view: it
is NOT how the published manifest was built, so comparing it with the
published manifest mixes the segmenter change with a mask-policy change.

Outputs are memmaps written batch by batch (58 457 x 224 x 224 float16 is
5.9 GB); ``run.json`` pins the checkpoint sha256, the pool (manifest sha256,
shape, splits), the clean code revision, n, and the sha256 of every output.

Pod commands (P=~/work/proyectonp; clean clone in $P/repo_rel)::

    P=$HOME/work/proyectonp; source $P/.venv/bin/activate
    export TORCH_HOME=$HOME/work/.torch; cd $P/repo_rel
    F=$P/salidas/final; S4=$P/salidas/s4_unet; OUT=$S4/a2_fst; mkdir -p $OUT
    PM=plan-segmentacion/camino-b-cnn/pseudo-mascaras/make_pseudo_masks.py
    EV=plan-segmentacion/comun/evaluacion
    # 1. smoke on GPU1: 512 images, prints images/s (partial: true, never used)
    CUDA_VISIBLE_DEVICES=1 python scripts/s4_predict_pool.py --checkpoint $S4/a2/best.pt \
      --cache-dir $F/cache --output-dir $OUT/smoke --limit 512 --batch-size 64
    # 2. the whole pool on GPU1, detached
    CUDA_VISIBLE_DEVICES=1 setsid nohup python scripts/s4_predict_pool.py \
      --checkpoint $S4/a2/best.pt --cache-dir $F/cache --output-dir $OUT/pool \
      --batch-size 64 < /dev/null > $OUT/pool.log 2>&1 &
    # 3. pseudo-masks with pseudo_r2's flags (cluster-final.json "pseudo"), CPU
    python $PM --saliency $OUT/pool/predictions.npy --source student \
      --output-dir $OUT/pseudo_a2 --field $F/cache/field.npy --images $F/cache/images.npy \
      --low 0.35 --high 0.65 --max-center-distance 0.5 --max-area-fraction 0.6 \
      --min-area-fraction 0.003
    # the flags must equal pseudo_r2's: this diff must print nothing
    K='{source,artifact_suppression,low,high,min_area_fraction,max_center_distance,max_area_fraction,region_mode}'
    diff <(jq -S "$K" $F/pseudo_r2/pseudo_masks_stats.json) \
         <(jq -S "$K" $OUT/pseudo_a2/pseudo_masks_stats.json)
    # label the masks as A2's in the manifest (make_pseudo_masks records "student")
    jq '. + {mask_source: "s4_unet_a2", mask_round: "a2"}' \
      $OUT/pseudo_a2/pseudo_masks_stats.json > $OUT/pseudo_a2/fst_stats.json
    # 4. FST manifest from A2's masks, original cache pixels and field, CPU
    setsid nohup python $EV/fst_pseudo_manifest.py --manifest $F/cache/manifest.csv \
      --pseudo-masks $OUT/pseudo_a2/pseudo_masks.npy --stats-json $OUT/pseudo_a2/fst_stats.json \
      --field $F/cache/field.npy --images $F/cache/images.npy \
      --output-json $OUT/fst/fst_pseudo_manifest.json \
      --output-csv $OUT/fst/fst_pseudo_manifest.csv < /dev/null > $OUT/fst.log 2>&1 &
    # 5. against the published (student pseudo_r2) manifest
    python $EV/compare_fst_manifests.py \
      --a $F/fst/fst_pseudo_manifest.json --label-a student_pseudo_r2 \
      --b $OUT/fst/fst_pseudo_manifest.json --label-b s4_unet_a2 \
      --output-json $OUT/comparison_vs_published.json

Step 5 attributes every difference to the masks only if the published
manifest is reproduced by ``--images`` (the ``sanity_published_vs_original``
check in ``compare_fst_manifests.py``). If that sanity diff is not all zero,
compare against ``salidas/v7_seg/fst_hair/original/fst_pseudo_manifest.json``
instead, which differs from step 4 only in the masks.

Cost: step 2 is one forward pass of a frozen ViT-B/14 + DPT decoder over
58 457 images at 224 px; step 1 measures images/s so the whole pool is
projected before launching it. Steps 3 and 4 are single-process CPU passes
(step 4 measured at ~10 ms/frame: 10-20 min); step 5 takes seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_SEG = REPO / "plan-segmentacion" / "comun"
for _extra in (REPO / "src", REPO / "scripts", _SEG / "evaluacion", _SEG / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dermssl import run_provenance as rp  # noqa: E402
from s4_provenance import pool_fingerprint, sha256_file  # noqa: E402

PREDICTIONS_FILE = "predictions.npy"  # the name student_engine.write_predictions uses
PREDICTIONS_DTYPE = np.float16  # and its dtype
THRESHOLD_MASKS_FILE = "threshold_masks.npy"
THRESHOLD_STATS_FILE = "threshold_masks_stats.json"
RUN_FILE = "run.json"
INFERENCE_PATH = "plan-segmentacion/comun/evaluacion/eval_student.predict_masks"
STUDENT_ARCH = "student_unet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detected by default")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N images only; a smoke artifact marked partial")
    parser.add_argument("--threshold-masks", action="store_true",
                        help="also write the evaluator's > 0.5 masks (secondary view)")
    parser.add_argument("--mask-source", default=None,
                        help="label recorded in the threshold masks' sidecar (e.g. s4_unet_a2)")
    args = parser.parse_args(argv)
    if args.threshold_masks and not args.mask_source:
        parser.error("--threshold-masks needs --mask-source: the manifest records who made them")
    return args


def _checkpoint_arch(path: Path) -> str:
    from ckpt_io import load_checkpoint

    return load_checkpoint(path).get("config", {}).get("arch", STUDENT_ARCH)


def _load_model(args, resolution: int):
    import torch

    import eval_student

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, trained_at = eval_student.load_student(args.checkpoint, device)
    if trained_at != resolution:
        raise SystemExit(f"ABORT: checkpoint resolution {trained_at} != cache resolution "
                         f"{resolution}; the pool would be resized behind the model's back")
    return model, device


def predict(args, images: np.ndarray, count: int, out: Path) -> float:
    """Fill ``predictions.npy`` (and the threshold masks) batch by batch; returns seconds."""
    from cache_io import create_cache
    from evaluate_segmentation import binarize

    import eval_student

    resolution = int(images.shape[1])
    model, device = _load_model(args, resolution)
    shape = (count, resolution, resolution)
    probabilities = create_cache(out / PREDICTIONS_FILE, shape, PREDICTIONS_DTYPE)
    masks = create_cache(out / THRESHOLD_MASKS_FILE, shape, np.uint8) if args.threshold_masks else None
    began = time.perf_counter()
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        batch = eval_student.predict_masks(model, np.array(images[start:stop]), device,
                                           stop - start, tta=False)
        probabilities[start:stop] = batch.astype(PREDICTIONS_DTYPE)
        if masks is not None:
            masks[start:stop] = binarize(batch).astype(np.uint8)
        if (stop // args.batch_size) % 100 == 0:
            print(f"  {stop}/{count}", flush=True)
    for array in (probabilities, masks):
        if array is not None:
            array.flush()
    return time.perf_counter() - began


def _threshold_stats(args, count: int) -> dict:
    from evaluate_segmentation import DEFAULT_THRESHOLD

    return {"mask_source": args.mask_source, "mask_round": None, "n_images": count,
            "threshold": float(DEFAULT_THRESHOLD), "rule": "probability > threshold",
            "ignore_band": False, "checkpoint": str(args.checkpoint),
            "reads_masks_or_labels": False}


def run_record(args, fingerprint: dict, count: int, seconds: float, revision: str,
               outputs: dict) -> dict:
    return {
        "experiment": "segmenter over the pool cache for the FST rerun (docs/27)",
        "checkpoint": {"path": str(args.checkpoint), "sha256": sha256_file(args.checkpoint),
                       "arch": _checkpoint_arch(args.checkpoint)},
        "pool": fingerprint, "n": count, "partial": count != fingerprint["n_images"],
        "inference": {"path": INFERENCE_PATH, "tta": False, "precision": "fp32",
                      "resolution": int(fingerprint["image_shape"][0])},
        "code_revision": revision, "outputs": outputs,
        "compute_seconds": seconds, "images_per_second": count / seconds if seconds else None,
        "next_step": "make_pseudo_masks.py --source student with pseudo_r2's flags",
        "reads_masks_or_labels": False, "complete": True,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.output_dir.expanduser()
    if out.exists():
        raise SystemExit(f"ABORT: {out} exists; pool predictions are never overwritten")
    try:
        revision = rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    fingerprint = pool_fingerprint(args.cache_dir)
    images = np.load(args.cache_dir / "images.npy", mmap_mode="r")
    count = min(args.limit or len(images), len(images))
    out.mkdir(parents=True)
    try:
        seconds = predict(args, images, count, out)
    except BaseException:
        # a half-written .npy has a valid header and the wrong contents
        for name in (PREDICTIONS_FILE, THRESHOLD_MASKS_FILE):
            (out / name).unlink(missing_ok=True)
        raise
    if args.threshold_masks:
        (out / THRESHOLD_STATS_FILE).write_text(
            json.dumps(_threshold_stats(args, count), indent=2) + "\n", encoding="utf-8")
    names = [PREDICTIONS_FILE, *([THRESHOLD_MASKS_FILE] if args.threshold_masks else [])]
    outputs = {name: sha256_file(out / name) for name in names}
    record = run_record(args, fingerprint, count, seconds, revision, outputs)
    (out / RUN_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": count, "seconds": seconds,
                      "images_per_second": record["images_per_second"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
