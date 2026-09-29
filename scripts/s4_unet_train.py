r"""S4-UNet A1/A2 (docs/27): train a DPT decoder on a frozen DINOv2, on pseudo-masks.

Registered on 2026-09-28, before running. A2 trains on the r0 CCAM
pseudo-masks student training started from; A1 on the DINO pseudo-masks of
``s4_dino_saliency.py``; the A2 variant swaps in the A* seed-42 encoder.
Pool, exclusions, loss, augmentation, pseudo-validation selection, schedule
and seed are the student's (``s4_unet_loop``); the defaults are one
student_r3 round from ``cluster-final.json`` (20 epochs, 32 per GPU,
Adam 1e-4, 1 warm-up epoch, cosine, best-val, bf16, seed 42). Reference
masks appear only in ``evaluate_segmentation.py``, the student's evaluator.

Pod commands (P=~/work/proyectonp; clean clone in $P/repo_rel)::

    P=$HOME/work/proyectonp; source $P/.venv/bin/activate
    export TORCH_HOME=$HOME/work/.torch; cd $P/repo_rel
    F=$P/salidas/final; S4=$P/salidas/s4_unet; V=$P/salidas/v7
    T="python -m torch.distributed.run --nproc_per_node=2 scripts/s4_unet_train.py"
    COMMON="--cache-dir $F/cache --wandb-project dermssl-segmentacion"
    # 1. timing smoke A2 (50 steps; prints s/step and projected GPU-h)
    $T $COMMON --variant A2 --pseudo-masks $F/pseudo_r0/pseudo_masks.npy \
      --output-dir $S4/smoke_a2 --timing-smoke 50
    # 2. full A2 (launch only if projected_gpu_hours <= 4, docs/27)
    setsid nohup $T $COMMON --variant A2 --pseudo-masks $F/pseudo_r0/pseudo_masks.npy \
      --output-dir $S4/a2 < /dev/null > $S4/a2.log 2>&1 &
    # 2b. A2 with the A* seed-42 encoder, same cost
    setsid nohup $T $COMMON --variant A2-astar \
      --backbone dinov2_vitb14:$V/final_A/simsiam_dinov2_backbone_final.pt \
      --pseudo-masks $F/pseudo_r0/pseudo_masks.npy --output-dir $S4/a2_astar \
      < /dev/null > $S4/a2_astar.log 2>&1 &
    # 3. A1 saliency + pseudo-masks: see the docstring of scripts/s4_dino_saliency.py
    # 4. full A1 (after pseudo_a1 exists; smoke it first like step 1)
    setsid nohup $T $COMMON --variant A1 --pseudo-masks $S4/pseudo_a1/pseudo_masks.npy \
      --output-dir $S4/a1 < /dev/null > $S4/a1.log 2>&1 &
    # 5. evaluation: the student_r3 evaluator and flags, on ISIC 2018 and HAM10000
    for RUN in a2 a2_astar a1; do for SET in isic2018 ham10000; do
      case $SET in
        isic2018) IMG=$P/datos_isic2018/ISIC2018_Task1-2_Training_Input
                  GT=$P/masks_eval/isic2018/ground_truth ;;
        ham10000) IMG=$P/eval_ham10000_images
                  GT=$P/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl ;;
      esac
      E=$S4/$RUN/evaluacion_$SET
      python plan-segmentacion/comun/evaluacion/evaluate_segmentation.py \
        --checkpoint $S4/$RUN/best.pt --image-dir $IMG --mask-dir $GT \
        --output-dir $E --eval-name ${SET}_s4_$RUN --overlay-dir $E/overlays \
        --hausdorff --batch-size 32 --fst-manifest $F/fst/fst_pseudo_manifest.csv \
        --strata-column tone_tercile --wandb-project dermssl-segmentacion
    done; done

Prediction (docs/27): Dice(S4-UNet) >= Dice(student_r3) - 0.01 = 0.7077 on
ISIC 2018. A relaunch of the same command resumes from ``last.pt``.

T7 (docs/27, registered 2026-09-28): ``--variant A2-T7 --tercile-manifest``
adds k = 2 oversampling of the dark luminance tercile at A2's budget and the
darkening augmentation (medium and light only, p = 0.5). ``--no-oversample``
is arm (b), augmentation only; arm (a) is ``--variant A2 --seed 43``. T7
refuses unless its registration holds and the manifest, the r0 pseudo-masks
and the cache manifest carry their registered sha256. Pod commands (run the
r0 check first, then keep ``python -m dermssl.hub_keepalive`` alongside)::

    T7=$P/salidas/t7; M=$F/strata_v2/fst_pseudo_manifest.csv
    python -m dermssl.hub_keepalive > $T7/keepalive.log 2>&1 &
    $T $COMMON --variant A2-T7 --tercile-manifest $M \
      --pseudo-masks $F/pseudo_r0/pseudo_masks.npy --output-dir $T7/t7 --timing-smoke 50
    setsid nohup $T $COMMON --variant A2-T7 --tercile-manifest $M \
      --pseudo-masks $F/pseudo_r0/pseudo_masks.npy --output-dir $T7/t7 < /dev/null > $T7/t7.log 2>&1
    setsid nohup $T $COMMON --variant A2 --seed 43 \
      --pseudo-masks $F/pseudo_r0/pseudo_masks.npy --output-dir $T7/a2_s43 < /dev/null > $T7/a2_s43.log 2>&1
    setsid nohup $T $COMMON --variant A2-T7 --no-oversample --tercile-manifest $M \
      --pseudo-masks $F/pseudo_r0/pseudo_masks.npy --output-dir $T7/t7_aug < /dev/null > $T7/t7_aug.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _extra in (REPO / "src", REPO / "scripts", REPO / "plan-segmentacion" / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import dist_utils as du  # noqa: E402
import gpu_augment as ga  # noqa: E402
import s4_unet_loop as loop  # noqa: E402
from cache_io import read_manifest  # noqa: E402
from dermssl import luminance_sampling as ls  # noqa: E402
from dermssl import pad_ufes_guard as pg  # noqa: E402
from dermssl import run_environment  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import torch_s4_unet as s4  # noqa: E402
from s4_provenance import (  # noqa: E402
    TRANSDUCTIVE_NOTE, parse_backbone, pool_fingerprint, sha256_file, weights_digest,
)
from wandb_logging import add_tracking_arguments  # noqa: E402

VARIANTS = ("A1", "A2", "A2-astar", "A2-T7")
T7_VARIANT = "A2-T7"
T7_SEED = 42
RUN_FILE = "run.json"
SMOKE_FILE = "timing_smoke.json"
RESUME_KEYS = ("variant", "code_revision", "pool", "pseudo_masks", "backbone", "schedule", "seed", "t7")
STUDENT_R3_ISIC_DICE = 0.7177
PREDICTION_MARGIN = 0.01

__all__ = ["main", "parse_args", "sha256_file"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-dir", required=True, help="student_r3 cache (images.npy, manifest.csv)")
    parser.add_argument("--pseudo-masks", required=True, help="pseudo_masks.npy (0, 1, 255)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--backbone", default="dinov2_vitb14:public", type=parse_backbone,
                        metavar="ARCH:SOURCE", help="SOURCE is 'public' or a state_dict file")
    parser.add_argument("--features", type=int, default=s4.DEFAULT_FEATURES)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32, help="per GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", default="adam", choices=("adam", "adamw"))
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--selection", default="best-val", choices=("best-val", "last"))
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--timing-smoke", type=int, default=0, metavar="STEPS",
                        help="time STEPS training steps, project GPU-hours, save nothing")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--tercile-manifest", default="", help="T7: fst_pseudo_manifest.csv (sha 57df1e65...)")
    parser.add_argument("--dark-oversample", type=int, default=ls.REGISTERED_K, choices=(ls.REGISTERED_K,),
                        metavar="K", help="T7: copies of each dark-tercile row per epoch (only the registered 2)")
    parser.add_argument("--no-oversample", action="store_true", help="T7 arm (b): K = 1, augmentation only")
    add_tracking_arguments(parser)
    return parser.parse_args(argv)


def _schedule(args) -> dict:
    keys = ("epochs", "batch_size", "lr", "weight_decay", "optimizer", "warmup_epochs",
            "dice_weight", "selection", "precision", "features", "max_steps_per_epoch")
    return {key: getattr(args, key) for key in keys}


def base_record(args, revision: str) -> dict:
    arch, source = args.backbone
    return {
        "experiment": "S4-UNet (docs/27, registered 2026-09-28)", "variant": args.variant,
        "code_revision": revision, "seed": args.seed,
        "pool": pool_fingerprint(Path(args.cache_dir)),
        "pseudo_masks": {"path": str(args.pseudo_masks), "sha256": sha256_file(Path(args.pseudo_masks))},
        "backbone": {"arch": arch, "source": source, "weights_sha256": weights_digest(source),
                     "frozen": True, "blocks": list(s4.REGISTERED_BLOCKS)},
        "schedule": _schedule(args),
        "evaluation_regime": TRANSDUCTIVE_NOTE,
        "prediction": {"metric": "ISIC 2018 Dice", "student_r3": STUDENT_R3_ISIC_DICE,
                       "threshold": round(STUDENT_R3_ISIC_DICE - PREDICTION_MARGIN, 4)},
        "reads_masks_or_labels": False,
    }


def _abort(message: str):
    raise SystemExit(f"ABORT: {message}")


def _t7_guard(args, record: dict) -> tuple[str, dict]:
    """The registration, the r0 teacher gate and the registered inputs, checked before any training step."""
    try:
        registration = ls.check_registration(REPO)
    except pg.RegistrationError as exc:
        _abort(str(exc))
    try:
        teacher = ls.check_teacher_gate(REPO)
    except ls.TeacherGateError as exc:
        _abort(str(exc))
    registered = {"el manifiesto de terciles": (sha256_file(Path(args.tercile_manifest)), ls.MANIFEST_SHA256),
                  "las pseudo-máscaras": (record["pseudo_masks"]["sha256"], ls.PSEUDO_R0_SHA256),
                  "el manifiesto de la caché": (record["pool"]["manifest_sha256"], ls.CACHE_MANIFEST_SHA256)}
    for what, (digest, expected) in registered.items():
        if digest != expected:
            _abort(f"{what} tiene sha256 {digest[:12]}, no el registrado {expected[:12]}")
    return registration, teacher


def t7_setup(args, record: dict) -> tuple[loop.T7Plan | None, dict | None]:
    """T7's plan and its record block, or (None, None) for every other variant."""
    if args.variant != T7_VARIANT:
        return None, None
    if not args.tercile_manifest:
        raise SystemExit("error: A2-T7 needs --tercile-manifest")
    if args.seed != T7_SEED:
        _abort(f"A2-T7 solo se entrena con la semilla registrada {T7_SEED}, no {args.seed} (el brazo (a) es A2)")
    registration, teacher = _t7_guard(args, record)
    rows = read_manifest(Path(args.cache_dir) / "manifest.csv")
    labels = ls.terciles_by_row(rows, Path(args.tercile_manifest), ls.MANIFEST_SHA256)
    darken = dict(ga.DARKEN_DEFAULTS)  # the registered darkening; no CLI override
    k = 1 if args.no_oversample else args.dark_oversample
    block = {"registration_commit": registration, "tercile_manifest_sha256": ls.MANIFEST_SHA256,
             "oversample_k": k, "teacher_check": teacher, "darken": {key: list(v) if isinstance(v, tuple) else v for key, v in darken.items()}
             | {"terciles": list(ls.DARKENED)}}
    return loop.T7Plan(labels=labels, k=k, darken=darken), block


def _t7_counts(run) -> dict:
    return {"n_dark_train_confident": int(run.dark_train.sum()),
            "n_darken_eligible_train_confident": int(run.eligible[run.train_indices].sum())}


def _check_resume(output_dir: Path, record: dict) -> None:
    """A resumed run must be the same experiment, or the weights mix two runs."""
    previous_path = output_dir / RUN_FILE
    if not (output_dir / "last.pt").is_file() or not previous_path.is_file():
        return
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    changed = [key for key in RESUME_KEYS if previous.get(key) != record.get(key)]
    if changed:
        raise SystemExit(f"ABORT: {output_dir} holds another run; differs in {changed}")


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        revision = rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    context = du.init_distributed(force_cpu=args.force_cpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = base_record(args, revision)
    plan, t7_block = t7_setup(args, record)
    run = loop.Run(args, context, t7=plan)
    record["training"] = loop.confident_counts(run)
    if plan is not None:
        record["t7"] = {**t7_block, **_t7_counts(run)}
    if args.timing_smoke > 0:
        report = {**loop.timing_smoke(run, args.timing_smoke), "code_revision": revision}
        if context.is_main:
            _write(output_dir / SMOKE_FILE, report)
            print(f"{report['seconds_per_step']:.4f} s/step -> "
                  f"{report['projected_gpu_hours']:.2f} GPU-h for {args.epochs} epochs")
        du.cleanup(context)
        return 0
    _check_resume(output_dir, record)
    arch, source = args.backbone
    config = s4.checkpoint_config(run.model, backbone_arch=arch, backbone_source=source,
                                  backbone_sha256=record["backbone"]["weights_sha256"],
                                  resolution=run.resolution)
    if context.is_main:
        _write(output_dir / RUN_FILE, {**record, "complete": False})
    summary = loop.train(run, config)
    if context.is_main:
        _write(output_dir / "timing.json", summary)
        _write(output_dir / RUN_FILE, {**record, "result": summary, "complete": True})
        run_environment.write_sidecar(output_dir / RUN_FILE, args)
    du.cleanup(context)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
