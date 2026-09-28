r"""S-diag: DINOv2 lesion masks with no training, for the external evaluator.

Method registered in docs/27 on 2026-09-28 (rules in ``dermssl.s_diag``):
``primary`` = per-image PCA at 224 (16x16 patches), ``kmeans`` = k-means k=2
at 224, ``pca448`` = the same PCA at 448 (32x32 patches). Each image is
resized to the input resolution as the evaluator does (bilinear, no crop, so
patches stay aligned with the reference grid), ImageNet-normalised like the
frozen bank, and passed once through the frozen backbone in fp32. The masks
are written as ``masks/<image_id>_<method>.png`` (0/255, 224x224), the layout
``evaluate_segmentation.py --pred-mask-dir`` already reads for the SegRNorm /
SegXNorm floors, so Dice, IoU and HD95 come from the same code as student_r3.

Pod run (3 weights x 3 methods x 2 reference sets). The clone must be clean
(``git status --porcelain`` empty) or the script aborts; TORCH_HOME must hold
the cached facebookresearch/dinov2 hub repo and weights, as for C1. The GPU
forward stays batched; ``--workers 36`` spreads the per-image PCA / k-means /
PNG work over 36 single-threaded processes (bit-identical to ``--workers 1``;
check it first with ``scripts/s_diag_parallel_check.py``). Order: every
``primary`` run first, then ``pca448``, then ``kmeans``, each (set, weight,
method) evaluated as soon as its masks exist::

    cd ~/work/proyectonp/repo_v7 && source ~/work/proyectonp/.venv/bin/activate
    python -c "import sklearn, skimage, scipy"  # else pip install scikit-learn scikit-image
    P=$HOME/work/proyectonp; V=$P/salidas/v7; S=$P/salidas/s_diag
    declare -A W=([public]=public
                  [astar]=$V/final_A/simsiam_dinov2_backbone_final.pt
                  [bstar]=$V/final_B/simsiam_dinov2_backbone_final.pt)
    declare -A IMG=([isic2018]=$P/datos_isic2018/ISIC2018_Task1-2_Training_Input
                    [ham10000]=$P/eval_ham10000_images)
    declare -A GT=([isic2018]=$P/masks_eval/isic2018/ground_truth
                   [ham10000]=$P/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl)
    for METHOD in primary pca448 kmeans; do
      for SET in isic2018 ham10000; do
        for NAME in public astar bstar; do
          RUN=$S/$SET/${NAME}_$METHOD; E=$S/evaluacion_${SET}_${NAME}_${METHOD}
          python scripts/s_diag_predict.py --backbone $NAME=dinov2_vitb14:${W[$NAME]} \
            --image-dir ${IMG[$SET]} --reference-set $SET --method $METHOD \
            --workers 36 --output-dir $RUN || break 3
          python plan-segmentacion/comun/evaluacion/evaluate_segmentation.py \
            --pred-mask-dir $RUN/masks --pred-mask-suffix _$METHOD \
            --image-dir ${IMG[$SET]} --mask-dir ${GT[$SET]} --eval-resolution 224 \
            --hausdorff --eval-name ${SET}_sdiag_${NAME}_${METHOD} --output-dir $E \
            --overlay-dir $E/overlays || break 3
        done
      done
    done

Smoke first: add ``--limit 16`` and an ``--output-dir`` under /tmp; the
evaluator refuses a partial set, so the smoke only checks the predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _extra in (REPO / "src", REPO / "scripts", REPO / "plan-segmentacion" / "comun" / "evaluacion"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

import cluster_frozen_bank as bank  # noqa: E402
import eval_inputs  # noqa: E402
from dermssl import run_provenance as rp  # noqa: E402
from dermssl import s_diag as sd  # noqa: E402

PATCH_SIZE = 14
# Same grid as evaluate_segmentation.EXTERNAL_RESOLUTION: masks land on it unchanged.
MASK_RESOLUTION = 224
KMEANS_SEED = 0
# Every image's PCA / k-means runs on one BLAS / OpenMP thread, serial or not:
# multi-threaded BLAS moves the PCA projection by ~1e-16, which could flip an
# Otsu tie, and N workers must use N cores, not N x (all cores) threads.
CPU_THREADS_PER_IMAGE = 1
WORKER_THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
# Images whose masks may be pending in the pool while the GPU extracts more.
BACKLOG_PER_WORKER = 4
METHODS = {"primary": ("pca", 224), "kmeans": ("kmeans", 224), "pca448": ("pca", 448)}
TRAINING_NOTE = (
    "no training: no parameter is trained or fitted on any image set; PCA and "
    "k-means are fitted per image on that image's own patch tokens, and no "
    "reference mask is read by this script"
)


def input_resolution(methods: list[str]) -> int:
    """The single backbone input resolution shared by ``methods``."""
    resolutions = {METHODS[method][1] for method in methods}
    if len(resolutions) != 1:
        raise ValueError(f"{methods} need one input resolution each; run them separately")
    return resolutions.pop()


def weights_digest(source: str) -> str:
    if source == bank.BACKBONE_SOURCE_PUBLIC:
        return "public"
    return rp.file_sha256(Path(source).expanduser())


def list_images(image_dir: Path, limit: int) -> list[Path]:
    """Every image the evaluator would pair, in its sorted order."""
    images = sorted(
        p for p in Path(image_dir).iterdir() if p.suffix.lower() in eval_inputs.IMAGE_EXTENSIONS
    )
    return images[:limit] if limit > 0 else images


def extract_tokens(model, paths: list[Path], resolution: int, device) -> np.ndarray:
    """Last-layer ``x_norm_patchtokens`` for a batch, fp32, (B, N, D)."""
    import torch

    from dermssl.torch_c1_extraction import to_normalized_tensor

    batch = torch.stack(
        [to_normalized_tensor(eval_inputs.load_image(path, resolution)) for path in paths]
    ).to(device)
    with torch.no_grad():
        tokens = model.forward_features(batch)["x_norm_patchtokens"]
    return tokens.float().cpu().numpy()


def _mask_args(masks_dir: Path, image_id: str, tokens: np.ndarray, methods, grid: int):
    kinds = {method: METHODS[method][0] for method in methods}
    shape = (MASK_RESOLUTION, MASK_RESOLUTION)
    return masks_dir, image_id, tokens, kinds, grid, shape, KMEANS_SEED, CPU_THREADS_PER_IMAGE


def write_masks(masks_dir: Path, image_id: str, tokens: np.ndarray, methods, grid: int) -> None:
    sd.write_mask_pngs(*_mask_args(masks_dir, image_id, tokens, methods, grid))


def _raise_failures(futures) -> None:
    for future in futures:
        future.result()


@contextmanager
def mask_writer(workers: int):
    """``submit(*write_masks_args)``: inline when serial, else a loky process pool.

    Each image's masks depend only on its own tokens (k-means keeps
    ``random_state`` per image, both paths use one CPU thread per image), so
    the pool writes the same bytes as the serial loop. The backlog is bounded so the GPU extraction of the next
    batches overlaps the workers without holding every token array in memory.
    """
    if workers == 1:
        yield write_masks
        return
    from joblib.externals.loky import get_reusable_executor

    executor = get_reusable_executor(max_workers=workers, env=WORKER_THREAD_ENV)
    pending = set()

    def submit(*write_args) -> None:
        pending.add(executor.submit(sd.write_mask_pngs, *_mask_args(*write_args)))
        if len(pending) > workers * BACKLOG_PER_WORKER:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            pending.difference_update(done)
            _raise_failures(done)

    try:
        yield submit
        _raise_failures(wait(pending).done)
    finally:
        executor.shutdown(wait=True)


def predict(args, masks_dir: Path, resolution: int) -> list[str]:
    """Load the frozen backbone once and write every image's masks."""
    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, arch, source = args.backbone
    model = bank.build_frozen_backbone(arch, source, resolution, device)["model"]
    grid = resolution // PATCH_SIZE
    paths = list_images(args.image_dir, args.limit)
    with mask_writer(args.workers) as submit:
        for start in range(0, len(paths), args.batch_size):
            chunk = paths[start : start + args.batch_size]
            for path, tokens in zip(chunk, extract_tokens(model, chunk, resolution, device)):
                if tokens.shape[0] != grid * grid:
                    raise ValueError(f"{path.name}: {tokens.shape[0]} tokens, expected {grid * grid}")
                submit(masks_dir, path.stem, tokens, list(args.method), grid)
    return [path.stem for path in paths]


def run_record(args, resolution: int, ids: list[str], revision: str) -> dict:
    name, arch, source = args.backbone
    return {
        "experiment": "S-diag (docs/27, method registered 2026-09-28)",
        "methods": list(args.method),
        "input_resolution": resolution,
        "grid": resolution // PATCH_SIZE,
        "mask_resolution": MASK_RESOLUTION,
        "backbone": {"name": name, "spec": f"{arch}:{source}", "weights_sha256": weights_digest(source)},
        "tokens": "forward_features()['x_norm_patchtokens'], last layer, fp32",
        "preprocessing": "bilinear resize to input_resolution (no crop), ImageNet mean/std",
        "kmeans": {"k": sd.KMEANS_CLUSTERS, "n_init": sd.KMEANS_N_INIT, "random_state": KMEANS_SEED},
        "code_revision": revision,
        "reference_set": args.reference_set,
        "image_dir": str(args.image_dir),
        "n_images": len(ids),
        "image_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
        "limit": args.limit,
        "workers": args.workers,
        "cpu_threads_per_image": CPU_THREADS_PER_IMAGE,
        "torch_home": os.environ.get("TORCH_HOME", ""),
        "training_images": TRAINING_NOTE,
        "complete": True,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--backbone", required=True, type=bank.parse_backbone_spec,
                        metavar="NAME=ARCH:SOURCE", help="SOURCE is 'public' or a weights file")
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-set", required=True, choices=("isic2018", "ham10000"))
    parser.add_argument("--method", action="append", required=True, choices=tuple(METHODS))
    parser.add_argument("--limit", type=int, default=0, help="first N sorted images (smoke)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="processes for the per-image masks; 1 = serial, same bytes")
    args = parser.parse_args(argv)
    if len(set(args.method)) != len(args.method):
        parser.error("--method repeated")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        resolution = input_resolution(args.method)
    except ValueError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    out = args.output_dir.expanduser()
    if out.exists():
        raise SystemExit(f"ABORT: {out} already exists; S-diag outputs are never overwritten")
    try:
        revision = rp.clean_revision(REPO)
    except rp.DirtyTreeError as exc:
        raise SystemExit(f"ABORT: {exc}") from exc
    masks_dir = out / "masks"
    masks_dir.mkdir(parents=True)
    ids = predict(args, masks_dir, resolution)
    record = run_record(args, resolution, ids, revision)
    (out / "run.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("methods", "reference_set", "n_images")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
