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
the cached facebookresearch/dinov2 hub repo and weights, as for C1::

    cd ~/work/proyectonp/repo_v7 && source ~/work/proyectonp/.venv/bin/activate
    python -c "import sklearn, skimage, scipy"  # else pip install scikit-learn scikit-image
    P=$HOME/work/proyectonp; V=$P/salidas/v7; S=$P/salidas/s_diag
    declare -A W=([public]=public
                  [astar]=$V/final_A/simsiam_dinov2_backbone_final.pt
                  [bstar]=$V/final_B/simsiam_dinov2_backbone_final.pt)
    for SET in isic2018 ham10000; do
      case $SET in
        isic2018) IMG=$P/datos_isic2018/ISIC2018_Task1-2_Training_Input
                  GT=$P/masks_eval/isic2018/ground_truth ;;
        ham10000) IMG=$P/eval_ham10000_images
                  GT=$P/masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl ;;
      esac
      for NAME in public astar bstar; do
        SPEC=$NAME=dinov2_vitb14:${W[$NAME]}
        python scripts/s_diag_predict.py --backbone $SPEC --image-dir $IMG \
          --reference-set $SET --method primary --method kmeans \
          --output-dir $S/$SET/${NAME}_224
        python scripts/s_diag_predict.py --backbone $SPEC --image-dir $IMG \
          --reference-set $SET --method pca448 --output-dir $S/$SET/${NAME}_448
        for METHOD in primary kmeans pca448; do
          RUN=${NAME}_224; [ $METHOD = pca448 ] && RUN=${NAME}_448
          E=$S/evaluacion_${SET}_${NAME}_${METHOD}
          python plan-segmentacion/comun/evaluacion/evaluate_segmentation.py \
            --pred-mask-dir $S/$SET/$RUN/masks --pred-mask-suffix _$METHOD \
            --image-dir $IMG --mask-dir $GT --eval-resolution 224 --hausdorff \
            --eval-name ${SET}_sdiag_${NAME}_${METHOD} --output-dir $E \
            --overlay-dir $E/overlays
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


def write_masks(masks_dir: Path, image_id: str, tokens: np.ndarray, methods, grid: int) -> None:
    from PIL import Image

    shape = (MASK_RESOLUTION, MASK_RESOLUTION)
    for method in methods:
        mask = sd.segment(METHODS[method][0], tokens, grid, shape, seed=KMEANS_SEED)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            masks_dir / f"{image_id}_{method}.png"
        )


def predict(args, masks_dir: Path, resolution: int) -> list[str]:
    """Load the frozen backbone once and write every image's masks."""
    import torch

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _, arch, source = args.backbone
    model = bank.build_frozen_backbone(arch, source, resolution, device)["model"]
    grid = resolution // PATCH_SIZE
    paths = list_images(args.image_dir, args.limit)
    for start in range(0, len(paths), args.batch_size):
        chunk = paths[start : start + args.batch_size]
        for path, tokens in zip(chunk, extract_tokens(model, chunk, resolution, device)):
            if tokens.shape[0] != grid * grid:
                raise ValueError(f"{path.name}: {tokens.shape[0]} tokens, expected {grid * grid}")
            write_masks(masks_dir, path.stem, tokens, args.method, grid)
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
    args = parser.parse_args(argv)
    if len(set(args.method)) != len(args.method):
        parser.error("--method repeated")
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
