"""C1 step 0: run student_r3 over HAM10000 and pin the resulting masks.

Every one of C1's 24 cells reads the same pseudo-mask file, so this artifact is
produced once, hashed, and referenced by hash from each cell. Without that, two
runs of this script produce files with the same name and different contents and
nothing downstream can tell which one made the published table.

The masks are written through a memmap rather than accumulated in RAM: 10.015
images at 450x600 is ~2.7 GB as uint8, which is enough to matter on a shared
node and turns a cheap inference pass into an OOM at 90 %.

SSL: student_r3 is the path-B U-Net trained without reference masks or dx
labels (docs/23). Its predictions are a self-supervised signal, so using them
as classifier input keeps C1 label-free. Reference masks appear nowhere here.

Inference: this script does not implement inference. It calls the published
``eval_student.load_student`` and ``eval_student.predict_masks`` -- the exact
functions behind the reported Dice -- because an independent implementation
already diverged twice without raising anything:

  * the repo holds two U-Net lineages whose checkpoints disagree on the state
    key (``model`` vs ``model_state_dict``). Only the KeyError stopped the
    wrong lineage from loading;
  * the published path normalises with ImageNet statistics and
    ``dermssl.torch_unet.infer_mask`` does not. Same weights, same image,
    different masks, no error.

Geometry: ``predict_masks`` returns probabilities on the squashed
``resolution x resolution`` grid the network was trained on. C1 needs a mask on
the ORIGINAL grid, so the probabilities are resized bilinearly first and
thresholded after. Thresholding at 224 and resizing labels afterwards would
take the decision on a coarser grid and then interpolate between 0 and 1,
which is a second, undeclared threshold.

WHICH student_r3: the one under ``salidas/final/``, not ``salidas/v2/``. Two
checkpoints share the name on the pod and the repo records both paths. The
published Dice (0,718 ISIC / 0,785 HAM10000) belongs to the FINAL run,
commit d81a492 (docs/24 table "Configuracion principal"); ``salidas/v2/`` is
the rejected max_area_fraction ablation, commit 91814a9, which scored 0,716 /
0,783 and was explicitly closed as a default. Conditioning C1 on the ablation
would mean the segmentation quality cited in the thesis does not describe the
masks the classifier actually saw -- and at a 0,002 Dice gap, nothing
downstream would look wrong.

Usage
-----
    python scripts/c1_predict_masks.py \
        --checkpoint ~/work/proyectonp/salidas/final/student_r3/best.pt \
        --image-dir ~/work/proyectonp/eval_ham10000_images \
        --output-dir ~/work/proyectonp/salidas/c1/masks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# Same recipe as evaluate_segmentation.py: the published evaluation modules
# import each other by bare name, so their folder and comun/nucleo have to be
# importable for eval_student to resolve ckpt_io and gpu_augment.
_EVALUACION = REPO / "plan-segmentacion" / "comun" / "evaluacion"
for _extra in (REPO / "src", _EVALUACION, REPO / "plan-segmentacion" / "comun" / "nucleo"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from dermssl import c1_mask_export as mx  # noqa: E402
from dermssl import c1_mask_provenance as mp  # noqa: E402
from dermssl import cli_args  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MASKS_NAME = "c1_pseudo_masks.npy"
MANIFEST_NAME = "c1_pseudo_masks_manifest.json"
EXPECTED_HAM10000 = 10_015

# Recorded in the manifest so the masks can be told apart from any other run of
# the same checkpoint. TTA stays off because the published Dice is the no-TTA
# number; turning it on here would make C1's masks better than the ones the
# thesis reports.
INFERENCE_PATH = "plan-segmentacion/comun/evaluacion/eval_student.predict_masks"
USE_TTA = False
UPSAMPLE = "bilinear_prob_then_threshold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        type=cli_args.sha256_arg,
        required=True,
        help=(
            "sha256 of the student_r3 checkpoint. Required, with no default: the "
            "script labels its output as produced by student_r3, and that claim has "
            "to be checked rather than assumed. Any compatible U-Net would otherwise "
            "load and be recorded under the wrong producer."
        ),
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detected by default")
    parser.add_argument(
        "--batch-size",
        type=cli_args.positive_int,
        default=32,
        help="images per forward pass; 32 is the default of the published evaluation",
    )
    parser.add_argument(
        "--limit",
        type=cli_args.positive_int,
        default=None,
        help="process only the first N images; for smoke runs, never for the real artifact",
    )
    parser.add_argument(
        "--allow-unexpected-count",
        action="store_true",
        help=f"proceed even if the folder does not hold {EXPECTED_HAM10000} images",
    )
    return parser.parse_args()


def discover_images(image_dir: Path) -> list[Path]:
    """Deterministic, sorted image list -- the mask index is positional."""
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"ABORT: no images under {image_dir}")
    return paths


def code_revision(repo: Path) -> str:
    import subprocess  # noqa: PLC0415

    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _published_threshold() -> float:
    """The published 0,5, read from the published module rather than copied.

    A literal here would keep saying 0,5 after someone changed the protocol,
    and the manifest would document a threshold the masks were not built with.
    """
    from evaluate_segmentation import DEFAULT_THRESHOLD  # noqa: PLC0415

    return float(DEFAULT_THRESHOLD)


def original_shape(path: Path) -> tuple[int, int]:
    """``(height, width)`` of the file on disk, without decoding the pixels."""
    from PIL import Image  # noqa: PLC0415

    with Image.open(path) as image:
        width, height = image.size
    return height, width


def assert_probability_map(array: np.ndarray) -> None:
    """Raise unless ``array`` is a finite probability map in [0, 1].

    ``predict_masks`` returns ``torch.sigmoid`` output, so this holds today.
    The contract lives at the callee anyway, for the same reason
    ``image_contract`` does: whoever changes the inference path next will not
    re-derive the invariant, and every way of breaking it is silent.

    NaN is the sharpest case. PIL propagates it through the resize and
    ``nan > 0.5`` is False, so a NaN map becomes an EMPTY mask -- which then
    trips the bbox and CLS fallbacks and reads as a hard image rather than as
    a broken forward pass. A range check alone would not catch it either:
    every comparison against NaN is False, so it slips past min/max bounds.
    """
    if array.size == 0:
        raise ValueError(
            f"el mapa de probabilidades tiene forma {array.shape} y ningun pixel. "
            "Un array vacio pasa todo chequeo de rango de forma vacua y llegaria "
            "a la mascara como un no-op silencioso."
        )
    if not np.isfinite(array).all():
        raise ValueError(
            "el mapa de probabilidades tiene NaN o infinito. Un rango no alcanza: "
            "toda comparacion contra NaN es False, asi que pasaria los limites y "
            "`nan > 0,5` produciria una mascara VACIA sin ningun error."
        )
    low, high = float(array.min()), float(array.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(
            f"el mapa de probabilidades va de {low:.4g} a {high:.4g}, fuera de [0, 1]. "
            "Si son logits en vez de probabilidades, todos los pixeles superan 0,5 "
            "de una y la mascara sale entera."
        )


def upsample_probabilities(probabilities: np.ndarray, height: int, width: int) -> np.ndarray:
    """Bilinear resize of the PROBABILITY map back to the original grid.

    Resizing before the threshold keeps the 0,5 decision as the only one; the
    alternative interpolates between two labels and calls the result a mask.
    PIL rather than a hand-rolled kernel, because the published pipeline
    resizes through PIL too and the two have to agree on pixel centres.
    """
    from PIL import Image  # noqa: PLC0415

    array = np.asarray(probabilities, dtype=np.float32)
    # Before the early return, not after: a map that already matches the target
    # shape skips the resize, and skipping the check with it is how the one
    # case nobody tests becomes the one case nobody validates.
    assert_probability_map(array)
    if array.shape == (height, width):
        return array
    resized = Image.fromarray(array, mode="F").resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def resolve_device(requested: str | None):
    import torch  # noqa: PLC0415

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_all(
    paths: list[Path],
    checkpoint: Path,
    device,
    masks_path: Path,
    batch_size: int,
) -> tuple[np.ndarray, list[str], int]:
    """Write one uint8 mask per image into a memmapped .npy, in path order.

    Returns the training resolution alongside the masks so the manifest states
    the grid the network actually saw rather than a constant copied here.
    """
    import eval_student  # noqa: PLC0415
    from eval_inputs import load_image  # noqa: PLC0415
    from evaluate_segmentation import DEFAULT_THRESHOLD, binarize  # noqa: PLC0415

    model, resolution = eval_student.load_student(checkpoint, device)

    height, width = original_shape(paths[0])
    masks = np.lib.format.open_memmap(
        masks_path, mode="w+", dtype=np.uint8, shape=(len(paths), height, width)
    )

    image_ids: list[str] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        for path in chunk:
            shape = original_shape(path)
            if shape != (height, width):
                raise SystemExit(
                    f"ABORT: {path.name} is {shape} but the artifact is "
                    f"{(height, width)}. A ragged stack cannot be indexed positionally; "
                    "resize upstream or store per-image files instead."
                )
        images = np.stack([load_image(path, resolution) for path in chunk])
        probabilities = eval_student.predict_masks(
            model, images, device, len(chunk), tta=USE_TTA
        )
        for offset, (path, probability) in enumerate(zip(chunk, probabilities, strict=True)):
            full = upsample_probabilities(probability, height, width)
            masks[start + offset] = mx.encode_mask(binarize(full, DEFAULT_THRESHOLD))
            image_ids.append(path.stem)

        done = start + len(chunk)
        if done % 500 < batch_size:
            print(f"  {done}/{len(paths)}", flush=True)

    masks.flush()
    return masks, image_ids, resolution


def main() -> int:
    args = parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not checkpoint.exists():
        raise SystemExit(f"ABORT: no checkpoint at {checkpoint}")

    checkpoint_sha = mx.sha256_file(checkpoint)
    # Derive the run from the file's content BEFORE touching the memmap: the v2
    # ablation loads exactly as well as the published checkpoint, so the only
    # thing that distinguishes them is the hash.
    segmentation_run, segmentation_commit = mp.resolve_run(checkpoint_sha)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise SystemExit(
            f"ABORT: checkpoint sha256 mismatch.\n"
            f"  expected {args.expected_checkpoint_sha256}\n"
            f"  found    {checkpoint_sha}\n"
            "This artifact would be labelled 'student_r3' without being it."
        )

    paths = discover_images(image_dir)
    partial = False
    if args.limit is not None:
        paths = paths[: args.limit]
        partial = True
        print(f"WARNING: --limit {args.limit} -- smoke artifact, not the real one")
    elif len(paths) != EXPECTED_HAM10000 and args.allow_unexpected_count:
        partial = True
    elif len(paths) != EXPECTED_HAM10000:
        raise SystemExit(
            f"ABORT: found {len(paths)} images, expected {EXPECTED_HAM10000}. "
            "A partial folder would silently produce a partial artifact; pass "
            "--allow-unexpected-count if this is deliberate."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_path = output_dir / MASKS_NAME

    device = resolve_device(args.device)
    print(f"device     : {device}")
    print(f"checkpoint : {checkpoint.name}")
    print(f"images     : {len(paths)}")
    print(f"inference  : {INFERENCE_PATH} (tta={USE_TTA}, upsample={UPSAMPLE})")

    try:
        masks, image_ids, resolution = predict_all(
            paths, checkpoint, device, masks_path, args.batch_size
        )
    except BaseException:
        # A half-written .npy has a valid header and the wrong contents: it would
        # hash fine on its own and silently become the input to all 24 cells.
        masks_path.unlink(missing_ok=True)
        raise
    coverage = mx.summarise_coverage(np.asarray(masks))
    del masks  # release the memmap before hashing the file

    manifest = mx.build_masks_manifest(
        n_images=len(image_ids),
        image_ids=image_ids,
        masks_sha256=mx.sha256_file(masks_path),
        checkpoint_sha256=checkpoint_sha,
        segmentation_run=segmentation_run,
        segmentation_commit=segmentation_commit,
        partial=partial,
        checkpoint_path=str(checkpoint),
        code_revision=code_revision(REPO),
        mask_shape=list(np.load(masks_path, mmap_mode="r").shape[1:]),
        coverage=coverage,
        inference={
            "path": INFERENCE_PATH,
            "tta": USE_TTA,
            "resolution": resolution,
            "threshold": _published_threshold(),
            "upsample": UPSAMPLE,
        },
    )
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== STEP 0 OK ===")
    print(f"masks     : {masks_path}")
    print(f"sha256    : {manifest['masks_sha256']}")
    print(f"empty     : {coverage['n_empty']}/{coverage['n_images']}")
    print(f"mean fg   : {coverage['mean_foreground_fraction']:.4f}")
    if coverage["n_empty"]:
        print(
            "NOTE: empty masks force the bbox fallback in (a) and the CLS fallback in (c). "
            "If this count is large, those conditions are collapsing toward the baseline."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
