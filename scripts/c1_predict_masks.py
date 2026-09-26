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
sys.path.insert(0, str(REPO / "src"))

from dermssl import c1_mask_export as mx  # noqa: E402
from dermssl import c1_mask_provenance as mp  # noqa: E402
from dermssl import cli_args  # noqa: E402

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
MASKS_NAME = "c1_pseudo_masks.npy"
MANIFEST_NAME = "c1_pseudo_masks_manifest.json"
EXPECTED_HAM10000 = 10_015


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


def load_image(path: Path) -> np.ndarray:
    """RGB as float32 in [0, 1] -- the scale student_r3 was trained on.

    ``infer_mask_with_confidence`` does NOT divide by 255 (it expects
    to_tensor output), so handing it a raw uint8 array saturates the network
    and returns masks that look plausible and are not. The callee now rejects
    that, but dividing here is the actual fix.
    """
    from PIL import Image  # noqa: PLC0415

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


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
) -> tuple[np.ndarray, list[str]]:
    """Write one uint8 mask per image into a memmapped .npy, in path order."""
    from dermssl.torch_unet import infer_mask, load_checkpoint  # noqa: PLC0415

    model = load_checkpoint(str(checkpoint), device)

    first = load_image(paths[0])
    height, width = first.shape[:2]
    masks = np.lib.format.open_memmap(
        masks_path, mode="w+", dtype=np.uint8, shape=(len(paths), height, width)
    )

    image_ids: list[str] = []
    for index, path in enumerate(paths):
        image = first if index == 0 else load_image(path)
        if image.shape[:2] != (height, width):
            raise SystemExit(
                f"ABORT: {path.name} is {image.shape[:2]} but the artifact is "
                f"{(height, width)}. A ragged stack cannot be indexed positionally; "
                "resize upstream or store per-image files instead."
            )
        masks[index] = mx.encode_mask(infer_mask(model, image, device))
        image_ids.append(path.stem)

        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{len(paths)}", flush=True)

    masks.flush()
    return masks, image_ids


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

    try:
        masks, image_ids = predict_all(paths, checkpoint, device, masks_path)
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
