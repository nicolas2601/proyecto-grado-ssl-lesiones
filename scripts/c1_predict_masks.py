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
import math
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

from dermssl import c1_extraction as ex  # noqa: E402
from dermssl import c1_mask_export as mx  # noqa: E402
from dermssl import c1_mask_provenance as mp  # noqa: E402
from dermssl import cli_args  # noqa: E402
from dermssl.pretrain import tracking  # noqa: E402

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

# Fixed in docs/27 §10.3 BEFORE the data, so they are a prediction rather than
# a rationalisation of whatever came out. The bank geometry is not changed --
# it is the one the v6 numbers came from -- it is measured, and if it costs
# more than this we stop and decide again with the number in hand.
MAX_FRACTION_EMPTIED_BY_CROP = 0.02
MAX_FRACTION_BELOW_HALF = 0.05


# The artifact has to BE the published segmentation, not merely resemble it.
# Same checkpoint, same inference, same images: the per-image Dice has to come
# out identical, and the residue is cuDNN non-determinism (4e-05 measured on
# ISIC). 1e-3 leaves two orders of magnitude of room above that and still
# catches a real divergence.
MAX_ABS_DIFF_VS_PUBLISHED = 1e-3


def evaluate_reference_gate(paired: dict) -> tuple[bool, list[str]]:
    """``(passes, reasons)`` for the per-image comparison against HAM10000.

    Paired by ``image_id`` against the published ``per_image.csv``, never a
    comparison of means: a mean can match while individual images are wrong,
    and with a non-random subset it does not even match (the first 20 ISIC
    images average 0,855 against a published 0,718).

    An incomplete join FAILS rather than averaging over whatever matched, and
    a non-finite drift FAILS rather than slipping through -- ``nan > 1e-3`` is
    False, which is exactly how a gate turns into a no-op.
    """
    reasons: list[str] = []

    n_matched = paired["n_matched"]
    n_published = paired["n_published"]
    if n_matched == 0:
        reasons.append(
            "no se emparejo ninguna imagen con el per_image.csv publicado: sin "
            "comparacion no hay gate aprobado"
        )
    elif n_matched != n_published:
        reasons.append(
            f"solo se emparejaron {n_matched} de {n_published} imagenes publicadas; "
            "un join parcial promedia sobre las que casualmente coincidieron"
        )

    drift = paired["max_abs_diff"]
    if not math.isfinite(drift):
        reasons.append(
            f"max_abs_diff es {drift}: toda comparacion contra NaN es False, asi que "
            "el umbral no lo frenaria y el gate quedaria en no-op"
        )
    elif drift > MAX_ABS_DIFF_VS_PUBLISHED:
        reasons.append(
            f"max_abs_diff {drift:.3e} por encima de {MAX_ABS_DIFF_VS_PUBLISHED:.0e}: "
            "el artefacto NO reproduce la segmentacion publicada"
        )

    return not reasons, reasons


class ReferenceVerifier:
    """Dice at 224 against the reference, computed INSIDE the prediction loop.

    It has to happen here, not afterwards from the ``.npy``. Re-squashing the
    finished original-resolution mask back to 224 does not reproduce the
    published mask: measured Dice 0,95 between them, 2504 of 50176 pixels
    different. Thresholding probabilities at 224 and downsampling a binary
    mask to 224 are different operations, and only the first is what produced
    the published number.

    The reference masks are read ONLY to verify the artifact. They never reach
    C1, which is what keeps the arm self-supervised.
    """

    def __init__(self, mask_dir: Path, published_csv: Path, resolution: int):
        self._mask_dir = Path(mask_dir)
        self._resolution = resolution
        self._published = self._load_published(Path(published_csv))
        self._diffs: list[float] = []
        self._matched: list[str] = []

    @staticmethod
    def _load_published(path: Path) -> dict[str, float]:
        import csv  # noqa: PLC0415

        with open(path, newline="", encoding="utf-8") as handle:
            return {row["image_id"]: float(row["dice"]) for row in csv.DictReader(handle)}

    def record_batch(self, image_ids, probabilities) -> None:
        from eval_inputs import load_binary_mask  # noqa: PLC0415
        from evaluate_segmentation import (  # noqa: PLC0415
            DEFAULT_THRESHOLD,
            binarize,
            compute_binary_metrics,
        )

        for image_id, probability in zip(image_ids, probabilities, strict=True):
            expected = self._published.get(image_id)
            if expected is None:
                continue  # not in the published evaluation; nothing to compare against
            mask_path = self._mask_dir / f"{image_id}_segmentation.png"
            if not mask_path.is_file():
                continue
            truth = load_binary_mask(mask_path, self._resolution)
            predicted = binarize(probability, DEFAULT_THRESHOLD)
            dice = compute_binary_metrics(predicted, truth)["dice"]
            self._diffs.append(abs(dice - expected))
            self._matched.append(image_id)

    def summary(self) -> dict:
        diffs = np.asarray(self._diffs, dtype=np.float64)
        return {
            "n_matched": len(self._matched),
            "n_published": len(self._published),
            # NaN rather than 0,0 with nothing recorded: a max over an empty
            # set is not "no drift", and evaluate_reference_gate refuses both.
            "max_abs_diff": float(diffs.max()) if diffs.size else float("nan"),
            "mean_abs_diff": float(diffs.mean()) if diffs.size else float("nan"),
        }


def evaluate_crop_gates(summary: dict) -> tuple[bool, list[str]]:
    """``(passes, reasons)`` for the crop-retention gate.

    Every failing condition is reported, not just the first: stopping at one
    hides how bad the rest is, and the decision afterwards needs the whole
    picture. ``n_empty_from_segmenter`` is deliberately NOT gated here -- a
    mask that arrived empty is the segmenter finding nothing, which is a
    different problem with a different owner, and it is reported separately.
    """
    reasons: list[str] = []

    emptied = summary["fraction_emptied_by_crop"]
    if emptied > MAX_FRACTION_EMPTIED_BY_CROP:
        reasons.append(
            f"n_emptied_by_crop: {emptied:.2%} de las imagenes tenia lesion y la perdio "
            f"entera en el recorte, por encima del {MAX_FRACTION_EMPTIED_BY_CROP:.0%}"
        )

    below_half = summary["fraction_below_half"]
    if below_half is None:
        reasons.append(
            "fraction_below_half es None: ninguna mascara tenia lesion, asi que no hay "
            "tasa que evaluar. Un lote sin lesiones no es un gate aprobado."
        )
    elif below_half > MAX_FRACTION_BELOW_HALF:
        reasons.append(
            f"n_below_half: {below_half:.2%} de las mascaras CON lesion retiene menos de "
            f"la mitad, por encima del {MAX_FRACTION_BELOW_HALF:.0%}"
        )

    return not reasons, reasons


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
    parser.add_argument(
        "--wandb-project",
        default="",
        help="proyecto de Weights & Biases; si se pide y no se puede registrar, aborta",
    )
    parser.add_argument("--wandb-group", default="v7-c1", help="grupo del run en W&B")
    parser.add_argument(
        "--reference-mask-dir",
        type=Path,
        default=None,
        help=(
            "mascaras de referencia de HAM10000, SOLO para verificar el artefacto. "
            "Nunca entran a C1; se usan para reproducir el Dice publicado."
        ),
    )
    parser.add_argument(
        "--published-per-image",
        type=Path,
        default=None,
        help="per_image.csv de la evaluacion publicada, para la comparacion pareada",
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
    verifier: "ReferenceVerifier | None" = None,
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
        if verifier is not None:
            # While the 224 probabilities still exist: a mask re-squashed from
            # the original grid is NOT the published mask (Dice 0,95 between
            # them), so this cannot be done afterwards from the .npy.
            verifier.record_batch([path.stem for path in chunk], probabilities)
        for offset, (path, probability) in enumerate(zip(chunk, probabilities, strict=True)):
            full = upsample_probabilities(probability, height, width)
            masks[start + offset] = mx.encode_mask(binarize(full, DEFAULT_THRESHOLD))
            image_ids.append(path.stem)

        done = start + len(chunk)
        if done % 500 < batch_size:
            print(f"  {done}/{len(paths)}", flush=True)

    masks.flush()
    return masks, image_ids, resolution


def build_step0_tracker(wandb_project: str, group: str, environ=None, importer=None):
    """Strict W&B tracker for step 0, or ``None`` when tracking was not asked for.

    Strict because this run produces the artifact all 24 cells consume: if it
    cannot be registered, nobody can point at where it came from afterwards.
    """
    extra = {} if importer is None else {"importer": importer}
    module = tracking.require_wandb(wandb_project, environ=environ, **extra)
    if module is None:
        return None
    return tracking.WandbTracker(
        module, project=wandb_project, run="c1-step0-masks", group=group, strict=True
    )


def main() -> int:
    args = parse_args()

    # Before torch, before reading 10.015 images: a run that cannot be
    # registered produces an artifact nobody can trace, and tracing it is the
    # whole point of this script.
    tracker = build_step0_tracker(args.wandb_project, args.wandb_group)

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

    # Both or neither: half the inputs would silently skip the check.
    if bool(args.reference_mask_dir) != bool(args.published_per_image):
        raise SystemExit(
            "ABORT: --reference-mask-dir y --published-per-image van juntos. Con uno "
            "solo la verificacion contra la segmentacion publicada se saltea en silencio."
        )
    verifier = None
    if args.reference_mask_dir:
        verifier = ReferenceVerifier(
            mask_dir=args.reference_mask_dir.expanduser(),
            published_csv=args.published_per_image.expanduser(),
            resolution=224,
        )

    try:
        masks, image_ids, resolution = predict_all(
            paths, checkpoint, device, masks_path, args.batch_size, verifier
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

    # How much lesion the bank crop costs, measured on the artifact that C1
    # will actually consume rather than estimated from a worked example.
    stored = np.load(masks_path, mmap_mode="r")
    retention = ex.summarise_crop_retention(
        np.asarray(stored[i]) == mx.LESION_LABEL for i in range(len(image_ids))
    )
    passes, reasons = evaluate_crop_gates(retention)

    reference = verifier.summary() if verifier is not None else None
    if reference is not None:
        reference_ok, reference_reasons = evaluate_reference_gate(reference)
        passes = passes and reference_ok
        reasons = reasons + reference_reasons

    if tracker is not None:
        tracker.start(
            {
                "artifact": "c1_pseudo_masks",
                "checkpoint_sha256": checkpoint_sha,
                "masks_sha256": manifest["masks_sha256"],
                "segmentation_run": segmentation_run,
                "segmentation_commit": segmentation_commit,
                "code_revision": manifest["code_revision"],
                "n_images": len(image_ids),
                "partial": partial,
                **{f"inference_{k}": v for k, v in manifest["inference"].items()},
            }
        )
        for key in ("n_empty", "mean_foreground_fraction"):
            tracker.log_metric(f"coverage_{key}", float(coverage[key]))
        for key in (
            "n_empty_from_segmenter",
            "n_emptied_by_crop",
            "n_below_half",
            "n_with_lesion",
            "fraction_empty_from_segmenter",
            "fraction_emptied_by_crop",
        ):
            tracker.log_metric(f"crop_{key}", float(retention[key]))
        if retention["fraction_below_half"] is not None:
            tracker.log_metric("crop_fraction_below_half", retention["fraction_below_half"])
        # Logged whether it passed or failed: a dashboard that only holds the
        # runs that cleared the gate is evidence for one side.
        if reference is not None:
            tracker.log_metric("reference_n_matched", float(reference["n_matched"]))
            tracker.log_metric("reference_max_abs_diff", reference["max_abs_diff"])
            tracker.log_metric("reference_mean_abs_diff", reference["mean_abs_diff"])
        tracker.log_metric("crop_gate_passes", 1.0 if passes else 0.0)
        tracker.end()
        if tracker.run_url:
            print(f"\nW&B run: {tracker.run_url}")

    print("\n=== STEP 0 ===")
    print(f"masks     : {masks_path}")
    print(f"sha256    : {manifest['masks_sha256']}")
    print(f"empty     : {coverage['n_empty']}/{coverage['n_images']}")
    print(f"mean fg   : {coverage['mean_foreground_fraction']:.4f}")
    print("\ncrop retention (bank geometry: Resize(255) + CenterCrop(224))")
    print(f"  empty from segmenter : {retention['n_empty_from_segmenter']}/{retention['n_images']}"
          f" ({retention['fraction_empty_from_segmenter']:.2%})")
    print(f"  emptied by the crop  : {retention['n_emptied_by_crop']}/{retention['n_images']}"
          f" ({retention['fraction_emptied_by_crop']:.2%})")
    if retention["fraction_below_half"] is not None:
        print(f"  retains under half   : {retention['n_below_half']}/{retention['n_with_lesion']}"
              f" ({retention['fraction_below_half']:.2%} of those WITH a lesion)")

    if coverage["n_empty"]:
        print(
            "\nNOTE: empty masks force the bbox fallback in (a) and the CLS fallback in (c). "
            "If this count is large, those conditions are collapsing toward the baseline."
        )

    if reference is not None:
        print("\nverificacion contra la segmentacion publicada (solo para verificar)")
        print(f"  emparejadas          : {reference['n_matched']}/{reference['n_published']}")
        print(f"  |delta| medio        : {reference['mean_abs_diff']:.3e}")
        print(f"  |delta| maximo       : {reference['max_abs_diff']:.3e}")

    if not passes:
        print("\n=== GATE FAILED — DO NOT RUN THE BANK GATE OR C1 ===")
        for reason in reasons:
            print(f"  {reason}")
        print(
            "\nFrenar no significa cambiar la geometria del banco: significa volver a "
            "decidir con el dato en la mano (docs/27 §10.3)."
        )
        return 1

    print("\n=== STEP 0 OK — crop gate passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
