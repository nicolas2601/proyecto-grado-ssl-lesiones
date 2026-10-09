"""Pod-side step of figure 3: image, reference mask and the two predicted masks of the 6 qualitative cases.

Runs once on the UNABIA pod (the images, reference masks and checkpoints live there). It reuses the project's own
evaluation code (``eval_student``/``eval_inputs``/``evaluate_segmentation``), so the predictions are exactly the
ones behind the per_image.csv; before writing, each case's Dice is recomputed and must match the committed CSV.
Output: a small compressed npz plus its provenance JSON (checkpoint sha256, code revision, device).

    python scripts/figures/fig3_extract_cases.py --out-dir resultados/figuras/fuentes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "plan-segmentacion"
for extra in (REPO / "src", PLAN / "comun" / "nucleo", PLAN / "comun" / "evaluacion", REPO / "scripts" / "figures"):
    sys.path.insert(0, str(extra))

import eval_inputs  # noqa: E402
import eval_student  # noqa: E402
from evaluate_segmentation import binarize, compute_binary_metrics  # noqa: E402
from fig3_masks import CASES_JSON, SEGMENTERS, load_dice  # noqa: E402

POD = Path("/home/jovyan/work/proyectonp")
CHECKPOINTS = {"student_r3": POD / "salidas/final/student_r3/best.pt", "A2": POD / "salidas/s4_unet/a2/best.pt"}
INPUTS = {
    "ham10000": (POD / "eval_ham10000_images", POD / "masks_eval/ham10000/HAM10000_segmentations_lesion_tschandl"),
    "isic2018": (POD / "datos_isic2018/ISIC2018_Task1-2_Training_Input", POD / "masks_eval/isic2018/ground_truth"),
}
DICE_TOLERANCE = 1e-4  # the re-inferred Dice of every case must match the committed CSV (lead's requirement)
EVAL_BATCH_SIZE = 32  # the original final.log evaluation command used --batch-size 32
NPZ_NAME, PROVENANCE_NAME = "fig3_casos.npz", "fig3_casos.provenance.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_revision() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def predict_case(model, pairs: list[tuple[str, Path, Path]], index: int, resolution: int, device):
    """Reproduce the evaluator's original inference batch before selecting one case."""
    start = index // EVAL_BATCH_SIZE * EVAL_BATCH_SIZE
    chunk = pairs[start:start + EVAL_BATCH_SIZE]
    images = np.stack([eval_inputs.load_image(path, resolution) for _, path, _ in chunk])
    probabilities = eval_student.predict_masks(model, images, device, len(chunk))
    return images[index - start], binarize(probabilities[index - start])


def extract(cases: list[dict], device) -> tuple[dict, dict]:
    models = {name: eval_student.load_student(path, device) for name, path in CHECKPOINTS.items()}
    committed = load_dice(REPO)
    pairs_by_dataset = {dataset: eval_inputs.pair_images_and_masks(*paths) for dataset, paths in INPUTS.items()}
    indexes = {dataset: {image_id: index for index, (image_id, _, _) in enumerate(pairs)}
               for dataset, pairs in pairs_by_dataset.items()}
    arrays, worst = {}, 0.0
    for case in cases:
        dataset, image_id = case["dataset"], case["image_id"]
        pairs = pairs_by_dataset[dataset]
        index = indexes[dataset][image_id]
        mask_path = pairs[index][2]
        base = f"{dataset}__{image_id}"
        for name, (model, resolution) in models.items():
            image, prediction = predict_case(model, pairs, index, resolution, device)
            reference = eval_inputs.load_binary_mask(mask_path, resolution)
            dice = compute_binary_metrics(prediction, reference)["dice"]
            worst = max(worst, abs(dice - committed[(name, dataset)][image_id]))
            arrays[f"{base}__{name}"] = prediction.astype(np.uint8)
        arrays[f"{base}__image"], arrays[f"{base}__reference"] = image, reference.astype(np.uint8)
    return arrays, {"max_abs_dice_difference_vs_csv": worst, "evaluation_batch_size": EVAL_BATCH_SIZE}


def main(argv: list[str] | None = None) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = json.loads((REPO / CASES_JSON).read_text(encoding="utf-8"))
    arrays, check = extract(cases, device)
    if check["max_abs_dice_difference_vs_csv"] > DICE_TOLERANCE:
        print(f"ABORT: re-inferred Dice differs from the committed CSV by {check['max_abs_dice_difference_vs_csv']:.2e}",
              file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in (NPZ_NAME, PROVENANCE_NAME):
        if (args.out_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite {args.out_dir / name}")
    np.savez_compressed(args.out_dir / NPZ_NAME, **arrays)
    provenance = {"cases": cases, "device": str(device), "code_revision": code_revision(), "threshold": 0.5,
                  "extractor_sha256": file_sha256(Path(__file__)),
                  "checkpoints": {name: {"path": str(p), "sha256": file_sha256(p)} for name, p in CHECKPOINTS.items()},
                  "segmenters": list(SEGMENTERS), **check}
    (args.out_dir / PROVENANCE_NAME).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"OK {args.out_dir / NPZ_NAME} max|dDice|={check['max_abs_dice_difference_vs_csv']:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
