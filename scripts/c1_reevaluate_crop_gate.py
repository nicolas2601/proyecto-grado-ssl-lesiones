"""Re-evaluate the step-0 crop gate under both crop offsets, side by side.

Reads the finished mask artifact; nothing is re-inferred, so the mask sha is
untouched. Both offsets are measured in the SAME pass over the SAME resized
mask, so the two columns differ only by the crop.

The verdict comes from the real gate (``evaluate_crop_gates``), not from
thresholds retyped here -- a second copy of a threshold is a second thing to
get out of sync.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/jovyan/work/proyectonp/repo_final")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from dermssl import c1_extraction as ex  # noqa: E402
from dermssl import c1_mask_export as mx  # noqa: E402
from dermssl import run_environment  # noqa: E402
from c1_predict_masks import evaluate_crop_gates  # noqa: E402

MASKS_DIR = Path("/home/jovyan/work/proyectonp/salidas/c1/masks")
MODES = ("floor", "round")


def crop(array, size, mode):
    height, width = array.shape[:2]
    if mode == "floor":
        top, left = max(0, (height - size) // 2), max(0, (width - size) // 2)
    else:
        top = max(0, int(round((height - size) / 2.0)))
        left = max(0, int(round((width - size) / 2.0)))
    return array[top : top + size, left : left + size]


def main():
    candidates = sorted(MASKS_DIR.glob("*.npy"))
    if len(candidates) != 1:
        raise SystemExit(f"esperaba un solo .npy, encontre {[p.name for p in candidates]}")
    stored = np.load(candidates[0], mmap_mode="r")
    n_images = int(stored.shape[0])
    print(f"artefacto: {candidates[0].name}  shape={stored.shape}", flush=True)

    counts = {m: {"emptied": 0, "below": 0} for m in MODES}
    n_empty_from_segmenter = 0
    n_lost_in_resize = 0
    n_with_lesion = 0
    n_differ = 0

    start = time.time()
    for index in range(n_images):
        mask = np.asarray(stored[index]) == mx.LESION_LABEL
        if not mask.any():
            n_empty_from_segmenter += 1
            continue
        n_with_lesion += 1

        resized = ex._resize_shorter_side(mask, ex._INTERMEDIATE)
        before = int(resized.sum())
        if before == 0:
            # Survived the segmenter, died in the NEAREST resize. Counted, not
            # skipped: a silent skip is the same no-op this gate exists to catch.
            n_lost_in_resize += 1
            continue

        retained = {}
        for mode in MODES:
            value = float(crop(resized, ex.RESOLUTION, mode).sum()) / before
            retained[mode] = value
            if value == 0.0:
                counts[mode]["emptied"] += 1
            if value < 0.5:
                counts[mode]["below"] += 1
        if retained["floor"] != retained["round"]:
            n_differ += 1

        if index % 2000 == 0:
            print(f"  {index}/{n_images}  {time.time() - start:.0f}s", flush=True)

    report = {
        "n_images": n_images,
        "n_empty_from_segmenter": n_empty_from_segmenter,
        "n_lost_in_resize": n_lost_in_resize,
        "n_with_lesion": n_with_lesion,
        "n_images_where_the_two_crops_differ": n_differ,
        "by_crop": {},
    }
    for mode in MODES:
        summary = {
            "n_images": n_images,
            "n_with_lesion": n_with_lesion,
            "n_empty_from_segmenter": n_empty_from_segmenter,
            "n_emptied_by_crop": counts[mode]["emptied"],
            "n_below_half": counts[mode]["below"],
            "fraction_empty_from_segmenter": n_empty_from_segmenter / n_images,
            "fraction_emptied_by_crop": counts[mode]["emptied"] / n_images,
            "fraction_below_half": (
                counts[mode]["below"] / n_with_lesion if n_with_lesion else None
            ),
            "retained": [],
        }
        passes, reasons = evaluate_crop_gates(summary)
        report["by_crop"][mode] = {
            "n_emptied_by_crop": summary["n_emptied_by_crop"],
            "n_below_half": summary["n_below_half"],
            "fraction_emptied_by_crop": summary["fraction_emptied_by_crop"],
            "fraction_below_half": summary["fraction_below_half"],
            "gate_passes": passes,
            "gate_reasons": reasons,
        }

    report["seconds"] = round(time.time() - start, 1)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    out = Path("/home/jovyan/work/proyectonp/salidas/c1/crop_regate_report.json")
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    run_environment.write_sidecar(out)


if __name__ == "__main__":
    main()
