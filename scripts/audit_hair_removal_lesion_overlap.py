"""C3(a)/(b): does the hair remover eat the lesion border, on PH2?

Measures, per PH2 image at the production resolution (224x224, bilinear - the
same resize `pipeline_segment_circle.segment_image` and
`segmentation_methods.common.data` apply before calling
`segmentation_methods.common.hair.remove_hair`), what fraction of each hair
mask's pixels fall INSIDE the expert lesion mask:

* ``shaped=False`` (``hair_mask``, what every production caller actually
  runs: ``pipeline_segment_circle.py``, ``common/data.py``,
  ``pilot_adapters.py``, and the teacher 026 recipe `generate_masks_026_pool.py`
  on UNABIA all leave ``shaped=False`` at its default).
* ``shaped=True`` (``elongated_hair_mask``, the elongation filter that exists
  for exactly this failure mode and that no production caller enables).

No model inference, no GPU: this is a pure OpenCV/NumPy measurement against
PH2's 200 expert dermatologist masks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from dermssl.segmentation_methods.common.hair import elongated_hair_mask, hair_mask

REPO = Path(__file__).resolve().parents[1]
PH2 = REPO / "data" / "ph2"
SIZE = 224  # matches pipeline_segment_circle.segment_image's resize before hair removal


def _load_resized(path: Path, resample: Image.Resampling) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.resize((SIZE, SIZE), resample=resample))


def _lesion_mask_224(mask_path: Path) -> np.ndarray:
    arr = _load_resized(mask_path, Image.Resampling.NEAREST)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 127


def _overlap_fraction(hair: np.ndarray, lesion: np.ndarray) -> float | None:
    hair_bool = hair.astype(bool)
    total = int(hair_bool.sum())
    if total == 0:
        return None
    return float((hair_bool & lesion).sum()) / total


def main() -> None:
    manifest = json.loads((PH2 / "manifest.json").read_text(encoding="utf-8"))
    rows = []

    for record in manifest["records"]:
        img_path = PH2 / record["image_path"]
        mask_path = PH2 / record["mask_path"]
        rgb = _load_resized(img_path, Image.Resampling.BILINEAR)
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        rgb = rgb[..., :3].astype(np.uint8)
        lesion = _lesion_mask_224(mask_path)

        unshaped = hair_mask(rgb)
        shaped = elongated_hair_mask(rgb)

        rows.append(
            {
                "image_id": record["image_id"],
                "hair_fraction_unshaped": float(unshaped.astype(bool).mean()),
                "hair_fraction_shaped": float(shaped.astype(bool).mean()),
                "overlap_with_lesion_unshaped": _overlap_fraction(unshaped, lesion),
                "overlap_with_lesion_shaped": _overlap_fraction(shaped, lesion),
                "lesion_fraction": float(lesion.mean()),
            }
        )

    def _stats(key: str) -> dict:
        values = [r[key] for r in rows if r[key] is not None]
        n_detected = len(values)
        n_no_hair = sum(1 for r in rows if r[key] is None)
        arr = np.asarray(values, dtype=np.float64)
        return {
            "n_images_with_hair_detected": n_detected,
            "n_images_no_hair_detected": n_no_hair,
            "mean": float(arr.mean()) if n_detected else None,
            "median": float(np.median(arr)) if n_detected else None,
            "p95": float(np.quantile(arr, 0.95)) if n_detected else None,
        }

    summary = {
        "dataset": "ph2",
        "n_images": len(rows),
        "resize": f"{SIZE}x{SIZE} bilinear (image), nearest (mask) - matches production",
        "overlap_with_lesion_unshaped": _stats("overlap_with_lesion_unshaped"),
        "overlap_with_lesion_shaped": _stats("overlap_with_lesion_shaped"),
        "hair_fraction_unshaped": _stats("hair_fraction_unshaped"),
        "hair_fraction_shaped": _stats("hair_fraction_shaped"),
    }

    out_dir = REPO / "resultados" / "segmentacion" / "v8" / "hair_removal_lesion_overlap_ph2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_image.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
