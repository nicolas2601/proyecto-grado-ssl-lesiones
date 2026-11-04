"""ITA per image over v7's frozen validation split, for stratified bias readout.

This measures only. It never calls augment_skin_tone and never writes an image.

Row order is the split's own val_ids order, verified two independent ways
before this script was written: c1_split_identity.ids_sha256 over the rebuilt
val_ids reproduces final_eval.json's eval_ids_sha256 exactly, and the split's
id2label over those ids reproduces final_eval.probas.npz's y_true elementwise
on all 2004 rows.

Teacher-026 masks exist for the subset of val ids that are also in the SSL
pool (HAM10000 ids largely appear inside ISIC 2019 -- docs/02-datasets.md).
Where a mask exists it is used; where it does not, estimate_fst falls back to
Otsu, and the row says which, because docs/63 measures the masked estimator as
2x-6x better by stratum and mixing the two silently would hide that.
"""
import json
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/jovyan/work/ita-ladder-20261104/src_fixed")
sys.path.insert(1, "/home/jovyan/work/proyectonp/repo_v8_canonical/src")
from dermssl.fitzpatrick import estimate_fst  # noqa: E402

MASK_ROOT = "/home/jovyan/work/seg_masks_026_pool_20261103"
V7 = "/home/jovyan/work/ita-ladder-20261104/v7"

masks = {}
with open(f"{MASK_ROOT}/full/masks.source_fit.jsonl") as fh:
    for line in fh:
        r = json.loads(line)
        if r.get("eligible") and r.get("mask_semantics") == "true=lesion":
            masks[r["image_id"]] = r["mask_path"]

rows = [json.loads(line) for line in open(f"{V7}/val_manifest.jsonl")]
print(f"filas: {len(rows)} | con mascara 026: {sum(1 for r in rows if r['image_id'] in masks)}", flush=True)

out = open(f"{V7}/val_ita.jsonl", "w")
for i, r in enumerate(rows, 1):
    img = np.asarray(Image.open(r["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
    mp = masks.get(r["image_id"])
    mask = None
    if mp is not None:
        m = np.asarray(Image.open(f"{MASK_ROOT}/{mp}").convert("L")) > 127
        if m.shape != img.shape[:2]:
            m = np.asarray(
                Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (img.shape[1], img.shape[0]), Image.NEAREST
                )
            ) > 127
        mask = m
    est = estimate_fst(img, mask=mask)
    out.write(json.dumps({
        "row": r["row"],
        "image_id": r["image_id"],
        "label": r["label"],
        "class": r["class"],
        "mask_source": "teacher_026" if mask is not None else "otsu_fallback",
        "ita": None if np.isnan(est.ita) else float(est.ita),
        "ita_status": est.status,
        "b_median": float(est.b_median),
        "skin_fraction": float(est.skin_fraction),
    }) + "\n")
    if i % 400 == 0:
        print(f"  {i}/{len(rows)}", flush=True)
out.close()
print(f"escrito: {V7}/val_ita.jsonl")
