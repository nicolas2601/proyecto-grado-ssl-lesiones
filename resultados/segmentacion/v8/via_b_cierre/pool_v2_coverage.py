"""Would porting v2 into the paired contrast change which images get treated?

The paired control/treated design requires the two arms to differ in exactly
one factor. v1 (what the materialized mirror used) abstains only when its
own preconditions fail; v2 adds abstentions v1 does not have
(no_reachable_ita_range, border_population_shift_too_large, and the unified
skin definition that excludes vignetting). If v2 treats a materially smaller
share of the pool, swapping it in changes coverage as well as magnitude --
a second factor through the back door.

Runs both modules over the same sampled records of the real ISIC pool the
mirror was built from, with the same hash-derived delta for v1 and v2's own
solver for v2, and reports where each one lands.
"""
import json
import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/jovyan/work/ita-ladder-20261104/src_fixed")
sys.path.insert(1, "/home/jovyan/work/proyectonp/repo_v8_canonical/src")
from dermssl import tone_augment_lab as v1  # noqa: E402
from dermssl import tone_augment_lab_v2 as v2  # noqa: E402
from dermssl.fitzpatrick import estimate_fst  # noqa: E402

assert "src_fixed" in v2.__file__, "resolved the pre-fix v2"
print(f"v1: {v1.__file__}\nv2: {v2.__file__}", flush=True)

RECORDS = "/home/jovyan/work/seg_masks_026_pool_20261103/full/masks.source_fit.jsonl"
MASK_ROOT = "/home/jovyan/work/seg_masks_026_pool_20261103"
OUT = "/home/jovyan/work/ita-ladder-20261104/pool_v2_coverage.jsonl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
STEP = 0

rows = []
with open(RECORDS) as fh:
    for line in fh:
        r = json.loads(line)
        if r.get("eligible") and r.get("mask_semantics") == "true=lesion":
            rows.append(r)
print(f"registros elegibles: {len(rows)}", flush=True)
rng = np.random.default_rng(42)
sample = [rows[i] for i in rng.choice(len(rows), min(N, len(rows)), replace=False)]
print(f"muestra: {len(sample)}", flush=True)

c1, c2 = Counter(), Counter()
with open(OUT, "w") as fh:
    for i, rec in enumerate(sample, 1):
        try:
            img = np.asarray(
                Image.open(rec["source_image_path"]).convert("RGB"), dtype=np.float32
            ) / 255.0
            mask = np.asarray(
                Image.open(f"{MASK_ROOT}/{rec['mask_path']}").convert("L")
            ) > 127
        except Exception as exc:  # noqa: BLE001
            c1["error_de_carga"] += 1
            c2["error_de_carga"] += 1
            if i < 4:
                print(f"  carga falló: {exc}", flush=True)
            continue
        if img.shape[:2] != mask.shape:
            img = np.asarray(
                Image.open(rec["source_image_path"]).convert("RGB").resize(
                    (mask.shape[1], mask.shape[0]), Image.BILINEAR
                ), dtype=np.float32
            ) / 255.0

        before = estimate_fst(img, mask=mask)
        sha = rec["mask_sha256"]  # per-image; source_revision is a dataset-wide constant

        # v1: hash-derived delta, symmetric cap, no reachable range.
        delta = v1.sample_ita_delta(42, STEP, sha)
        try:
            _, r1 = v1.augment_skin_tone_from_mask_record(
                img, mask, rec, ita_delta=delta, feather_sigma=0.0
            )
            c1["aplicada" if r1.get("applied") else r1.get("fallback_reason", "no_aplicada")] += 1
        except Exception as exc:  # noqa: BLE001
            r1 = {"applied": False, "fallback_reason": f"excepcion:{type(exc).__name__}"}
            c1[r1["fallback_reason"]] += 1

        # v2: its own solver over the per-image reachable range.
        _, r2 = v2.augment_skin_tone_v2(
            img, mask, 42, STEP, sha, policy_cap_degrees=None, feather_sigma=0.0
        )
        c2["aplicada" if r2.get("applied") else r2.get("fallback_reason", "no_aplicada")] += 1

        fh.write(json.dumps({
            "image_id": rec["image_id"],
            "ita_before": None if np.isnan(before.ita) else float(before.ita),
            "status_before": before.status,
            "b_before": float(before.b_median),
            "area_fraction": rec.get("area_fraction"),
            "v1_applied": bool(r1.get("applied")),
            "v1_reason": r1.get("fallback_reason"),
            "v1_delta": float(delta),
            "v2_applied": bool(r2.get("applied")),
            "v2_reason": r2.get("fallback_reason"),
            "v2_delta": r2.get("ita_delta_raw"),
        }) + "\n")
        if i % 250 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

n = len(sample)
for name, c in (("v1 (modulo congelado, en produccion)", c1), ("v2 (instrumento de esta sesion)", c2)):
    print(f"\n{name}")
    for k, v in c.most_common():
        print(f"  {k:38s} {v:5d}  ({v / n:5.1%})")
print(f"\nescrito: {OUT}")
