"""Via B re-measured on real S-SYNTH renders, after the aggregation fix.

Identical protocol to via_b2.py (same index, same seed 42, same stratification
by ground-truth melanin, feather_sigma=0.0, no policy cap) with one change:
``src_fixed`` precedes the canonical checkout on sys.path, so
``tone_augment_lab_v2`` is the post-6997206 module whose curve aggregates ITA as
the angle of the medians. Everything else is held fixed so the only moving part
is the fix itself.

Closure metric: |ita_after - record["ita_objetivo"]|, where ita_after is
measured by ``fitzpatrick.estimate_fst`` on the produced image.
"""
import hashlib
import json
import re
import sys
from collections import Counter

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/jovyan/work/ita-ladder-20261104/src_fixed")
sys.path.insert(1, "/home/jovyan/work/proyectonp/repo_v8_canonical/src")
from dermssl.fitzpatrick import estimate_fst  # noqa: E402
from dermssl import tone_augment_lab_v2 as tav2  # noqa: E402

print(f"modulo v2: {tav2.__file__}", flush=True)
assert "src_fixed" in tav2.__file__, "resolved the pre-fix module"

INDEX = "/home/jovyan/work/ita-ladder-20261104/ssynth_images.txt"
OUT = "/home/jovyan/work/ita-ladder-20261104/via_b3_rows.jsonl"
MEL_RE = re.compile(r"/mel_([0-9.]+)/")
PER_MEL = int(sys.argv[1]) if len(sys.argv) > 1 else 4

paths = [line.strip() for line in open(INDEX) if line.strip()]
by = {}
for p in paths:
    m = MEL_RE.search(p)
    if m:
        by.setdefault(m.group(1), []).append(p)
rng = np.random.default_rng(42)
sample = []
for mel in sorted(by, key=float):
    g = sorted(by[mel])
    sample += [g[i] for i in rng.choice(len(g), min(PER_MEL, len(g)), replace=False)]
print(f"grupos de melanina: {len(by)} | muestras: {len(sample)}", flush=True)

reasons = Counter()
with open(OUT, "w") as fh:
    for i, p in enumerate(sample, 1):
        img = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        gt = np.asarray(Image.open(p.replace("image.png", "mask.png")).convert("L")) > 127
        before = estimate_fst(img, mask=gt)
        if before.status != "ok":
            reasons["ita_no_medible_antes"] += 1
            continue
        sha = hashlib.sha256(p.encode()).hexdigest()
        out, rec = tav2.augment_skin_tone_v2(
            img, gt, 42, 0, sha, policy_cap_degrees=None, feather_sigma=0.0
        )
        if not rec.get("applied"):
            reasons[rec.get("fallback_reason", "sin_motivo")] += 1
            continue
        reasons["aplicada"] += 1
        after = estimate_fst(np.asarray(out, dtype=np.float32), mask=gt)
        fh.write(json.dumps({
            "mel": float(MEL_RE.search(p).group(1)),
            "ita_before": float(before.ita),
            "ita_after": None if np.isnan(after.ita) else float(after.ita),
            "status_after": after.status,
            "b_before": float(before.b_median), "b_after": float(after.b_median),
            "record": {k: v for k, v in rec.items()
                       if isinstance(v, (int, float, str, bool, type(None)))},
        }) + "\n")
        if i % 50 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

print("\nDESTINO DE CADA IMAGEN:")
for k, v in reasons.most_common():
    print(f"  {k:36s} {v:4d}  ({v / len(sample):5.1%})")
print(f"escrito: {OUT}")
