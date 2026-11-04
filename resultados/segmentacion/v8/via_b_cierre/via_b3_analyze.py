"""Closure figure for via B after the aggregation fix, on real S-SYNTH.

Compares the post-fix run (via_b3_rows.jsonl) against the pre-fix run
(via_b2_nofeather.jsonl) on the metric that defines the residual:
|ita_after - ita_objetivo|, with ita_after measured by estimate_fst on the
produced image. The two runs share seed, index, stratification, feather_sigma
and cap, so the module is the only difference; the pre-fix run used a smaller
per-melanin draw, so its sample is a subset in intent, not identity.
"""
import json

import numpy as np


def load(path):
    rows = [json.loads(line) for line in open(path)]
    keep = [r for r in rows
            if r["ita_after"] is not None and r["record"].get("ita_objetivo") is not None]
    signed = np.array([r["ita_after"] - r["record"]["ita_objetivo"] for r in keep])
    return rows, keep, signed


def report(label, path):
    rows, keep, s = load(path)
    e = np.abs(s)
    print(f"\n{label}")
    print(f"  filas escritas        {len(rows)}")
    print(f"  con ITA medible       {len(keep)}")
    print(f"  |error| mediana       {np.median(e):8.4f}")
    print(f"  |error| p95           {np.percentile(e, 95):8.4f}")
    print(f"  |error| maximo        {e.max():8.4f}")
    print(f"  firmado min / max     {s.min():+8.4f} / {s.max():+8.4f}")
    print(f"  sesgo (positivos)     {(s > 0).sum()}/{len(s)}  ({(s > 0).mean():.1%})")
    return keep, s


pre_keep, pre_s = report("ANTES DEL FIX  (via_b2_nofeather.jsonl)", "via_b2_nofeather.jsonl")
post_keep, post_s = report("DESPUES DEL FIX (via_b3_rows.jsonl)", "via_b3_rows.jsonl")

print("\nPOR TERCIL DE MELANINA VERDADERA (post-fix)")
mel = np.array([r["mel"] for r in post_keep])
e = np.abs(post_s)
q = np.quantile(mel, [1 / 3, 2 / 3])
for name, sel in [
    ("bajo ", mel <= q[0]),
    ("medio", (mel > q[0]) & (mel <= q[1])),
    ("alto ", mel > q[1]),
]:
    if sel.sum():
        print(f"  mel {name}  n={sel.sum():4d}  mediana {np.median(e[sel]):7.4f}"
              f"  p95 {np.percentile(e[sel], 95):7.4f}  max {e[sel].max():7.4f}")

print("\nABSTENCIONES (post-fix, desde el log)")
