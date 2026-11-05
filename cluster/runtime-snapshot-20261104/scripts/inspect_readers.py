import json, glob, os
for f in sorted(glob.glob("results/public-frozen-encoder-comparison-v1/*reader*.json")) + ["results/ssl-night-comparison-20261026-v1/final-night-report.json"]:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f, "ERR", e); continue
    if "runs" in d:
        m = [r["metrics"]["macro_f1"] for r in d["runs"]] if "metrics" in d["runs"][0] else None
        print(os.path.basename(f), d.get("kind", d.get("reader")), "mean", None if not m else round(sum(m)/len(m), 4), "n", len(d["runs"]))
    else:
        print(os.path.basename(f), list(d.keys())[:6])
n = json.load(open("results/ssl-night-comparison-20261026-v1/final-night-report.json"))
print(json.dumps(n["separate_public_encoder_comparators"], ensure_ascii=False)[:1400])
