import json
n = json.load(open("results/ssl-night-comparison-20261026-v1/final-night-report.json"))
for r in n["separate_public_encoder_comparators"]:
    io = r["image_only"]; im = r.get("image_plus_metadata", {})
    print(r["representation"], "|", r["scope"], "| img", round(io["macro_f1"],4), "| img+meta", round(im.get("macro_f1", -1),4), "| dIMG", json.dumps(r.get("image_delta_vs_adapted_SSL", {}).get("point")), "| dMETA", json.dumps(r.get("metadata_delta_vs_adapted_SSL_metadata", {}).get("point")))
print("baseline+meta keys", [k for k in n["baseline_PAD_reader_metrics"].keys()][:10])
