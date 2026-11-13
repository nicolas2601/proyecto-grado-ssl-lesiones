import json, sys
r = json.load(open(sys.argv[1]))
specs = sys.argv[2].split(",")
sets = ["milk_dev", "milk_test", "pad_test", "ham_holdout", "ddi", "fitz", "bosque", "internet"]
def row(name, v):
    f = lambda x: "  -  " if x is None else "%.2f" % x
    return "%-22s bacc %s f1 %s nonNV>NV %s melRec %s mel>nv %s malAUC %s" % (
        name, f(v["balanced_accuracy"]), f(v["macro_f1"]), f(v["non_nv_predicted_nv"]),
        f(v["recall"].get("mel")), f(v["mel_to_nv"]), f(v["malignant_auc"]))
for s in sets:
    if s not in r["baselines"]["shipped_HAM7"]:
        continue
    for part in ("all", "clin", "derm"):
        if part not in r["baselines"]["shipped_HAM7"][s]:
            continue
        print(row(f"{s}/{part} SHIP", r["baselines"]["shipped_HAM7"][s][part]))
        for sp in specs:
            print(row(f"{s}/{part} {sp}", r["readers"][sp][s][part]))
