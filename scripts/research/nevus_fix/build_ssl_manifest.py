"""SSL manifest: shipped mixed-fit rows + MILK clinical images of train/dev lesions in the clinical bucket.

No label field is written. MILK test lesions and every external test set are excluded.
"""
import csv, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "/home/jovyan/work/nevus-fix-20261113/scripts")
from nevus_study import split_of

W = Path("/home/jovyan/work")
base = [json.loads(l) for l in (W / "clf1_clinical_reference_20261024/config/ssl-overlay-v1/mixed-fit.jsonl").read_text().splitlines() if l.strip()]
milk = [r for r in csv.DictReader((W / "milk10k/manifest-clean.csv").open()) if r["modality"] == "clinica"
        and split_of(r["lesion_id"]) in ("milk_train", "milk_dev")]
extra = [{"identity_component_id": "milk:" + r["lesion_id"], "image_id": r["image_id"], "image_path": r["image_path"],
          "role": "pad_adapt"} for r in milk]
rows = base + extra
out = W / "nevus-fix-20261113/ssl/mixed-fit-clin.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(base), len(extra), hashlib.sha256(out.read_bytes()).hexdigest())
