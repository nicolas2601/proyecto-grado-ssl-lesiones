import json, sys, hashlib
from pathlib import Path
import numpy as np, torch
from PIL import Image
B = Path("/home/jovyan/work/derm-v10.2-evidence/models/classifier")
sys.path.insert(0, str(B / "code")); sys.path.insert(0, str(B))
from dermssl.release import moco_d4
from preprocess import preprocess_image
torch.set_num_threads(8)
model, _ = moco_d4.load_checkpoint("/home/jovyan/work/nevus-fix-20261113/release/classifier.pt")
old = json.loads((B / "parity-fixtures.json").read_text())
new, pad_drift = [], 0.0
for item in old:
    with Image.open(B / item["image"]) as im:
        x = torch.from_numpy(preprocess_image(im, 448))
    with torch.inference_mode():
        ham, pad = (t[0].double().numpy() for t in model(x))
    pad_drift = max(pad_drift, float(np.abs(pad - np.asarray(item["expected"]["PAD"])).max()))
    new.append({**item, "expected": {"HAM": ham.tolist(), "PAD": item["expected"]["PAD"]}})
out = Path("/home/jovyan/work/nevus-fix-20261113/release/parity-fixtures.json")
out.write_text(json.dumps(new))
print(json.dumps({"pad_drift_vs_shipped": pad_drift, "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "n": len(new)}))
