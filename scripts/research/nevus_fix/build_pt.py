import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/jovyan/work/derm-v10.2-evidence/models/classifier/code")
from dermssl.release import moco_d4
src = Path("/home/jovyan/work/release_v10_1/build-d4-candidate/classifier.pt")
assert hashlib.sha256(src.read_bytes()).hexdigest() == "6c3a8dfb2d5c8cabeb86554925e26613c45edaee57ada07b3b33839008489309"
model, meta = moco_d4.load_checkpoint(src)
head = np.load("/home/jovyan/work/nevus-fix-20261113/results/head-cell23.npz", allow_pickle=False)
assert head["classes"].tolist() == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
model.ham_weight.copy_(torch.from_numpy(head["weight"]).float())
model.ham_bias.copy_(torch.from_numpy(head["bias"]).float())
prov = dict(meta["provenance"])
prov.update({"release": "v10.4.0", "ham_head": "multi-domain reader cell:2:3 (HAM, ISIC2019 non-HAM, PAD adapt, MILK10k train split)",
             "ham_head_npz_sha256": hashlib.sha256(Path("/home/jovyan/work/nevus-fix-20261113/results/head-cell23.npz").read_bytes()).hexdigest(),
             "parent_checkpoint_sha256": "6c3a8dfb2d5c8cabeb86554925e26613c45edaee57ada07b3b33839008489309"})
out = Path("/home/jovyan/work/nevus-fix-20261113/release/classifier.pt")
out.parent.mkdir(parents=True, exist_ok=True)
moco_d4.save_checkpoint(out, model, meta["config"], prov)
m2, _ = moco_d4.load_checkpoint(out)
x = torch.randn(1, 3, 448, 448)
with torch.inference_mode():
    a = m2(x)[0]; f = m2.features(x)
    z = f @ torch.from_numpy(head["weight"]).T + torch.from_numpy(head["bias"])
print(json.dumps({"sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "diff": float((a - torch.softmax(z, -1)).abs().max())}))
