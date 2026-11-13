import json, sys, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper
from PIL import Image, ImageOps
OLD = "/home/nicolas/Documentos/release-v10.2/assets-v10.2.0/v10.2.0-classifier.onnx"
a = {i.name: numpy_helper.to_array(i) for i in onnx.load(OLD).graph.initializer}
b = {i.name: numpy_helper.to_array(i) for i in onnx.load("model.onnx").graph.initializer}
changed = sorted(k for k in a if not np.array_equal(a[k], b[k]))
assert set(a) == set(b) and changed == ["ham_bias", "onnx::MatMul_11673"], changed
MEAN = np.asarray((0.485, 0.456, 0.406), np.float32); STD = np.asarray((0.229, 0.224, 0.225), np.float32)
def prep(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
    x = (np.asarray(im, np.float32) / np.float32(255.0) - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], np.float32)
feats = np.load("internet.npz"); head = np.load("head-cell23.npz")
ids = feats["image_ids"].tolist()
import csv
rows = list(csv.DictReader(open("../internet/labels.csv")))
so = ort.SessionOptions(); so.intra_op_num_threads = 8
new, old = ort.InferenceSession("model.onnx", so), ort.InferenceSession(OLD, so)
dn, do, agree = [], [], 0
for r in rows[: int(sys.argv[1])]:
    i = ids.index(r["file"].split("/")[-1].rsplit(".", 1)[0])
    x = prep("../internet/" + r["file"])
    pn = new.run(None, {"pixel_values": x})[0][0]; po = old.run(None, {"pixel_values": x})[0][0]
    z = feats["features"][i] @ head["weight"].T + head["bias"]; e = np.exp(z - z.max()); exp_new = e / e.sum()
    dn.append(np.abs(pn - exp_new).max()); do.append(np.abs(po - feats["ham7"][i]).max())
    agree += int(pn.argmax() == exp_new.argmax())
print(json.dumps({"changed_initializers": changed, "n": len(dn), "max_diff_new_vs_offline": float(max(dn)),
                  "max_diff_old_vs_offline_control": float(max(do)), "argmax_agree": agree}))
