"""Does the extraction batch size (16 in E3 vs 64 in v7.1) explain B71 0.7106 vs 0.7141? Re-extracts the val
`full` condition at batch 64 and scores it with the same probe (train features from E3). Diagnostic only."""
import sys
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from sklearn.metrics import f1_score  # noqa: E402

import e3_io  # noqa: E402
import e3_run  # noqa: E402
from dermssl import probe_threads  # noqa: E402
from dermssl import v71_grid as vg  # noqa: E402

P = Path.home() / "work/proyectonp"
C1 = P / "salidas/c1"
bs = int(sys.argv[1])
argv = ["--ham-root", str(Path.home() / "work/datos/ham10000"), "--image-dir", str(P / "eval_ham10000_images"),
        "--split-index", str(C1 / "c1_split_index.json"), "--holdout", str(C1 / "selection/selection_holdout.json"),
        "--weights-dir", str(P / "salidas/v7"), "--masks", str(C1 / "masks/c1_pseudo_masks.npy"),
        "--masks-manifest", str(C1 / "masks/c1_pseudo_masks_manifest.json"),
        "--tone-manifest", str(P / "salidas/final/fst/fst_pseudo_manifest.csv"),
        "--out-dir", str(P / "salidas/e3_diag"), "--stage", "final", "--device", "cuda", "--batch-size", str(bs)]
rt = e3_run.parse_args(argv)
rt = e3_io.build_runtime(rt)
ids = rt.ids["val"]
images = np.stack([rt.resize_crop(rt.load_item(i)[0], 224) for i in ids])
xv_new = rt.extract_fn(images, 8)
F = P / "salidas/e3/features"
ids_t = rt.ids["train"]
img_t = np.stack([rt.resize_crop(rt.load_item(i)[0], 224) for i in ids_t])
xt_new = rt.extract_fn(img_t, 1)
xt, xv_old = np.load(F / "train__full.npy"), np.load(F / "val__full.npy")
print("train re-extracted at batch", bs, "max|dF| vs E3 train", float(np.abs(xt_new - xt).max()))
print("batch", bs, "| max|dF| vs E3 val feats", float(np.abs(xv_new - xv_old).max()), "mean", float(np.abs(xv_new - xv_old).mean()))
y_tr, y_val = rt.labels_of("train"), rt.labels_of("val")
z = np.load(REPO / "resultados/clasificacion/v71/v71_final_eval.probas.npz", allow_pickle=True)
p71 = z["B71"]
p = probe_threads.call_pinned(vg.fit_predict_views, xt_new[0], y_tr, xv_new, rt.n_classes, "balanced")
p = p.mean(0) if p.ndim == 3 else p
print("macro-F1 with val feats re-extracted at batch", bs, ":", round(f1_score(y_val, p.argmax(1), average="macro"), 4),
      "| v71:", round(f1_score(y_val, p71.argmax(1), average="macro"), 4),
      "| argmax agreement with v71:", float((p.argmax(1) == p71.argmax(1)).mean()))
