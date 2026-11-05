import json, sys
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from evaluate_overlay_readout import load_labels
arm_dir = sys.argv[1]
def run(kind, base_path, new_path, meta, sha):
    base=json.load(open(base_path)); new=json.load(open(new_path))
    assert base["image_ids"]==new["image_ids"]
    cls=base["classes"]; C=np.array(cls); g=np.asarray(base["groups"])
    y=load_labels(kind,[{"image_id":i} for i in base["image_ids"]],Path(meta),sha)
    mf=lambda yt,p: f1_score(yt,C[np.argmax(p,1)],labels=cls,average="macro",zero_division=0)
    ug=np.unique(g); idx={k:np.where(g==k)[0] for k in ug}
    rng=np.random.default_rng(0); S=[np.concatenate([idx[k] for k in rng.choice(ug,len(ug))]) for _ in range(500)]
    nm=[mf(y,np.array(r["oof_probabilities"])) for r in new["runs"]]; bm=[mf(y,np.array(r["oof_probabilities"])) for r in base["runs"]]
    pn=np.array(new["runs"][0]["oof_probabilities"]); pb=np.array(base["runs"][0]["oof_probabilities"])
    ds=[mf(y[s],pn[s])-mf(y[s],pb[s]) for s in S]
    return f"mean {np.mean(nm):.4f} d_mean {np.mean(nm)-np.mean(bm):+.4f} d_seed0 {nm[0]-bm[0]:+.4f} IC95 {np.round(np.percentile(ds,[2.5,97.5]),4).tolist()}"
B="results/baseline-current-recipe-seed42-evaluation-v1/"
print("PAD", run("PAD", B+"balanced-linear-reader-pad-adapt.json", f"{arm_dir}/PAD-readout.json", "/home/jovyan/work/pad_ufes_20/metadata.csv", "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"))
print("HAM", run("HAM", B+"balanced-linear-reader-ham-dev.json", f"{arm_dir}/HAM-readout-06aee5.json", "/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv", "06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5"))
