#!/usr/bin/env python3
"""Reader variants on frozen CLS4-D4 features: per-class bias calibration, MLP reader, LR+MLP ensemble.

Selection uses milk_dev only (score = clinical bacc + 0.5 * dermoscopic bacc, the v10.4 rule).
Every other set is report-only. The encoder is frozen; labels only enter the reader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

import labels as L
import nevus_study as S
from metrics import evaluate

OFFSET_GRID = np.round(np.arange(-1.5, 1.51, 0.1), 2)
SEED = 20261113


def ham7(u8):
    return np.asarray([L.HAM7.index(L.U8_TO_HAM7[u]) for u in u8])


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def bacc(y, pred):
    present = np.unique(y)
    return float(np.mean([(pred[y == c] == c).mean() for c in present]))


def dev_score(y, logits, mod):
    pred = logits.argmax(1)
    return bacc(y[mod == "clin"], pred[mod == "clin"]) + 0.5 * bacc(y[mod == "derm"], pred[mod == "derm"])


def fit_offsets(y, logits, mod, sweeps=3):
    """Coordinate ascent over additive per-class logit offsets."""
    off = np.zeros(logits.shape[1])
    best = dev_score(y, logits, mod)
    for _ in range(sweeps):
        for k in range(logits.shape[1]):
            for v in OFFSET_GRID:
                trial = off.copy()
                trial[k] = v
                s = dev_score(y, logits + trial, mod)
                if s > best + 1e-9:
                    best, off = s, trial
    return off, best


def nested_offsets(y, logits, mod, groups):
    out = np.empty_like(logits)
    for fit, held in GroupKFold(5).split(logits, y, groups):
        off, _ = fit_offsets(y[fit], logits[fit], mod[fit])
        out[held] = logits[held] + off
    return dev_score(y, out, mod)


class MLP(torch.nn.Module):
    def __init__(self, d, hidden, classes, p):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(d, hidden), torch.nn.GELU(),
                                       torch.nn.Dropout(p), torch.nn.Linear(hidden, classes))

    def forward(self, x):
        return self.net(x)


def train_mlp(train, dev, seed, hidden=1024, p=0.5, epochs=40, lr=3e-4, wd=0.05):
    torch.manual_seed(seed)
    dev_t = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = train["x"].mean(0), train["x"].std(0) + 1e-6
    xt = torch.tensor((train["x"] - mu) / sd, dtype=torch.float32, device=dev_t)
    yt = torch.tensor(ham7(train["y"]), device=dev_t)
    w = torch.tensor(S.weights(np.asarray([L.U8_TO_HAM7[u] for u in train["y"]]), train["mod"], "cell", 2.0),
                     dtype=torch.float32, device=dev_t)
    xd = torch.tensor((dev["x"] - mu) / sd, dtype=torch.float32, device=dev_t)
    yd, md = ham7(dev["y"]), dev["mod"]
    model = MLP(xt.shape[1], hidden, len(L.HAM7), p).to(dev_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best = (-1.0, None, -1)
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(xt), generator=gen).split(512):
            idx = idx.to(dev_t)
            loss = (torch.nn.functional.cross_entropy(model(xt[idx]), yt[idx], reduction="none") * w[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            s = dev_score(yd, model(xd).cpu().numpy(), md)
        if s > best[0]:
            best = (s, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    model.eval()

    def logits(x):
        with torch.no_grad():
            return model(torch.tensor((x - mu) / sd, dtype=torch.float32, device=dev_t)).cpu().numpy()
    return logits, best[0], best[2]


def summarize(sets, logit_fn):
    out = {}
    for name, d in sets.items():
        r = evaluate(d["y"], softmax(logit_fn(d["x"])))
        out[name] = {k: r[k] for k in ("n", "balanced_accuracy", "recall", "non_nv_predicted_nv", "malignant_auc")}
        for m in ("clin", "derm"):
            mask = d["mod"] == m
            if 20 <= mask.sum() < len(mask):
                out[name][m] = round(evaluate(d["y"][mask], softmax(logit_fn(d["x"][mask])))["balanced_accuracy"], 4)
    return out


def brief(tag, res):
    row = {n: round(r["balanced_accuracy"], 3) for n, r in res.items()}
    nv = {n: round(r["recall"].get("nv", float("nan")), 2) for n, r in res.items()}
    print(json.dumps({"variant": tag, "bacc": row, "nv_recall": nv}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feats", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True, help="saved LR head (weight, bias)")
    parser.add_argument("--extra-train", default="", help="comma-separated extra training sources")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--save-mlp", type=Path)
    args = parser.parse_args()
    names = list(S.TRAIN) + [n for n in args.extra_train.split(",") if n]
    train = S.concat([d for n in names if (d := S.load(args.feats, n)) is not None])
    sets = {n: d for n in S.REPORT if (d := S.load(args.feats, n)) is not None}
    dev = sets["milk_dev"]
    yd, md = ham7(dev["y"]), dev["mod"]
    head = np.load(args.head)
    lr = lambda x: x @ head["weight"].T + head["bias"]  # noqa: E731
    report = {"train_counts": {n: int((train["source"] == n).sum()) for n in np.unique(train["source"])}, "variants": {}}

    def add(tag, fn, extra=None):
        res = summarize(sets, fn)
        report["variants"][tag] = {"dev_score": dev_score(yd, fn(dev["x"]), md), "sets": res, **(extra or {})}
        brief(tag, res)

    add("lr", lr)
    off, s = fit_offsets(yd, lr(dev["x"]), md)
    add("lr+offsets", lambda x: lr(x) + off,
        {"offsets": dict(zip(L.HAM7, off.tolist())), "nested_dev_score": nested_offsets(yd, lr(dev["x"]), md, dev["group"])})
    mlps = []
    for seed in range(args.seeds):
        fn, s, ep = train_mlp(train, dev, SEED + seed)
        mlps.append(fn)
        print(json.dumps({"mlp_seed": seed, "dev_score": round(s, 4), "epoch": ep}), flush=True)
    mlp = lambda x: np.log(np.mean([softmax(f(x)) for f in mlps], 0) + 1e-12)  # noqa: E731
    add("mlp", mlp)
    ens = lambda x: np.log(0.5 * softmax(lr(x)) + 0.5 * np.exp(mlp(x)) + 1e-12)  # noqa: E731
    add("lr+mlp", ens)
    off2, _ = fit_offsets(yd, ens(dev["x"]), md)
    add("lr+mlp+offsets", lambda x: ens(x) + off2,
        {"offsets": dict(zip(L.HAM7, off2.tolist())), "nested_dev_score": nested_offsets(yd, ens(dev["x"]), md, dev["group"])})
    unbiased = {t: v.get("nested_dev_score", v["dev_score"]) for t, v in report["variants"].items()}
    best = max(unbiased, key=unbiased.get)
    report["selection"] = {"rule": "max milk_dev clin bacc + 0.5 derm bacc (nested estimate for offset variants)",
                           "scores": unbiased, "best": best}
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({"best": best, "scores": {t: round(v["dev_score"], 4) for t, v in report["variants"].items()}}))


if __name__ == "__main__":
    main()
