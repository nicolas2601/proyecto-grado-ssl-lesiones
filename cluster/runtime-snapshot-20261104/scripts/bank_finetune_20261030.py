"""Supervised fine-tuning of a bank model inside the fixed patient-grouped nested CV.

Same folds, seeds, classes and report format as evaluate_overlay_readout.py, so arm_stats.py compares
the result against the SSL baseline unchanged. Epochs are selected on an inner patient-grouped
validation split; the outer test fold is never used for training or selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_overlay_readout import CLASSES, group_sha, load_labels, validate_support  # noqa: E402
from probe_preprocessing_readout import INNER_FOLDS, OUTER_FOLDS, SEEDS, grouped_folds, metrics  # noqa: E402

OVERLAY = ROOT / "results/content-exclusion-overlay-v1"
META = {"PAD": ("/home/jovyan/work/pad_ufes_20/metadata.csv",
                "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527", "pad_adapt"),
        "HAM": ("/home/jovyan/work/datos/ham10000/HAM10000_metadata.csv",
                "06aee5e603d313ba69de38263053c792a1dff030eb9c44a96ceb477ceefb99a5", "ham_dev")}
EPOCHS, BATCH, WEIGHT_DECAY = 10, 32, 0.05


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cross_val_oof(labels, groups, classes, train_predict):
    """Nested grouped CV; train_predict(train_in, valid, test, seed) -> probabilities for `test`."""
    validate_support(labels, groups, classes)
    runs = []
    for seed in SEEDS:
        oof, seen, traces = np.zeros((len(labels), len(classes))), np.zeros(len(labels), int), []
        for fold, (train, test) in enumerate(grouped_folds(labels, groups, OUTER_FOLDS, seed)):
            inner_train, valid = next(iter(grouped_folds(labels[train], groups[train], INNER_FOLDS, seed + fold)))
            probabilities = train_predict(train[inner_train], train[valid], test, seed * 10 + fold)
            if probabilities.shape != (len(test), len(classes)):
                raise ValueError("Trainer returned probabilities of the wrong shape")
            oof[test] = probabilities
            seen[test] += 1
            traces.append({"fold": fold, "train_groups_sha256": group_sha(train, groups),
                           "test_groups_sha256": group_sha(test, groups)})
        if not np.all(seen == 1):
            raise ValueError("OOF coverage must be exactly one prediction per allowed image")
        runs.append({"seed": seed, "metrics": metrics(labels, oof, classes), "folds": traces,
                     "oof_probabilities": oof.tolist()})
        print(json.dumps({"CV_seed": seed, "macro_f1": runs[-1]["metrics"]["macro_f1"]}), flush=True)
    return runs


def build_report(kind, rows, labels, runs, provenance):
    classes = CLASSES[kind]
    groups = np.asarray([row["identity_component_id"] for row in rows])
    report = {"kind": kind, "classes": classes, "image_ids": [r["image_id"] for r in rows],
              "groups": groups.tolist(), "provenance": provenance, "runs": runs,
              "scope": "Exploratory supervised fine-tuning development, not clinical validation; CV seeds are not independent training replicates",
              "settings": {"reader": "supervised fine-tune", "epochs": EPOCHS, "batch": BATCH,
                           "outer": OUTER_FOLDS, "inner": INNER_FOLDS, "CV_seeds": SEEDS}}
    report["summary"] = {name: {"mean": float(np.mean([r["metrics"][name] for r in runs])),
                                "sd_ddof0": float(np.std([r["metrics"][name] for r in runs]))}
                         for name in ("macro_f1", "balanced_accuracy", "ece")}
    report["support"] = {label: {"images": int(np.sum(labels == label)),
                                 "groups": len(set(groups[labels == label]))} for label in classes}
    return report


def augment(batch):
    """Random flips and 90-degree turns plus mild brightness/contrast jitter, per sample, on the GPU."""
    import torch
    flip_h = torch.rand(len(batch), device=batch.device) < 0.5
    flip_v = torch.rand(len(batch), device=batch.device) < 0.5
    batch = torch.where(flip_h.view(-1, 1, 1, 1), batch.flip(3), batch)
    batch = torch.where(flip_v.view(-1, 1, 1, 1), batch.flip(2), batch)
    contrast = 1 + 0.2 * (torch.rand(len(batch), 1, 1, 1, device=batch.device) - 0.5)
    brightness = 0.2 * (torch.rand(len(batch), 1, 1, 1, device=batch.device) - 0.5)
    return batch * contrast + brightness


def make_trainer(model_id, images, y_index, n_classes, device):
    import torch
    from sklearn.metrics import f1_score
    from torch import nn
    from bank_models_20261030 import build_backbone, normalize, spec
    info = spec(model_id)

    def predict(model, indices):
        model.eval()
        outputs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(indices), 64):
                batch = normalize(images[indices[start:start + 64]].to(device), model_id)
                outputs.append(model(batch).float().softmax(dim=1).cpu())
        return torch.cat(outputs).numpy()

    def train_predict(train_in, valid, test, seed):
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        model = build_backbone(model_id, n_classes).to(device)
        counts = np.bincount(y_index[train_in], minlength=n_classes).clip(min=1)
        weights = torch.tensor(counts.sum() / (n_classes * counts), dtype=torch.float32, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=info["lr"], weight_decay=WEIGHT_DECAY)
        steps = EPOCHS * int(np.ceil(len(train_in) / BATCH))
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=info["lr"], total_steps=steps, pct_start=0.1)
        best_f1, best_state = -1.0, None
        rng = np.random.default_rng(seed)
        for _ in range(EPOCHS):
            model.train()
            order = rng.permutation(train_in)
            for start in range(0, len(order), BATCH):
                chunk = order[start:start + BATCH]
                if len(chunk) < 2:
                    continue
                batch = augment(normalize(images[chunk].to(device), model_id))
                target = torch.tensor(y_index[chunk], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = nn.functional.cross_entropy(model(batch).float(), target, weight=weights)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                schedule.step()
            score = f1_score(y_index[valid], predict(model, valid).argmax(1), labels=list(range(n_classes)),
                             average="macro", zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        return predict(model, test)

    return train_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=sorted(CLASSES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing evidence")
    import torch
    from bank_models_20261030 import load_images, spec
    metadata, metadata_sha, role = META[args.kind]
    manifest = OVERLAY / f"{role}.jsonl"
    receipt = json.loads((OVERLAY / "receipt.json").read_text())
    if receipt["status"] != "COMPLETE" or file_sha256(manifest) != receipt["output_sha256"][manifest.name]:
        raise ValueError("Overlay provenance mismatch")
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    labels = load_labels(args.kind, rows, Path(metadata), metadata_sha)
    classes = CLASSES[args.kind]
    y_index = np.asarray([classes.index(label) for label in labels])
    groups = np.asarray([row["identity_component_id"] for row in rows])
    images = load_images(rows, spec(args.model)["interp"])
    started = time.monotonic()
    trainer = make_trainer(args.model, images, y_index, len(classes), torch.device("cuda"))
    runs = cross_val_oof(labels, groups, classes, trainer)
    report = build_report(args.kind, rows, labels, runs, {
        "model": args.model, "mode": "finetune", "lr": spec(args.model)["lr"],
        "manifest_sha256": file_sha256(manifest), "metadata_sha256": metadata_sha,
        "overlay_sha256": file_sha256(OVERLAY / "receipt.json")})
    report["seconds"] = time.monotonic() - started
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "summary": report["summary"], "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
