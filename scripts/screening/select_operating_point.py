#!/usr/bin/env python3
"""Select the melanoma-priority operating point on development data (RULE.md in the output's folder).

Reads frozen 448 px CLS4+D4 development features for three encoder seeds, refits the shipped reader
recipe out of fold, and writes a policy bound to the shipped classifier bundle. No sealed role is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dermssl.release import screening_select as ss  # noqa: E402

ENCODER_SEEDS = (42, 43, 44)
SHIPPED_SEED = 42
CV_SEEDS = (20261025, 20261026, 20261027)
NESTED_SEED_OFFSET = 1000
N_SPLITS = 5
READER = {"C": 100.0, "class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000, "random_state": 42}
TASKS = {
    "HAM": {"role": "ham_dev", "model_key": "MOCO_HAM7", "target": "mel", "nevus": "nv"},
    "PAD": {"role": "pad_adapt", "model_key": "MOCO_PAD6", "target": "MEL", "nevus": "NEV"},
}
RULE = ROOT / "resultados/clasificacion/v10/screening/RULE.md"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def l2_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def load_task(features_dir: Path, labels: dict, task: str, seed: int):
    path = features_dir / f"s{seed}-{TASKS[task]['role']}-cls4_d4.npz"
    data = np.load(path, allow_pickle=False)
    rows = labels[task]
    if data["image_ids"].tolist() != rows["image_ids"] or data["groups"].tolist() != rows["groups"]:
        raise ValueError(f"{path.name}: rows do not match the development label table")
    classes = rows["classes"]
    y = np.asarray([classes.index(label) for label in rows["labels"]])
    return l2_rows(data["features"]), y, np.asarray(rows["groups"]), classes, sha256_file(path)


def fit_reader(x: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return LogisticRegression(**READER).fit(x, y)


def oof_probabilities(x, y, groups, cv_seed: int) -> np.ndarray:
    from sklearn.model_selection import StratifiedGroupKFold

    probabilities = np.zeros((len(y), int(y.max()) + 1))
    splitter = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=cv_seed)
    for train, valid in splitter.split(x, y, groups):
        if set(groups[train]) & set(groups[valid]):
            raise ValueError("identity group leaks across folds")
        probabilities[valid] = fit_reader(x[train], y[train]).predict_proba(x[valid])
    return probabilities


def curve(y, probabilities, classes, target, nevus) -> dict[float, dict]:
    return ss.operating_curve(y, probabilities, classes, target, nevus)


def nested_estimate(y, groups, probabilities, classes, target, nevus, cv_seed: int) -> dict:
    """Select delta on four folds of the OOF rows, apply it to the fifth; metrics on the assembled rows."""
    from sklearn.model_selection import StratifiedGroupKFold

    t = classes.index(target)
    held_out = np.zeros(len(y), dtype=int)
    chosen = []
    splitter = StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=cv_seed + NESTED_SEED_OFFSET)
    for select_rows, apply_rows in splitter.split(probabilities, y, groups):
        sub = curve(y[select_rows], probabilities[select_rows], classes, target, nevus)
        delta = ss.select_delta({d: m["macro_f1"] for d, m in sub.items()})
        held_out[apply_rows] = ss.screening_predictions(probabilities[apply_rows], t, delta)
        chosen.append(delta)
    return {"deltas": chosen, "metrics": ss.operating_metrics(y, held_out, classes, target, nevus)}


def run_one(job: tuple) -> dict:
    features_dir, labels, task, seed, cv_seed = job
    x, y, groups, classes, features_sha = load_task(features_dir, labels, task, seed)
    spec = TASKS[task]
    started = time.monotonic()
    probabilities = oof_probabilities(x, y, groups, cv_seed)
    return {
        "task": task, "encoder_seed": seed, "cv_seed": cv_seed, "features_sha256": features_sha,
        "curve": {str(d): m for d, m in curve(y, probabilities, classes, spec["target"], spec["nevus"]).items()},
        "nested": nested_estimate(y, groups, probabilities, classes, spec["target"], spec["nevus"], cv_seed),
        "seconds": round(time.monotonic() - started, 1),
    }


def mean_metrics(items: list[dict]) -> dict[str, float]:
    return {key: float(np.mean([item[key] for item in items])) for key in items[0]}


def summarize(task: str, runs: list[dict]) -> dict:
    own = [r for r in runs if r["task"] == task]
    shipped = [r for r in own if r["encoder_seed"] == SHIPPED_SEED]
    pooled = {d: float(np.mean([r["curve"][str(d)]["macro_f1"] for r in shipped])) for d in ss.DELTA_GRID}
    delta = ss.select_delta(pooled)
    by_seed = {}
    for seed in ENCODER_SEEDS:
        seed_runs = [r for r in own if r["encoder_seed"] == seed]
        seed_f1 = {d: float(np.mean([r["curve"][str(d)]["macro_f1"] for r in seed_runs])) for d in ss.DELTA_GRID}
        by_seed[str(seed)] = {
            "selected_delta": ss.select_delta(seed_f1),
            "at_shipped_delta": mean_metrics([r["curve"][str(delta)] for r in seed_runs]),
            "at_zero": mean_metrics([r["curve"]["0.0"] for r in seed_runs]),
            "nested": mean_metrics([r["nested"]["metrics"] for r in seed_runs]),
            "nested_deltas": [d for r in seed_runs for d in r["nested"]["deltas"]],
        }
    return {"delta": delta, "shipped_seed_macro_f1_by_delta": {str(d): v for d, v in pooled.items()}, "by_encoder_seed": by_seed}


def shipped_head_check(features_dir: Path, labels: dict, bundle_dir: Path, task: str) -> dict:
    """Refit the shipped recipe on all development rows and compare with the bundle head."""
    x, y, _, classes, _ = load_task(features_dir, labels, task, SHIPPED_SEED)
    head = json.loads((bundle_dir / f"heads/{task}-reader.json").read_text())
    if head["classes"] != classes:
        raise ValueError(f"{task}: bundle head classes differ from the development classes")
    logits = x @ np.asarray(head["weight"]).T + np.asarray(head["bias"])
    shipped = np.exp(logits - logits.max(axis=1, keepdims=True))
    shipped /= shipped.sum(axis=1, keepdims=True)
    refit = fit_reader(x, y).predict_proba(x)
    return {"max_abs_probability_diff": float(np.abs(shipped - refit).max()),
            "argmax_agreement": float(np.mean(shipped.argmax(1) == refit.argmax(1)))}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="development rows: image_ids, labels, groups")
    parser.add_argument("--bundle-dir", type=Path, required=True, help="shipped v10.1.0 classifier bundle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    labels = json.loads(args.labels.read_text())
    manifest_path = args.bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    jobs = [(args.features_dir, labels, task, seed, cv) for task in TASKS for seed in ENCODER_SEEDS for cv in CV_SEEDS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(run_one, jobs))
    tasks = {}
    for task, spec in TASKS.items():
        tasks[task] = {**spec, **summarize(task, runs),
                       "shipped_head_check": shipped_head_check(args.features_dir, labels, args.bundle_dir, task),
                       "head_sha256": manifest["files"][f"heads/{task}-reader.json"]}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"rule_sha256": sha256_file(RULE), "script_sha256": sha256_file(Path(__file__)),
              "module_sha256": sha256_file(Path(ss.__file__)), "labels_sha256": sha256_file(args.labels),
              "reader": READER, "cv_seeds": CV_SEEDS, "encoder_seeds": ENCODER_SEEDS, "grid": ss.DELTA_GRID,
              "manifest_sha256": sha256_file(manifest_path), "model_sha256": manifest["files"]["model.onnx"],
              "release": manifest["release"], "tasks": tasks, "runs": runs}
    (args.output / "screening-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({t: {"delta": v["delta"], "head_check": v["shipped_head_check"]} for t, v in tasks.items()}))


if __name__ == "__main__":
    main()
