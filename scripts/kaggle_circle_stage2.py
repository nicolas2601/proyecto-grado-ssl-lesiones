"""Run CIRCLe Stage 2 on real frozen-backbone feature pairs.

The input NPZ must contain features extracted from the same frozen backbone:
``x_train_original``, ``x_train_shifted``, ``y_train``, ``x_val``, ``y_val``,
``fst_buckets_val`` and scalar ``backbone_frozen=True``. Shifted features must
come from images transformed with ``dermssl.circle.shift_skin_tone`` and the
lesion masks produced during Stage 1. This runner never invents features,
labels, buckets or outcomes.

The Stage 1 gate JSON may be either ``{"status": "FAIL"}`` or a v5 summary
with a ``gates`` list containing ``G3 equidad por fototipo``. PASS/SKIP writes
an explicit skipped result. FAIL trains three matched linear probes: baseline,
regularization-only and regularization plus a 50/50 synthetic blend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl import circle, metrics  # noqa: E402

SMOKE_EPOCHS = 2
SMOKE_MAX_TRAIN = 256
SMOKE_MAX_VAL = 128
REQUIRED_BUNDLE_KEYS = (
    "x_train_original",
    "x_train_shifted",
    "y_train",
    "x_val",
    "y_val",
    "fst_buckets_val",
    "backbone_frozen",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--feature-bundle", required=True)
    parser.add_argument("--stage1-gate", required=True)
    parser.add_argument("--output-json", default="circle_stage2_results.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-ratio", type=float, default=0.5)
    parser.add_argument("--min-bucket-n", type=int, default=30)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not Path(args.feature_bundle).is_file():
        raise ValueError(f"feature bundle not found: {args.feature_bundle}")
    if not Path(args.stage1_gate).is_file():
        raise ValueError(f"Stage 1 gate JSON not found: {args.stage1_gate}")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if not 0.0 <= args.synthetic_ratio <= 1.0:
        raise ValueError("synthetic-ratio must be within [0, 1]")
    if args.min_bucket_n <= 0 or args.bootstrap_resamples <= 0:
        raise ValueError("min-bucket-n and bootstrap-resamples must be positive")


def load_stage1_gate_status(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "status" in payload:
        return str(payload["status"])
    for gate in payload.get("gates", []):
        if gate.get("gate", gate.get("name")) == "G3 equidad por fototipo":
            return str(gate.get("status", "SKIP"))
    raise ValueError("Stage 1 JSON has no G3 equidad por fototipo gate")


def _class_weights(torch_module, y_train: np.ndarray, n_classes: int, device):
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    return torch_module.tensor(weights, dtype=torch_module.float32, device=device)


def _train_variant(
    torch_module,
    functional,
    circle_torch,
    *,
    mode: str,
    x_original: np.ndarray,
    x_shifted: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    reg_weight: float,
    synthetic_ratio: float,
    seed: int,
    device,
):
    torch_module.manual_seed(seed)
    probe = circle_torch.ProbeHead(x_original.shape[1], n_classes).to(device)
    optimizer = circle_torch.build_probe_optimizer(probe, lr=lr)
    criterion = torch_module.nn.CrossEntropyLoss(
        weight=_class_weights(torch_module, y_train, n_classes, device)
    )
    xo = torch_module.from_numpy(np.asarray(x_original, dtype=np.float32)).to(device)
    xs = torch_module.from_numpy(np.asarray(x_shifted, dtype=np.float32)).to(device)
    yt = torch_module.from_numpy(np.asarray(y_train, dtype=np.int64)).to(device)

    rng = np.random.RandomState(seed)
    probe.train()
    for epoch in range(epochs):
        order = rng.permutation(len(y_train))
        for start in range(0, len(order), batch_size):
            idx_np = order[start : start + batch_size]
            idx = torch_module.as_tensor(idx_np, dtype=torch_module.long, device=device)
            original = xo[idx]
            shifted = xs[idx]
            targets = yt[idx]
            optimizer.zero_grad(set_to_none=True)
            if mode == "baseline":
                loss = criterion(probe(original), targets)
            elif mode == "regularization":
                loss = circle_torch.joint_regularization_loss(
                    probe,
                    original,
                    shifted,
                    targets,
                    criterion,
                    reg_weight=reg_weight,
                )
            elif mode == "synthetic_blend":
                choose_shifted = circle.synthetic_blend_mask(
                    len(idx_np),
                    target_ratio=synthetic_ratio,
                    seed=seed + epoch * 100_000 + start,
                )
                choose_shifted = torch_module.as_tensor(
                    choose_shifted, dtype=torch_module.bool, device=device
                )
                mixed = torch_module.where(choose_shifted[:, None], shifted, original)
                logits_original = probe(original)
                logits_shifted = probe(shifted)
                task_loss = criterion(probe(mixed), targets)
                loss = task_loss + reg_weight * functional.mse_loss(
                    logits_original, logits_shifted
                )
            else:
                raise ValueError(f"unknown Stage 2 variant: {mode}")
            loss.backward()
            optimizer.step()
    return probe


def _predict_proba(torch_module, probe, x_val: np.ndarray, device) -> np.ndarray:
    probe.eval()
    values = torch_module.from_numpy(np.asarray(x_val, dtype=np.float32)).to(device)
    with torch_module.no_grad():
        return torch_module.softmax(probe(values), dim=1).cpu().numpy()


def run(args: argparse.Namespace, gate_status: str) -> dict:
    import torch
    import torch.nn.functional as F

    from dermssl.finetune import circle_torch

    bundle = np.load(args.feature_bundle, allow_pickle=False)
    missing = [key for key in REQUIRED_BUNDLE_KEYS if key not in bundle.files]
    if missing:
        raise ValueError(f"feature bundle missing keys: {missing}")
    if not bool(np.asarray(bundle["backbone_frozen"]).item()):
        raise ValueError("feature bundle must attest backbone_frozen=True")

    x_original = np.asarray(bundle["x_train_original"], dtype=np.float32)
    x_shifted = np.asarray(bundle["x_train_shifted"], dtype=np.float32)
    y_train = np.asarray(bundle["y_train"], dtype=np.int64)
    x_val = np.asarray(bundle["x_val"], dtype=np.float32)
    y_val = np.asarray(bundle["y_val"], dtype=np.int64)
    fst_val = np.asarray(bundle["fst_buckets_val"], dtype=object)
    if x_original.shape != x_shifted.shape or len(x_original) != len(y_train):
        raise ValueError("training feature pairs and labels must have matching shapes")
    if len(x_val) != len(y_val) or len(y_val) != len(fst_val):
        raise ValueError("validation features, labels and FST buckets must align")
    if x_original.ndim != 2 or x_val.ndim != 2 or x_original.shape[1] != x_val.shape[1]:
        raise ValueError("feature arrays must be 2-D with one shared feature dimension")

    epochs = SMOKE_EPOCHS if args.smoke else args.epochs
    bootstrap_resamples = min(100, args.bootstrap_resamples) if args.smoke else args.bootstrap_resamples
    if args.smoke:
        x_original = x_original[:SMOKE_MAX_TRAIN]
        x_shifted = x_shifted[:SMOKE_MAX_TRAIN]
        y_train = y_train[:SMOKE_MAX_TRAIN]
        x_val = x_val[:SMOKE_MAX_VAL]
        y_val = y_val[:SMOKE_MAX_VAL]
        fst_val = fst_val[:SMOKE_MAX_VAL]

    n_classes = int(max(y_train.max(), y_val.max())) + 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = {}
    for name in ("baseline", "regularization", "synthetic_blend"):
        probe = _train_variant(
            torch,
            F,
            circle_torch,
            mode=name,
            x_original=x_original,
            x_shifted=x_shifted,
            y_train=y_train,
            n_classes=n_classes,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            reg_weight=args.reg_weight,
            synthetic_ratio=args.synthetic_ratio,
            seed=args.seed,
            device=device,
        )
        variants[name] = _predict_proba(torch, probe, x_val, device)

    summaries = {
        name: metrics.stratified_summary(
            y_val, proba, fst_val, n_classes, min_n=args.min_bucket_n
        )
        for name, proba in variants.items()
    }
    bootstrap_inputs = {
        name: {
            "y_true": y_val,
            "proba": proba,
            "buckets": fst_val,
            "n_classes": n_classes,
        }
        for name, proba in variants.items()
    }
    comparison = metrics.comparison_report(
        summaries["baseline"],
        summaries["regularization"],
        summaries["synthetic_blend"],
        min_bucket_n=args.min_bucket_n,
        bootstrap_inputs=bootstrap_inputs,
        n_resamples=bootstrap_resamples,
        seed=args.seed,
    )
    return {
        "status": "completed",
        "stage1_gate_status": gate_status,
        "smoke": args.smoke,
        "valid_for_thesis": not args.smoke,
        "protocol": {
            "probe": "linear",
            "backbone_frozen": True,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "reg_weight": args.reg_weight,
            "synthetic_ratio": args.synthetic_ratio,
            "seed": args.seed,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "stratified": summaries,
        "comparison": comparison,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        validate_args(args)
        gate_status = load_stage1_gate_status(Path(args.stage1_gate))
        if not circle.should_run_stage2(gate_status):
            result = {
                "status": "skipped",
                "stage1_gate_status": gate_status,
                "reason": "Stage 1 did not report measurable bias",
                "smoke": args.smoke,
                "valid_for_thesis": False,
            }
        else:
            result = run(args, gate_status)
    except ValueError as exc:
        raise SystemExit(f"config error: {exc}") from exc
    output = Path(args.output_json)
    _write_json(output, result)
    print(f"wrote {output} | status={result['status']}")


if __name__ == "__main__":
    main()
