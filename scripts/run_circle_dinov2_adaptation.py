"""CIRCLe Stage 2 Domain Adaptation for DINOv2 (Brazo 2 Classifier).

Couples DermLIP-SLIC segmentation (with TiSage superpixel contrast prior and
bio-admissible component selection) with CIRCLe CIE-Lab tone shifting strictly
on healthy perilesional skin to train and evaluate fair linear probes on a
frozen DINOv2 representation (CRISP-ML(Q) Brazo 2).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dermssl import circle  # noqa: E402
from dermssl.pipeline_segment_circle import DermLIPCirclePipeline  # noqa: E402

LOGGER = logging.getLogger("circle_dinov2_adaptation")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESOLUTION = 224


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Backbone & Models
    parser.add_argument(
        "--backbone",
        default="dinov2_vitb14",
        choices=("dinov2_vitb14", "dinov2_vits14"),
        help="DINOv2 architecture to freeze and extract representations from.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        default=None,
        help="Optional path to custom SSL pretrained weights (e.g. A*/B* SimSiam DINOv2).",
    )
    parser.add_argument(
        "--segmenter-checkpoint",
        type=Path,
        required=True,
        help="Path to DermLIP teacher U-Net weights (e.g. Run 010 best_teacher_model.pt).",
    )
    # Data & Manifests
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the source training/validation dermatoscopic images.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="CSV with image metadata: image_id, label, lesion_id/patient_id, and optional fst.",
    )
    parser.add_argument(
        "--feature-bundle",
        type=Path,
        default=None,
        help="Optional path to pre-extracted NPZ feature bundle (skips extraction if exists).",
    )
    # Output & Device
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for feature bundles, checkpoints, and evaluation reports.",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="Compute device (e.g. cuda:0, cuda:1, cpu).",
    )
    # Hyperparameters
    parser.add_argument(
        "--shift-buckets",
        type=int,
        default=2,
        help="FST level shift for CIRCLe synthesis (e.g. +1 or +2 toward darker skin).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for the probe head.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for extraction and training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for probe optimizer (AdamW).",
    )
    parser.add_argument(
        "--reg-weight",
        type=float,
        default=1.0,
        help="Regularization weight lambda for MSE loss between original and shifted logits.",
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=0.5,
        help="Proportion of synthetic shifted samples in the synthetic_blend probe.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast smoke validation (16 train, 8 val, 2 epochs).",
    )
    return parser.parse_args(argv)


def load_dinov2_backbone(
    arch: str, weights_path: Optional[Path], device: Any
) -> Any:
    """Load DINOv2 vision transformer, freeze all weights and place in eval mode."""
    import torch

    LOGGER.info("Loading DINOv2 backbone '%s' on %s...", arch, device)
    if weights_path and Path(weights_path).is_file():
        LOGGER.info("Loading custom pretrained weights from: %s", weights_path)
        backbone = torch.hub.load("facebookresearch/dinov2", arch, pretrained=False)
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Strip potential prefixes
        clean_state = {k.replace("backbone.", "").replace("encoder.", ""): v for k, v in state_dict.items()}
        backbone.load_state_dict(clean_state, strict=False)
    else:
        LOGGER.info("Loading official public weights from torch.hub facebookresearch/dinov2...")
        backbone = torch.hub.load("facebookresearch/dinov2", arch, pretrained=True)

    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    return backbone.to(device)


def preprocess_pil_for_dinov2(img: Image.Image, device: Any) -> Any:
    """Apply DINOv2 standard preprocessing: Resize(256), CenterCrop(224), Normalize."""
    import torch
    from torchvision import transforms

    t = transforms.Compose([
        transforms.Resize(int(RESOLUTION * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = t(img).unsqueeze(0).to(device)
    return tensor


def extract_features_with_circle(
    pipeline: DermLIPCirclePipeline,
    backbone: Any,
    image_paths: List[Path],
    labels: List[int],
    device: Any,
    shift_buckets: int = 2,
    smoke_limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Extract DINOv2 representations for both original and CIRCLe-shifted images."""
    import torch

    if smoke_limit:
        image_paths = image_paths[:smoke_limit]
        labels = labels[:smoke_limit]

    feats_orig_list: List[np.ndarray] = []
    feats_shifted_list: List[np.ndarray] = []
    valid_labels: List[int] = []

    qa_summary = {
        "total": len(image_paths),
        "admissible_shifts": 0,
        "qa_fallbacks": 0,
    }

    LOGGER.info("Extracting DINOv2 + CIRCLe features for %d images...", len(image_paths))

    for i, img_path in enumerate(image_paths):
        if not img_path.is_file():
            continue
        try:
            img_pil = Image.open(img_path).convert("RGB")
            img_rgb = np.asarray(img_pil)

            # 1. Segment using unified DermLIP + TiSage SLIC + Bio-Admissibility
            seg_res = pipeline.segment_image(img_rgb)

            # 2. Apply CIRCLe shift strictly to healthy perilesional skin
            shifted_rgb, was_shifted = pipeline.apply_circle_shift(
                img_rgb, seg_res, shift_buckets=shift_buckets
            )
            if was_shifted:
                qa_summary["admissible_shifts"] += 1
            else:
                qa_summary["qa_fallbacks"] += 1

            # 3. Extract DINOv2 representations
            t_orig = preprocess_pil_for_dinov2(img_pil, device)
            t_shift = preprocess_pil_for_dinov2(Image.fromarray(shifted_rgb), device)

            with torch.no_grad():
                f_orig = backbone(t_orig).squeeze(0).cpu().numpy()
                f_shift = backbone(t_shift).squeeze(0).cpu().numpy()

            feats_orig_list.append(f_orig)
            feats_shifted_list.append(f_shift)
            valid_labels.append(labels[i])

            if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
                LOGGER.info("Feature extraction progress: %d / %d (%.1f%%)", i + 1, len(image_paths), (i + 1) / len(image_paths) * 100)
        except Exception as exc:
            LOGGER.warning("Error processing %s: %s", img_path.name, exc)

    return (
        np.asarray(feats_orig_list, dtype=np.float32),
        np.asarray(feats_shifted_list, dtype=np.float32),
        np.asarray(valid_labels, dtype=np.int64),
        qa_summary,
    )


def train_linear_probe(
    x_orig: np.ndarray,
    x_shift: np.ndarray,
    y: np.ndarray,
    mode: str,
    n_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    reg_weight: float,
    synthetic_ratio: float,
    seed: int,
    device: Any,
) -> Any:
    """Train linear probe head under baseline, regularization, or synthetic-blend mode."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from dermssl.finetune.circle_torch import ProbeHead, build_probe_optimizer, joint_regularization_loss

    torch.manual_seed(seed)
    in_features = x_orig.shape[1]
    probe = ProbeHead(in_features, n_classes).to(device)
    optimizer = build_probe_optimizer(probe, lr=lr)

    # Class-balanced loss weights
    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    xo = torch.from_numpy(x_orig).to(device)
    xs = torch.from_numpy(x_shift).to(device)
    yt = torch.from_numpy(y).to(device)

    rng = np.random.RandomState(seed)
    probe.train()

    for epoch in range(epochs):
        perm = rng.permutation(len(y))
        for start in range(0, len(perm), batch_size):
            idx = torch.as_tensor(perm[start : start + batch_size], dtype=torch.long, device=device)
            orig_b = xo[idx]
            shift_b = xs[idx]
            target_b = yt[idx]

            optimizer.zero_grad(set_to_none=True)

            if mode == "baseline":
                loss = criterion(probe(orig_b), target_b)
            elif mode == "regularization":
                loss = joint_regularization_loss(
                    probe, orig_b, shift_b, target_b, criterion, reg_weight=reg_weight
                )
            elif mode == "synthetic_blend":
                choose_shift = circle.synthetic_blend_mask(
                    len(idx), target_ratio=synthetic_ratio, seed=seed + epoch * 1000 + start
                )
                mask_t = torch.as_tensor(choose_shift, dtype=torch.bool, device=device)[:, None]
                mixed = torch.where(mask_t, shift_b, orig_b)
                task_loss = criterion(probe(mixed), target_b)
                reg_term = F.mse_loss(probe(orig_b), probe(shift_b))
                loss = task_loss + reg_weight * reg_term
            else:
                raise ValueError(f"Unknown mode: {mode}")

            loss.backward()
            optimizer.step()

    probe.eval()
    return probe


def evaluate_probe(
    probe: Any, x_val: np.ndarray, y_val: np.ndarray, n_classes: int, device: Any
) -> Dict[str, Any]:
    """Compute Macro-F1, Balanced Accuracy, Accuracy, and per-class recall/precision."""
    import torch
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

    probe.eval()
    with torch.no_grad():
        xv = torch.from_numpy(x_val).to(device)
        logits = probe(xv)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_val, preds))
    overall_acc = float((preds == y_val).mean())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, preds, labels=list(range(n_classes)), zero_division=0
    )

    per_class = {}
    for c in range(n_classes):
        per_class[str(c)] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }

    return {
        "macro_f1": round(macro_f1, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "accuracy": round(overall_acc, 4),
        "per_class": per_class,
    }


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the end-to-end CIRCLe + DINOv2 domain adaptation experiment."""
    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    LOGGER.info("Starting CIRCLe + DINOv2 Adaptation Experiment (Device: %s)", device)

    # 1. Feature Extraction or Loading cached Bundle
    bundle_path = args.feature_bundle or (out_dir / "circle_dinov2_features.npz")
    if bundle_path.is_file():
        LOGGER.info("Loading pre-computed feature bundle: %s", bundle_path)
        bundle = np.load(bundle_path, allow_pickle=False)
        x_train_orig = bundle["x_train_original"]
        x_train_shift = bundle["x_train_shifted"]
        y_train = bundle["y_train"]
        x_val = bundle["x_val"]
        y_val = bundle["y_val"]
        qa_summary = {"cached": True}
    else:
        # Load DINOv2 & DermLIP-SLIC
        backbone = load_dinov2_backbone(args.backbone, args.backbone_weights, device)
        seg_pipeline = DermLIPCirclePipeline(
            checkpoint_path=str(args.segmenter_checkpoint),
            device=str(device),
        )

        # Gather images & labels
        all_imgs = sorted(list(Path(args.images_dir).glob("*.jpg")) + list(Path(args.images_dir).glob("*.png")))
        if not all_imgs:
            raise FileNotFoundError(f"No images found in {args.images_dir}")

        # Load real clinical labels and lesion/patient IDs from metadata CSV
        image_to_label: Dict[str, int] = {}
        image_to_group: Dict[str, str] = {}
        ham_classes = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
        class_to_idx = {c: i for i, c in enumerate(ham_classes)}

        if args.metadata_csv and Path(args.metadata_csv).is_file():
            import csv
            with open(args.metadata_csv, "r", encoding="utf-8") as csv_f:
                reader = csv.DictReader(csv_f)
                for row in reader:
                    img_id = row.get("image_id", "").strip()
                    dx = row.get("dx", row.get("diagnostic", row.get("label", ""))).strip().lower()
                    group_id = row.get("lesion_id", row.get("patient_id", img_id)).strip()
                    if dx in class_to_idx:
                        image_to_label[img_id] = class_to_idx[dx]
                        image_to_group[img_id] = group_id

        # Filter and align images with verified labels
        valid_items = []
        for p in all_imgs:
            stem = p.stem
            if stem in image_to_label:
                valid_items.append((p, image_to_label[stem], image_to_group[stem]))

        if not valid_items:
            if not args.smoke:
                raise ValueError(
                    f"Metadata CSV ({args.metadata_csv}) did not match any image IDs in {args.images_dir}. "
                    "For valid thesis evaluation, verified clinical labels are mandatory."
                )
            LOGGER.warning("No matched metadata in smoke mode; falling back to diagnostic index assignment.")
            valid_items = [(p, i % len(ham_classes), p.stem) for i, p in enumerate(all_imgs)]

        # Group-aware (lesion_id disjoint) deterministic split
        rng = np.random.RandomState(args.seed)
        unique_groups = sorted(list(set(item[2] for item in valid_items)))
        rng.shuffle(unique_groups)

        train_groups = set(unique_groups[:int(0.8 * len(unique_groups))])
        train_imgs = [item[0] for item in valid_items if item[2] in train_groups]
        train_labels = [item[1] for item in valid_items if item[2] in train_groups]
        val_imgs = [item[0] for item in valid_items if item[2] not in train_groups]
        val_labels = [item[1] for item in valid_items if item[2] not in train_groups]

        smoke_limit = 16 if args.smoke else None
        smoke_val = 8 if args.smoke else None

        LOGGER.info("--- Extracting Training Features (Original + CIRCLe Shifted) ---")
        x_train_orig, x_train_shift, y_train, qa_summary = extract_features_with_circle(
            seg_pipeline, backbone, train_imgs, train_labels, device,
            shift_buckets=args.shift_buckets, smoke_limit=smoke_limit
        )

        LOGGER.info("--- Extracting Validation Features ---")
        x_val, _, y_val, _ = extract_features_with_circle(
            seg_pipeline, backbone, val_imgs, val_labels, device,
            shift_buckets=0, smoke_limit=smoke_val
        )

        # Save bundle
        np.savez_compressed(
            bundle_path,
            x_train_original=x_train_orig,
            x_train_shifted=x_train_shift,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            backbone_frozen=True,
        )
        LOGGER.info("Feature bundle saved successfully to: %s", bundle_path)

    unique_classes, y_train_mapped = np.unique(y_train, return_inverse=True)
    n_classes = len(unique_classes)
    class_map = {int(c): i for i, c in enumerate(unique_classes)}
    y_val_mapped = np.array([class_map.get(int(c), 0) for c in y_val], dtype=np.int64)

    epochs = 2 if args.smoke else args.epochs

    # 2. Train the 3 Comparative Probe Heads
    results = {
        "experiment": "circle_dinov2_adaptation_brazo2",
        "backbone": args.backbone,
        "n_classes": n_classes,
        "qa_summary": qa_summary,
        "probes": {},
    }

    variants = ("baseline", "regularization", "synthetic_blend")
    for variant in variants:
        LOGGER.info(">>> Training Probe Variant: %s (epochs=%d, lr=%.1e, lambda=%.2f) <<<", variant, epochs, args.lr, args.reg_weight)
        probe = train_linear_probe(
            x_orig=x_train_orig,
            x_shift=x_train_shift,
            y=y_train_mapped,
            mode=variant,
            n_classes=n_classes,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            reg_weight=args.reg_weight,
            synthetic_ratio=args.synthetic_ratio,
            seed=args.seed,
            device=device,
        )

        eval_metrics = evaluate_probe(probe, x_val, y_val_mapped, n_classes, device)
        LOGGER.info("Variant '%s' Results: Macro-F1: %.4f | Balanced Acc: %.4f | Acc: %.4f",
                    variant, eval_metrics["macro_f1"], eval_metrics["balanced_accuracy"], eval_metrics["accuracy"])
        results["probes"][variant] = eval_metrics

    # Save summary report
    report_file = out_dir / "adaptation_report.json"
    report_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Full adaptation report saved to: %s", report_file)
    return results


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
