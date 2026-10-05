"""Reference implementation of the frozen-backbone classifier pipeline.

This is the executable version of `classifier_manifest.json`. The manifest says
what the pipeline is; this file proves it, end to end, on real images. A demo
backend that disagrees with this script disagrees with the paper.

Deliberately standalone: no sklearn, no repo imports, no project layout. Only
torch, torchvision, numpy and PIL. That is the point -- if reproducing the
published numbers needed the training repo, the artifact would not be portable,
and "it works on my machine" would be the only guarantee on offer.

The preprocessing is read FROM the manifest rather than hardcoded, so the
artifact stays the single source of truth. One exception is asserted: the
resize edge must not be 256. `int(224 * 1.14)` is 255, and 256 is the value a
careful person reaches for from memory. That single off-by-one silently shifts
every crop and quietly degrades the predictions -- no exception, no warning,
just worse numbers. It is the most likely way a consumer drifts, so it fails
loudly here instead.

Usage
-----
Verify the pipeline reproduces the shipped fixture (do this first, always):

    python infer_reference.py \
        --arm-dir ssl_proposed_vitb14 \
        --weights /path/to/simsiam_dinov2_backbone_final.pt \
        --images-dir /path/to/HAM10000/images

Classify an arbitrary image (only meaningful once the golden check passes):

    python infer_reference.py \
        --arm-dir ssl_proposed_vitb14 \
        --weights /path/to/simsiam_dinov2_backbone_final.pt \
        --image /path/to/lesion.jpg

The first run downloads the DINOv2 architecture definition via torch.hub and
caches it. The downloaded weights are discarded: the checkpoint replaces them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

FORBIDDEN_RESIZE = 256
PROBE_KEYS = ("classes", "feature_dim", "mean", "scale", "coef", "intercept")


def load_manifest(arm_dir: Path) -> dict:
    path = arm_dir / "classifier_manifest.json"
    if not path.exists():
        raise SystemExit(f"ABORT: no manifest at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_transform(manifest: dict) -> transforms.Compose:
    pre = manifest["preprocess"]
    resize_to = int(pre["resize_to"])

    if resize_to == FORBIDDEN_RESIZE:
        raise SystemExit(
            "ABORT: resize_to is 256. The bank used int(224 * 1.14) = 255. "
            "Using 256 shifts the crop and degrades every prediction silently."
        )

    return transforms.Compose(
        [
            transforms.Resize(resize_to, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(int(pre["center_crop"])),
            transforms.ToTensor(),
            transforms.Normalize(pre["normalize_mean"], pre["normalize_std"]),
        ]
    )


def load_backbone(manifest: dict, weights: Path, device: torch.device) -> torch.nn.Module:
    if not weights.exists():
        raise SystemExit(f"ABORT: no weights at {weights}")

    detail = manifest["backbone_detail"]
    model = torch.hub.load(detail["hub"], detail["arch"], pretrained=False)

    state = torch.load(weights, map_location=device, weights_only=True)
    # strict=True on purpose: the checkpoint is backbone-only and must match
    # exactly. A silent partial load is how you get a half-random encoder that
    # still returns plausible-looking 768-dim vectors.
    model.load_state_dict(state)

    model.eval()
    return model.to(device)


def load_probe(arm_dir: Path, arm: str) -> dict:
    path = arm_dir / f"{arm}_probe.npz"
    if not path.exists():
        raise SystemExit(f"ABORT: no probe at {path}")

    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in PROBE_KEYS if k not in z.files]
        if missing:
            raise SystemExit(f"ABORT: probe missing keys {missing}")
        probe = {k: z[k] for k in PROBE_KEYS}

    dim = int(probe["feature_dim"])
    n_classes = len(probe["classes"])
    expected = {
        "mean": (dim,),
        "scale": (dim,),
        "coef": (n_classes, dim),
        "intercept": (n_classes,),
    }
    for key, shape in expected.items():
        if tuple(probe[key].shape) != shape:
            raise SystemExit(f"ABORT: {key} has shape {probe[key].shape}, expected {shape}")

    return probe


@torch.no_grad()
def extract_features(model: torch.nn.Module, image: Path, transform, device) -> np.ndarray:
    """Plain forward. The DINOv2 hub model already returns the pooled embedding.

    Calling forward_features or slicing tokens here produces a different vector
    of the same shape -- which the probe will happily consume and mislabel.
    """
    pixels = Image.open(image).convert("RGB")
    batch = transform(pixels).unsqueeze(0).to(device)
    return model(batch).float().cpu().numpy()[0]


def predict(feature: np.ndarray, probe: dict) -> np.ndarray:
    """softmax(coef @ ((x - mean) / scale) + intercept), in float32."""
    z = (feature.astype(np.float32) - probe["mean"]) / probe["scale"]
    logits = probe["coef"] @ z + probe["intercept"]
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()


def resolve_image(images_dir: Path, image_id: str) -> Path:
    for suffix in (".jpg", ".JPG", ".jpeg", ".png"):
        candidate = images_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    raise SystemExit(f"ABORT: no image file for {image_id} under {images_dir}")


def verify_golden(arm_dir: Path, images_dir: Path, model, transform, probe, device) -> int:
    golden = json.loads((arm_dir / "golden.json").read_text(encoding="utf-8"))
    classes = list(golden["classes"])
    tol = float(golden["tolerance"])

    failures = []
    for case in golden["cases"]:
        image = resolve_image(images_dir, case["image_id"])
        feature = extract_features(model, image, transform, device)
        got = predict(feature, probe)

        want = np.array([case["probabilities"][c] for c in classes], dtype=np.float64)
        worst = float(np.abs(got - want).max())
        predicted = classes[int(np.argmax(got))]

        status = "ok " if worst <= tol else "FAIL"
        if worst > tol:
            failures.append((case["image_id"], worst))

        print(
            f"  {status} {case['image_id']}  true={case['true_class']:<5} "
            f"pred={predicted:<5} (expected {case['predicted_class']:<5}) "
            f"max_abs_diff={worst:.2e}"
        )

    if failures:
        print(f"\nGOLDEN FAILED: {len(failures)}/{len(golden['cases'])} cases exceed tol={tol:g}")
        print("This is NOT numeric noise. Check, in order:")
        print("  1. resize edge is 255, not 256")
        print("  2. plain model(x) forward -- not forward_features, no token slicing")
        print("  3. no L2-normalisation of the feature vector")
        print("  4. the checkpoint actually loaded (strict=True, no missing keys)")
        return 1

    print(f"\nGOLDEN OK: {len(golden['cases'])}/{len(golden['cases'])} cases within tol={tol:g}")
    return 0


def classify_one(image: Path, model, transform, probe, device) -> int:
    classes = list(probe["classes"])
    feature = extract_features(model, image, transform, device)
    proba = predict(feature, probe)

    order = np.argsort(proba)[::-1]
    print(f"\n{image.name}")
    for rank in order:
        bar = "#" * int(round(proba[rank] * 40))
        print(f"  {classes[rank]:<6} {proba[rank]:6.2%}  {bar}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, help="HAM10000 images, for --verify")
    parser.add_argument("--image", type=Path, help="classify a single image instead")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.images_dir and not args.image:
        raise SystemExit("ABORT: pass --images-dir (verify golden) or --image (classify one)")

    arm_dir = args.arm_dir.resolve()
    manifest = load_manifest(arm_dir)
    arm = manifest["arm"]
    device = torch.device(args.device)

    print(f"arm      : {arm}")
    print(f"backbone : {manifest['backbone']}  (frozen)")
    print(f"probe    : {manifest['probe']['formula']}")
    print(f"device   : {device}")

    transform = build_transform(manifest)
    probe = load_probe(arm_dir, arm)
    model = load_backbone(manifest, args.weights.resolve(), device)

    if args.image:
        return classify_one(args.image.resolve(), model, transform, probe, device)

    print("\nverifying golden fixture:")
    return verify_golden(arm_dir, args.images_dir.resolve(), model, transform, probe, device)


if __name__ == "__main__":
    sys.exit(main())
