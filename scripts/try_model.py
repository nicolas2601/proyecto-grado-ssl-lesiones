"""Load a published backbone and classify one image, end to end.

The smallest thing that proves a downloaded checkpoint works: load it, check
it is frozen and complete, extract features for a folder of labelled images,
fit a linear probe and report the metrics. No GPU required - it runs on CPU,
slower.

    python scripts/try_model.py --backbone ssl_proposed_dinov2_vitb14_backbone.pt \\
        --ham-root /data/ham10000 --limit 400

With ``--image`` it instead prints the feature vector for a single file,
which is the check to run first when a download is suspect.

The architecture is read from the run.json published next to the checkpoint,
so a ViT-B download does not have to be paired with a remembered flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ARCH = "dinov2_vits14"
RESOLUTION = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backbone", required=True,
                        help="Path to a published .pt, or the literal 'public' for the unadapted DINOv2")
    parser.add_argument("--arch", default=None,
                        help="torch.hub name of the architecture; deduced from the run.json "
                             "next to the checkpoint when omitted")
    parser.add_argument("--image", type=Path, default=None,
                        help="Single image: print its feature vector and exit")
    parser.add_argument("--ham-root", type=Path, default=None,
                        help="HAM10000 directory (metadata CSV plus images) for the probe")
    parser.add_argument("--limit", type=int, default=400,
                        help="Images to use; a few hundred is enough to prove the checkpoint works")
    parser.add_argument("--resolution", type=int, default=RESOLUTION)
    parser.add_argument("--device", default=None, help="cuda or cpu; autodetected when omitted")
    return parser.parse_args(argv)


def manifest_beside(checkpoint: str) -> Path | None:
    """The run.json published next to a checkpoint, in either release layout."""
    path = Path(checkpoint)
    candidates = [path.parent / "run.json"]
    if path.name.endswith("_backbone.pt"):
        arm = path.name[: -len("_backbone.pt")]
        candidates.insert(0, path.parent / f"{arm}_run.json")
    return next((c for c in candidates if c.is_file()), None)


def resolve_arch(checkpoint: str, explicit: str | None) -> str:
    """Prefer the architecture recorded with the weights over a fixed default.

    More than one architecture is published, and the default used to be
    ViT-S. Handed a ViT-B checkpoint it failed with a shape error that reads
    like a corrupt download instead of a missing flag. The manifest travels
    next to the weights precisely so nobody has to remember which is which.
    """
    if explicit:
        return explicit
    if checkpoint == "public":
        return ARCH
    manifest = manifest_beside(checkpoint)
    if manifest is None:
        return ARCH
    recorded = json.loads(manifest.read_text(encoding="utf-8")).get("backbone")
    return recorded or ARCH


def load_backbone(path: str, arch: str, device):
    """The published state_dict on top of the architecture, frozen."""
    import torch

    pretrained = path == "public"
    model = torch.hub.load("facebookresearch/dinov2", arch, pretrained=pretrained)
    if not pretrained:
        state = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"el checkpoint no encaja con {arch}: faltan {len(missing)} claves, "
                f"sobran {len(unexpected)}. Primeras faltantes: {missing[:3]}"
            )
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = sum(p.requires_grad for p in model.parameters())
    if trainable:
        raise SystemExit(f"{trainable} parametros quedaron entrenables; el extractor debe estar congelado")
    return model


def preprocess(resolution: int):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(int(resolution * 1.14)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _resolve_device(requested: str | None):
    import torch

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_one(model, image_path: Path, resolution: int, device) -> dict:
    import torch
    from PIL import Image

    tensor = preprocess(resolution)(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor.to(device))
    vector = features[0].cpu()
    return {
        "image": str(image_path),
        "dim": int(vector.shape[0]),
        "norm": round(float(vector.norm()), 4),
        "first_values": [round(float(v), 4) for v in vector[:8]],
    }


def _labelled_images(ham_root: Path, limit: int):
    """(paths, labels, class names) from the HAM10000 metadata, balanced by class."""
    import pandas as pd

    metadata = next(ham_root.rglob("HAM10000_metadata.csv"), None)
    if metadata is None:
        raise SystemExit(f"no se encontro HAM10000_metadata.csv bajo {ham_root}")
    table = pd.read_csv(metadata)
    by_id = {path.stem: path for path in ham_root.rglob("*.jpg")}
    table = table[table.image_id.isin(by_id)]
    if table.empty:
        raise SystemExit(f"el metadata no coincide con ninguna imagen bajo {ham_root}")

    classes = sorted(table.dx.unique())
    per_class = max(2, limit // len(classes))
    sample = table.groupby("dx", group_keys=False).head(per_class)
    paths = [by_id[image_id] for image_id in sample.image_id]
    labels = [classes.index(dx) for dx in sample.dx]
    return paths, labels, classes


def evaluate_probe(model, ham_root: Path, limit: int, resolution: int, device) -> dict:
    """Features for a labelled sample, one linear probe, the usual metrics."""
    import numpy as np
    import torch
    from PIL import Image
    from sklearn.model_selection import train_test_split

    from dermssl.metrics import summarize
    from dermssl.probes import run_probe

    paths, labels, classes = _labelled_images(ham_root, limit)
    transform = preprocess(resolution)
    vectors = []
    with torch.no_grad():
        for path in paths:
            tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            vectors.append(model(tensor)[0].cpu().numpy())

    features = np.stack(vectors)
    targets = np.array(labels)
    x_train, x_val, y_train, y_val = train_test_split(
        features, targets, test_size=0.3, random_state=42, stratify=targets
    )
    proba = run_probe("logreg_balanced", x_train, y_train, x_val, len(classes))
    metrics = summarize(y_val, proba, len(classes))
    return {
        "classes": classes,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "feature_dim": int(features.shape[1]),
        "metrics": {name: round(float(value), 4) for name, value in metrics.items()},
    }


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    arch = resolve_arch(args.backbone, args.arch)
    model = load_backbone(args.backbone, arch, device)
    print(f"backbone cargado: {args.backbone} ({arch}) en {device}, congelado")

    if args.image is not None:
        print(json.dumps(describe_one(model, args.image, args.resolution, device), indent=2))
        return
    if args.ham_root is None:
        raise SystemExit("se necesita --image o --ham-root")
    print(json.dumps(
        evaluate_probe(model, args.ham_root, args.limit, args.resolution, device),
        indent=2, ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
