"""Frozen OpenCLIP image features (squash 224, OpenAI CLIP normalisation) for approved PAD/HAM development manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
APPROVED = {"pad_adapt.jsonl", "ham_dev.jsonl"}
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="ViT-B-16")
    parser.add_argument("--weights-file", default="open_clip_model.safetensors")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing frozen features")
    overlay = json.loads(args.receipt.read_text())
    if (args.manifest.name not in APPROVED or args.manifest.parent != args.receipt.parent
            or overlay["output_sha256"][args.manifest.name] != args.manifest_sha
            or digest(args.receipt) != args.receipt_sha or digest(args.manifest) != args.manifest_sha):
        raise ValueError("Only approved PAD/HAM development manifests with matching receipt may be extracted")
    dl = json.loads((args.weights_dir / "download-receipt.json").read_text())
    if dl["status"] != "COMPLETE" or dl["gated"] is not False or dl["license"] != "MIT" or dl["remote_code_downloaded"] is not False:
        raise ValueError("Approved public weights receipt required")
    for name, rec in dl["files"].items():
        if digest(args.weights_dir / name) != rec["sha256"]:
            raise ValueError("Downloaded weights changed")
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(args.arch, pretrained=str(args.weights_dir / args.weights_file))
    model = model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    prep = T.Compose([T.Resize((224, 224), antialias=True), T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    feats = []
    with torch.no_grad():
        for start in range(0, len(rows), 64):
            batch = torch.stack([prep(Image.open(r["image_path"]).convert("RGB")) for r in rows[start:start + 64]])
            feats.append(model.encode_image(batch.to(next(model.parameters()).device)).float().cpu().numpy())
    features = np.concatenate(feats)
    if features.shape[0] != len(rows) or not np.isfinite(features).all():
        raise ValueError("Invalid OpenCLIP feature matrix")
    np.savez_compressed(args.output, features=features, image_ids=np.asarray([r["image_id"] for r in rows]),
                        groups=np.asarray([r["identity_component_id"] for r in rows]))
    print(json.dumps({"images": len(rows), "dim": int(features.shape[1])}), flush=True)


if __name__ == "__main__":
    main()
