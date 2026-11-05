"""Development ablation: frozen-encoder CLS features under alternative preprocessing (no training)."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATHS = json.loads((ROOT / "config/paths.json").read_text())
FROZEN = Path(PATHS["frozen_experiment_root"])
sys.path[:0] = [str(ROOT / "scripts"), str(FROZEN / "scripts"), str(FROZEN / "code/src")]

from evaluate_clinical_reference import MODELS, load_inputs  # noqa: E402
from evaluate_final_refit import load_encoder  # noqa: E402
from dermssl.pretrain.image_input import load_rgb  # noqa: E402
from phone_contract import digest  # noqa: E402

MEAN, STD = (.485, .456, .406), (.229, .224, .225)
# name -> (square side, hflip TTA); side is a multiple of the 14-px patch, aspect is squashed so no lesion is cropped
VARIANTS = {"squash224": (224, False), "squash336": (336, False), "squash448": (448, False), "squash336_tta": (336, True)}


def extract(model, rows, side, tta):
    import torch
    from torchvision import transforms as T
    transform = T.Compose([T.Resize((side, side), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(MEAN, STD)])
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            batch = torch.stack([transform(load_rgb(r["image_path"])) for r in rows[start:start + 16]]).to("cuda:0")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feats = model.forward_features(batch)["x_norm_clstoken"].float()
                if tta:
                    feats = (feats + model.forward_features(batch.flip(-1))["x_norm_clstoken"].float()) / 2
            out.append(feats.cpu().numpy())
    return np.concatenate(out)


def main():
    protocol, rows, expected = load_inputs()
    output = ROOT / "results/preprocessing-variants-v1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output.mkdir(exist_ok=False)
    ids = np.asarray([r["image_id"] for r in rows])
    for name in MODELS:
        weights = Path(PATHS["models"][name])
        if digest(weights) != protocol["encoders"][name]:
            raise ValueError("Frozen encoder checkpoint changed")
        model = load_encoder(weights)
        for variant, (side, tta) in VARIANTS.items():
            np.savez_compressed(output / f"{name}_{variant}.npz", features=extract(model, rows, side, tta), image_ids=ids)
            print(json.dumps({"event": "saved", "encoder": name, "variant": variant}), flush=True)
        del model
    for r in rows:
        if digest(r["image_path"]) != expected[r["image_id"]]:
            raise ValueError("Clinical image bytes changed during extraction")


if __name__ == "__main__":
    main()
