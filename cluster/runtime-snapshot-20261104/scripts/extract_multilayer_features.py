"""Export frozen DINOv2 multilayer features for PAD target-adapt development."""
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

SIDE = 224
BATCH_SIZE = 16
MEAN, STD = (.485, .456, .406), (.229, .224, .225)


def extract_batch(model, batch):
    import torch

    model.eval()
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            blocks = model.get_intermediate_layers(batch, n=4, return_class_token=True)
    if len(blocks) != 4:
        raise ValueError("Encoder must return four intermediate blocks")
    cls_tokens = [cls.float() for _, cls in blocks]
    if any(token.ndim != 2 for token in cls_tokens):
        raise ValueError("Each intermediate block must have a rank-two CLS token")
    cls_features = torch.cat(cls_tokens, dim=1)
    final_patches = blocks[-1][0].float()
    if final_patches.ndim != 3 or final_patches.shape[1] == 0:
        raise ValueError("Final intermediate block must contain patch tokens")
    patch_features = final_patches.mean(dim=1)
    if not torch.isfinite(cls_features).all() or not torch.isfinite(patch_features).all():
        raise ValueError("Encoder produced non-finite multilayer features")
    return cls_features.cpu().numpy(), patch_features.cpu().numpy()


def create_output_directory(output):
    output.mkdir(parents=True, exist_ok=False)


def extract_rows(model, rows):
    import torch
    from torchvision import transforms as T

    transform = T.Compose([
        T.Resize((SIDE, SIDE), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    cls, patches = [], []
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        batch = torch.stack([transform(load_rgb(row["image_path"])) for row in batch_rows]).to("cuda:0")
        cls_batch, patch_batch = extract_batch(model, batch)
        cls.append(cls_batch)
        patches.append(patch_batch)
    return np.concatenate(cls), np.concatenate(patches)


def main():
    protocol, rows, expected = load_inputs()
    output = ROOT / "results/multilayer-features-v1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    create_output_directory(output)
    ids = np.asarray([row["image_id"] for row in rows])
    groups = np.asarray([row["identity_component_id"] for row in rows])
    for name in MODELS:
        weights = Path(PATHS["models"][name])
        if digest(weights) != protocol["encoders"][name]:
            raise ValueError("Frozen encoder checkpoint changed")
        model = load_encoder(weights)
        cls, mean_patch = extract_rows(model, rows)
        np.savez_compressed(output / f"{name}.npz", cls=cls, mean_patch=mean_patch,
                            image_ids=ids, groups=groups)
        print(json.dumps({"event": "saved", "encoder": name, "images": len(ids),
                          "cls_dimensions": cls.shape[1], "patch_dimensions": mean_patch.shape[1]}), flush=True)
        del model
    for row in rows:
        if digest(row["image_path"]) != expected[row["image_id"]]:
            raise ValueError("Clinical image bytes changed during extraction")


if __name__ == "__main__":
    main()
