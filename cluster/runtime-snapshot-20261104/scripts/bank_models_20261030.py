"""Baseline bank on the current split: model registry, preprocessing and backbone builders.

All weights are public and already local (HF cache or torch hub cache); nothing is downloaded here.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

IMAGE_SIZE = 224
HUB = Path.home() / ".cache/torch/hub/checkpoints"
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
SUPERVISED, SSL_PUBLIC = "supervisado", "SSL puro"

MODELS = {
    "alexnet": dict(kind="alexnet", lr=1e-4, label="AlexNet (supervisado ImageNet)", section=SUPERVISED,
                    weights=HUB / "alexnet-owt-7be5be79.pth", mean_std=IMAGENET, interp="bilinear"),
    "vgg19": dict(kind="timm", name="vgg19.tv_in1k", lr=1e-4, label="VGG19 (supervisado ImageNet)",
                  section=SUPERVISED, mean_std=IMAGENET, interp="bilinear"),
    "efficientnet-b0": dict(kind="timm", name="efficientnet_b0.ra_in1k", lr=1e-4,
                            label="EfficientNet-B0 (supervisado ImageNet)", section=SUPERVISED,
                            mean_std=IMAGENET, interp="bicubic"),
    "resnet50": dict(kind="timm", name="resnet50.tv2_in1k", lr=1e-4, label="ResNet-50 (supervisado ImageNet)",
                     section=SUPERVISED, mean_std=IMAGENET, interp="bilinear"),
    "vit-b16": dict(kind="timm", name="vit_base_patch16_224.augreg_in1k", lr=3e-5,
                    label="ViT-B/16 (supervisado ImageNet)", section=SUPERVISED,
                    mean_std=((0.5,) * 3, (0.5,) * 3), interp="bicubic"),
    "simsiam-r50": dict(kind="simsiam_r50", lr=1e-4, label="SimSiam ResNet-50 (SSL ImageNet, 100 épocas)",
                        section=SSL_PUBLIC, weights=HUB / "simsiam_r50_100ep.pth.tar", mean_std=IMAGENET,
                        interp="bilinear"),
    "dinov2-vits14": dict(kind="timm", name="vit_small_patch14_dinov2.lvd142m", lr=3e-5, img_size=IMAGE_SIZE,
                          label="DINOv2 ViT-S/14 (SSL público)", section=SSL_PUBLIC, mean_std=IMAGENET,
                          interp="bicubic"),
    "dinov2-vitb14": dict(kind="timm", name="vit_base_patch14_dinov2.lvd142m", lr=3e-5, img_size=IMAGE_SIZE,
                          label="DINOv2 ViT-B/14 (SSL público)", section=SSL_PUBLIC, mean_std=IMAGENET,
                          interp="bicubic"),
}
SIMSIAM_PREFIX = "module.encoder."


def spec(model_id):
    if model_id not in MODELS:
        raise ValueError(f"Unknown bank model: {model_id}")
    return MODELS[model_id]


def simsiam_encoder_state(state_dict):
    """Keep only the backbone tensors of a FAIR SimSiam checkpoint, without the projector head."""
    kept = {key[len(SIMSIAM_PREFIX):]: value for key, value in state_dict.items()
            if key.startswith(SIMSIAM_PREFIX) and not key[len(SIMSIAM_PREFIX):].startswith("fc.")}
    if not kept:
        raise ValueError("No SimSiam encoder tensors found")
    return kept


class Gap(nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3))


class FeatureMap(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_features(x)


def build_backbone(model_id, num_classes=0):
    """num_classes == 0 -> module returning pooled features; > 0 -> module returning logits."""
    info = spec(model_id)
    if info["kind"] == "timm":
        import timm
        extra = {"img_size": info["img_size"]} if "img_size" in info else {}
        if model_id == "vgg19" and num_classes == 0:
            model = timm.create_model(info["name"], pretrained=True, num_classes=1000)
            return nn.Sequential(FeatureMap(model), Gap())
        return timm.create_model(info["name"], pretrained=True, num_classes=num_classes, **extra)
    import torchvision
    if info["kind"] == "alexnet":
        model = torchvision.models.alexnet()
        model.load_state_dict(torch.load(info["weights"], map_location="cpu", weights_only=True))
        if num_classes == 0:
            return nn.Sequential(model.features, Gap())
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model
    model = torchvision.models.resnet50()
    checkpoint = torch.load(info["weights"], map_location="cpu", weights_only=False)
    model.load_state_dict(simsiam_encoder_state(checkpoint["state_dict"]), strict=False)
    model.fc = nn.Identity() if num_classes == 0 else nn.Linear(2048, num_classes)
    return model


def load_images(rows, interp):
    """uint8 tensor (N, 3, 224, 224): direct resize, no crop, same size for every model."""
    from PIL import Image
    mode = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC}[interp]
    out = np.empty((len(rows), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for index, row in enumerate(rows):
        with Image.open(row["image_path"]) as image:
            out[index] = np.asarray(image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), mode))
    return torch.from_numpy(out).permute(0, 3, 1, 2).contiguous()


def normalize(batch_uint8, model_id):
    mean, std = (torch.tensor(v, device=batch_uint8.device).view(1, 3, 1, 1) for v in spec(model_id)["mean_std"])
    return (batch_uint8.float() / 255.0 - mean) / std
