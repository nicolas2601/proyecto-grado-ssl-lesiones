"""The published checkpoint has to be loadable by someone who only downloaded it."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import try_model  # noqa: E402

torch = pytest.importorskip("torch")


class _Tiny(torch.nn.Module):
    """Stands in for DINOv2: same contract, no download."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 224 * 224, dim)

    def forward(self, images):
        return self.linear(images.flatten(1))


def _patch_hub(monkeypatch, model):
    monkeypatch.setattr(torch.hub, "load", lambda *a, **k: model)


def test_the_public_backbone_needs_no_checkpoint(monkeypatch):
    seen = {}

    def fake_load(repo, arch, pretrained=False):
        seen["pretrained"] = pretrained
        return _Tiny()

    monkeypatch.setattr(torch.hub, "load", fake_load)
    try_model.load_backbone("public", "dinov2_vits14", torch.device("cpu"))
    assert seen["pretrained"] is True


def test_a_published_checkpoint_is_loaded_and_frozen(monkeypatch, tmp_path):
    reference = _Tiny()
    path = tmp_path / "backbone.pt"
    torch.save(reference.state_dict(), path)
    _patch_hub(monkeypatch, _Tiny())

    model = try_model.load_backbone(str(path), "dinov2_vits14", torch.device("cpu"))

    assert sum(p.requires_grad for p in model.parameters()) == 0
    assert not model.training


def test_a_checkpoint_from_another_architecture_is_rejected(monkeypatch, tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"algo.que.no.existe": torch.zeros(2)}, path)
    _patch_hub(monkeypatch, _Tiny())

    with pytest.raises(SystemExit, match="no encaja"):
        try_model.load_backbone(str(path), "dinov2_vits14", torch.device("cpu"))


def test_one_image_reports_its_feature_vector(monkeypatch, tmp_path):
    pytest.importorskip("torchvision")
    from PIL import Image

    image = tmp_path / "lesion.png"
    Image.new("RGB", (300, 300), (180, 130, 100)).save(image)
    _patch_hub(monkeypatch, _Tiny())
    model = try_model.load_backbone("public", "dinov2_vits14", torch.device("cpu"))

    described = try_model.describe_one(model, image, 224, torch.device("cpu"))

    assert described["dim"] == 8
    assert len(described["first_values"]) == 8
    assert described["image"].endswith("lesion.png")


def test_the_preprocessing_matches_the_bank(monkeypatch):
    pytest.importorskip("torchvision")
    from PIL import Image

    tensor = try_model.preprocess(224)(Image.new("RGB", (400, 400), (10, 20, 30)))
    assert tuple(tensor.shape) == (3, 224, 224)


def test_parse_args_requires_a_backbone():
    args = try_model.parse_args(["--backbone", "public", "--image", "/x.png"])
    assert args.backbone == "public"
    # Unset means "deduce it from the manifest next to the weights", not ViT-S:
    # more than one architecture is published. See tests/test_try_model_arch.py.
    assert args.arch is None
    with pytest.raises(SystemExit):
        try_model.parse_args([])
