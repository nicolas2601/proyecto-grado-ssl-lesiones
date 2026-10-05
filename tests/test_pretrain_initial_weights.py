"""--initial-weights loads a backbone file, anchors L2-SP to it, and says so in run.json."""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch import nn

from dermssl.pretrain import config as config_mod
from dermssl.pretrain import run_contract, torch_loop, torch_simsiam


def _cfg(tmp_path, **over):
    base = {"method": "simsiam_dinov2", "image_root": str(tmp_path), "output_dir": str(tmp_path / "out"),
            "split_hash": "a" * 64, "max_steps": 10, "anchor_l2sp": 1e-4}
    return config_mod.PretrainConfig(**{**base, **over})


def _weights(tmp_path, value: float = 3.0):
    path = tmp_path / "backbone.pt"
    torch.save(nn.Linear(2, 2).state_dict() | {"weight": torch.full((2, 2), value), "bias": torch.zeros(2)}, path)
    return path


class TestResolution:
    @pytest.mark.parametrize("label", ["", "random", "imagenet1k_v2", "dinov2-public-lvd142m"])
    def test_a_label_is_not_a_path(self, label):
        assert config_mod.initial_weights_path(label) is None

    def test_an_existing_file_is_resolved(self, tmp_path):
        path = _weights(tmp_path)
        assert config_mod.initial_weights_path(str(path)) == path

    @pytest.mark.parametrize("value", ["/no/such/backbone.pt", "relative/dir/w.pth", "w.pt"])
    def test_a_missing_path_is_a_typed_error_with_context(self, value):
        with pytest.raises(config_mod.InitialWeightsError, match="initial_weights"):
            config_mod.initial_weights_path(value)


class TestValidate:
    def test_a_missing_file_fails_before_any_gpu(self, tmp_path):
        with pytest.raises(config_mod.InitialWeightsError):
            config_mod.validate(_cfg(tmp_path, initial_weights=str(tmp_path / "gone.pt")))

    def test_a_file_is_accepted_for_simsiam_dinov2(self, tmp_path):
        cfg = config_mod.validate(_cfg(tmp_path, initial_weights=str(_weights(tmp_path))))
        assert cfg.initial_weights.endswith("backbone.pt")

    def test_other_methods_refuse_a_file_instead_of_ignoring_it(self, tmp_path):
        with pytest.raises(ValueError, match="solo aplica a simsiam_dinov2"):
            config_mod.validate(_cfg(tmp_path, method="dino_continual", anchor_l2sp=0.0,
                                     initial_weights=str(_weights(tmp_path))))


class TestRunManifest:
    def test_a_loaded_file_is_recorded_with_its_sha256(self, tmp_path):
        path = _weights(tmp_path)
        manifest = run_contract.build_run_manifest(_cfg(tmp_path, initial_weights=str(path)), gpu_count=1)
        assert manifest["initial_weights_loaded"] is True
        assert manifest["initial_weights_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_the_default_manifest_is_unchanged(self, tmp_path):
        manifest = run_contract.build_run_manifest(_cfg(tmp_path), gpu_count=1)
        assert manifest["initial_weights"] == "dinov2-public-lvd142m"
        assert "initial_weights_loaded" not in manifest and "initial_weights_sha256" not in manifest


class TestLoader:
    def test_it_loads_strictly(self, tmp_path):
        backbone = nn.Linear(2, 2)
        torch_simsiam.load_backbone_weights(backbone, _weights(tmp_path, 3.0))
        assert torch.equal(backbone.weight, torch.full((2, 2), 3.0))

    def test_a_mismatched_state_is_refused(self, tmp_path):
        with pytest.raises(RuntimeError):
            torch_simsiam.load_backbone_weights(nn.Linear(3, 3), _weights(tmp_path))


class TestBuildOrder:
    """The L2-SP anchor must be a copy of the LOADED weights, not of the public ones."""

    @pytest.fixture
    def fake_simsiam(self, monkeypatch):
        class Fake(nn.Module):
            def __init__(self, name, pretrained=False):
                super().__init__()
                self.backbone = nn.Linear(2, 2)
                nn.init.zeros_(self.backbone.weight)  # stands for the public weights

        monkeypatch.setattr(torch_loop.torch_simsiam, "SimSiam", Fake)

    def _build(self, tmp_path, **over):
        from PIL import Image

        Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
        cfg = config_mod.validate(_cfg(tmp_path, **over))
        return torch_loop._build(cfg, torch.device("cpu"))

    def test_the_anchor_holds_the_loaded_weights(self, tmp_path, fake_simsiam):
        model, *_, extras = self._build(tmp_path, initial_weights=str(_weights(tmp_path, 3.0)))
        assert torch.equal(model.backbone.weight, torch.full((2, 2), 3.0))
        assert torch.equal(extras["anchors"].initial["weight"], torch.full((2, 2), 3.0))

    def test_with_the_default_label_nothing_is_loaded(self, tmp_path, fake_simsiam):
        model, *_, extras = self._build(tmp_path)
        assert torch.equal(model.backbone.weight, torch.zeros(2, 2))
        assert torch.equal(extras["anchors"].initial["weight"], torch.zeros(2, 2))


class TestNeverClaimAnInitWithoutLoadingIt:
    """Paula's invariant: initial_weights=<file> without a verified load must not run or be written."""

    def test_run_json_with_a_path_and_no_loaded_flag_cannot_be_written(self, tmp_path):
        manifest = {"initial_weights": str(_weights(tmp_path))}
        with pytest.raises(config_mod.InitialWeightsNotLoadedError, match="initial_weights_loaded"):
            run_contract.write_run_manifest(manifest, tmp_path / "run.json")
        assert not (tmp_path / "run.json").exists()

    def test_run_json_with_a_path_and_loaded_false_cannot_be_written(self, tmp_path):
        manifest = {"initial_weights": "/x/w.pt", "initial_weights_loaded": False, "initial_weights_sha256": "a" * 64}
        with pytest.raises(config_mod.InitialWeightsNotLoadedError):
            run_contract.write_run_manifest(manifest, tmp_path / "run.json")

    def test_run_json_with_a_path_and_no_sha256_cannot_be_written(self, tmp_path):
        with pytest.raises(config_mod.InitialWeightsNotLoadedError, match="sha256"):
            run_contract.write_run_manifest({"initial_weights": "/x/w.pt", "initial_weights_loaded": True},
                                            tmp_path / "run.json")

    def test_a_built_manifest_with_a_file_is_writable(self, tmp_path):
        manifest = run_contract.build_run_manifest(_cfg(tmp_path, initial_weights=str(_weights(tmp_path))), gpu_count=1)
        run_contract.write_run_manifest(manifest, tmp_path / "run.json")
        assert (tmp_path / "run.json").is_file()

    def test_a_label_manifest_is_writable_without_the_flag(self, tmp_path):
        run_contract.write_run_manifest({"initial_weights": "dinov2-public-lvd142m"}, tmp_path / "run.json")

    def test_build_refuses_when_the_loader_did_not_load(self, tmp_path, monkeypatch):
        class Fake(nn.Module):
            def __init__(self, name, pretrained=False):
                super().__init__()
                self.backbone = nn.Linear(2, 2)

        from PIL import Image

        Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
        monkeypatch.setattr(torch_loop.torch_simsiam, "SimSiam", Fake)
        monkeypatch.setattr(torch_loop.torch_simsiam, "load_backbone_weights", lambda backbone, path: None)
        cfg = config_mod.validate(_cfg(tmp_path, initial_weights=str(_weights(tmp_path, 3.0))))
        with pytest.raises(config_mod.InitialWeightsNotLoadedError, match="no coincide"):
            torch_loop._build(cfg, torch.device("cpu"))

    def test_regression_pretrained_true_builds_public_weights_but_the_file_wins(self, tmp_path, monkeypatch):
        """The original bug: pretrained=True gave public weights, the anchor copied them, the file was ignored."""
        seen = {}

        class Public(nn.Module):
            def __init__(self, name, pretrained=False):
                super().__init__()
                seen["pretrained"] = pretrained
                self.backbone = nn.Linear(2, 2)
                nn.init.constant_(self.backbone.weight, 7.0)  # what torch.hub would return

        from PIL import Image

        Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
        monkeypatch.setattr(torch_loop.torch_simsiam, "SimSiam", Public)
        cfg = config_mod.validate(_cfg(tmp_path, initial_weights=str(_weights(tmp_path, 3.0))))
        model, *_, extras = torch_loop._build(cfg, torch.device("cpu"))
        assert seen["pretrained"] is True
        assert torch.equal(model.backbone.weight, torch.full((2, 2), 3.0))
        assert torch.equal(extras["anchors"].initial["weight"], torch.full((2, 2), 3.0))
