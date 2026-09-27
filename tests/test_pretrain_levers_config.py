"""docs/28 levers at the config and CLI boundary, without torch.

The contract is two-sided: every lever defaults to the v6 value, so an old
command line builds the same config it always did; and a lever set where it
cannot act (wrong method, no tone pair) is refused before any GPU time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dermssl.pretrain import config as cfg_mod
from dermssl.pretrain import run_contract
from dermssl.pretrain.levers import Levers

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pretrain as launcher  # noqa: E402

BASE = ["--method", "simsiam_dinov2", "--image-root", "/i", "--output-dir", "/o"]


def make_config(**overrides):
    base = dict(method="simsiam_dinov2", image_root="/i", output_dir="/o", split_hash="a" * 64)
    base.update(overrides)
    return cfg_mod.PretrainConfig(**base)


class TestDefaultsReproduceV6:
    def test_an_old_command_line_builds_the_v6_lever_values(self):
        cfg = launcher.build_config(launcher.parse_args(BASE))
        assert cfg_mod.levers_of(cfg).is_default()
        assert cfg.hue_jitter == 0.1 and cfg.exclude_ids_file == ""
        assert cfg.consistency_weight == 1.0

    def test_the_run_manifest_records_every_hyperparameter(self):
        manifest = run_contract.build_run_manifest(make_config(max_steps=1), gpu_count=2)
        hp = manifest["hyperparameters"]
        assert hp["effective_lr"] == pytest.approx(5e-4 * 128 / 1024)
        assert hp["batch_size_per_gpu"] == 64 and hp["weight_decay"] == 1e-4
        assert hp["hue_jitter"] == 0.1 and hp["grad_clip_norm"] == cfg_mod.GRAD_CLIP_NORM
        assert manifest["adaptation_levers"] == {
            "freeze_blocks": 0, "layer_decay": 1.0, "anchor_l2sp": 0.0,
            "anchor_distill": 0.0, "tone_consistency_weight": 0.0,
        }
        assert manifest["exclude_ids_sha256"] is None


class TestFlagsReachTheConfig:
    def test_every_lever_flag_is_wired(self):
        args = launcher.parse_args(BASE + [
            "--freeze-blocks", "8", "--layer-decay", "0.75", "--anchor-l2sp", "1e-3",
            "--anchor-distill", "0.1", "--hue-jitter", "0.02",
        ])
        cfg = launcher.build_config(args)
        assert cfg_mod.levers_of(cfg) == Levers(
            freeze_blocks=8, layer_decay=0.75, anchor_l2sp=1e-3, anchor_distill=0.1
        )
        assert cfg.hue_jitter == 0.02

    def test_tone_consistency_without_the_tone_pair_is_refused(self):
        cfg = make_config(tone_consistency_weight=0.5)
        with pytest.raises(ValueError, match="original_transformed"):
            cfg_mod.validate(cfg)


class TestLeverScope:
    def test_levers_are_refused_outside_simsiam_dinov2(self):
        with pytest.raises(ValueError, match="simsiam_dinov2"):
            cfg_mod.validate(make_config(method="dino_continual", layer_decay=0.75))

    def test_the_default_levers_are_fine_on_any_method(self):
        cfg_mod.validate(make_config(method="dino_continual"))

    @pytest.mark.parametrize("hue", [-0.01, 0.51])
    def test_hue_outside_colorjitter_range_is_refused(self, hue):
        with pytest.raises(ValueError, match="hue_jitter"):
            cfg_mod.validate(make_config(hue_jitter=hue))


class TestExcludedIds:
    def test_json_and_line_formats_load_the_same_ids(self, tmp_path):
        as_json = tmp_path / "ids.json"
        as_json.write_text(json.dumps(["ISIC_1", "ISIC_2"]))
        as_lines = tmp_path / "ids.txt"
        as_lines.write_text("ISIC_1\nISIC_2\n")
        assert cfg_mod.load_excluded_ids(as_json) == cfg_mod.load_excluded_ids(as_lines) == {"ISIC_1", "ISIC_2"}

    def test_an_empty_file_is_refused(self, tmp_path):
        empty = tmp_path / "ids.txt"
        empty.write_text("\n")
        with pytest.raises(ValueError):
            cfg_mod.load_excluded_ids(empty)

    def test_the_manifest_hashes_the_exclusion_file(self, tmp_path):
        ids = tmp_path / "val_ids.txt"
        ids.write_text("ISIC_1\n")
        manifest = run_contract.build_run_manifest(
            make_config(max_steps=1, exclude_ids_file=str(ids)), gpu_count=0
        )
        assert manifest["exclude_ids_file"] == str(ids)
        assert len(manifest["exclude_ids_sha256"]) == 64


class TestExclusionCannotBeSilent:
    """Review P1-1: an exclusion that matches nothing used to train as if it worked."""

    def _pool(self, tmp_path, names_and_bytes):
        paths = []
        for name, payload in names_and_bytes:
            path = tmp_path / name
            path.write_bytes(payload)
            paths.append(path)
        return paths

    def test_every_requested_id_must_be_found(self, tmp_path):
        pool = self._pool(tmp_path, [("ISIC_1.jpg", b"a"), ("ISIC_2.jpg", b"b")])
        with pytest.raises(ValueError, match="no aparecen"):
            cfg_mod.apply_exclusion(pool, frozenset({"ISIC_1", "ISIC_9"}))

    def test_a_byte_identical_copy_under_another_id_is_also_dropped(self, tmp_path):
        pool = self._pool(tmp_path, [
            ("ISIC_1.jpg", b"same-image"),
            ("ISIC_77_downsampled.jpg", b"same-image"),
            ("ISIC_2.jpg", b"other"),
        ])
        kept, report = cfg_mod.apply_exclusion(pool, frozenset({"ISIC_1"}))
        assert [p.name for p in kept] == ["ISIC_2.jpg"]
        assert report == {"n_ids": 1, "n_removed_by_id": 1, "n_removed_by_content": 1}

    def test_pairs_are_matched_on_the_original_image(self, tmp_path):
        originals = self._pool(tmp_path, [("ISIC_1.jpg", b"x"), ("ISIC_2.jpg", b"y")])
        pairs = [(o, tmp_path / f"t_{o.name}") for o in originals]
        kept, _ = cfg_mod.apply_exclusion(pairs, frozenset({"ISIC_2"}))
        assert kept == [pairs[0]]


def test_hue_jitter_on_the_dino_arm_is_refused():
    """Review P2: it was validated and recorded but never reached the DINO dataset."""
    with pytest.raises(ValueError, match="hue_jitter"):
        cfg_mod.validate(make_config(method="dino_continual", hue_jitter=0.02))
