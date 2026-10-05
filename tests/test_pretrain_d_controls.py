"""D2 / D3 launcher builder (docs/33 E2): same recipe, init and seed; only the pool differs."""

from __future__ import annotations

from pathlib import Path

import pytest

from dermssl.pretrain import config as config_mod
from dermssl.pretrain import d_controls as dc
from dermssl.pretrain import v7_finals
from scripts import pretrain as pretrain_cli

KW = {"backbone_weights": Path("/w/A_s42/simsiam_dinov2_backbone_final.pt"),
      "exclude_ids_file": Path("/w/pretrain_exclude_ids_clean.txt")}


def _argv(variant: str, root: str = "/r/root", out: str = "/r/out") -> list[str]:
    return dc.pretrain_argv(variant, image_root=Path(root), output_dir=Path(out), **KW)


def _value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class TestRecipe:
    def test_it_carries_the_registered_recipe(self):
        argv = _argv("D2")
        expected = {"--method": "simsiam_dinov2", "--backbone": "dinov2_vitb14", "--freeze-blocks": "8",
                    "--layer-decay": "0.75", "--hue-jitter": "0.02", "--anchor-l2sp": "0.0001",
                    "--anchor-distill": "0", "--head-warmup-steps": "150", "--max-steps": "2000",
                    "--checkpoint-every-steps": "500", "--seed": "42", "--batch-size-per-gpu": "128",
                    "--num-workers": "0", "--precision": "bf16",
                    "--wandb-project": "dermssl-segmentacion", "--wandb-group": "bench-clean-d",
                    "--initial-weights": str(KW["backbone_weights"]),
                    "--exclude-ids-file": str(KW["exclude_ids_file"])}
        assert {flag: _value(argv, flag) for flag in expected} == expected
        assert argv[argv.index("--checkpoint-fractions") + 1:][:4] == ["0.25", "0.5", "0.75", "1.0"]

    def test_the_shared_flags_agree_with_the_v7_finals_recipe_except_the_step_budget(self):
        shared = {f: v for f, v in v7_finals.COMMON if f not in {"--max-steps", "--wandb-group", "--split-hash",
                                                                 "--data-revision"}}
        argv = _argv("D3")
        assert {f: _value(argv, f) for f in shared} == shared

    def test_it_has_no_tone_term(self):
        assert "--tone-consistency-weight" not in _argv("D2")

    @pytest.mark.parametrize("variant", ["D2", "D3"])
    def test_it_parses_and_validates_with_the_real_cli(self, variant, tmp_path):
        weights = tmp_path / "A_s42.pt"
        weights.write_bytes(b"w")
        argv = dc.pretrain_argv(variant, image_root=Path("/r/root"), output_dir=Path("/r/out"),
                                **{**KW, "backbone_weights": weights})
        cfg = config_mod.validate(pretrain_cli.build_config(pretrain_cli.parse_args(argv)))
        assert cfg.max_steps == 2000 and cfg.anchor_l2sp == 1e-4
        assert config_mod.initial_weights_path(cfg.initial_weights) == weights

    def test_a_missing_backbone_file_is_refused_by_the_real_cli(self):
        with pytest.raises(config_mod.InitialWeightsError):
            config_mod.validate(pretrain_cli.build_config(pretrain_cli.parse_args(_argv("D2"))))

    def test_an_unknown_variant_is_refused(self):
        with pytest.raises(ValueError, match="D4"):
            _argv("D4")


class TestOnlyThePoolDiffers:
    def test_d2_and_d3_differ_only_in_root_and_output(self):
        d2, d3 = _argv("D2", "/r/d2", "/r/D2_s42"), _argv("D3", "/r/d3", "/r/D3_s42")
        assert dc.differing_flags(d2, d3) == {"--image-root", "--output-dir"}
        dc.check_only_pool_differs(d2, d3)

    def test_another_difference_is_refused(self):
        d2, d3 = _argv("D2"), _argv("D3")
        d3[d3.index("--seed") + 1] = "43"
        with pytest.raises(dc.ControlDriftError, match="--seed"):
            dc.check_only_pool_differs(d2, d3)

    def test_a_flag_present_on_one_side_only_is_a_difference(self):
        assert "--resume" in dc.differing_flags(_argv("D2") + ["--resume"], _argv("D3"))


def _pool(tmp_path: Path, n: int = 3) -> Path:
    pool = tmp_path / "originals"
    pool.mkdir()
    for i in range(n):
        (pool / f"ISIC_{i}.jpg").write_bytes(b"x" * (i + 1))
    return pool


def _pad(tmp_path: Path, ids=("PAT_1_1_1", "PAT_2_2_2")) -> dict[str, Path]:
    src = tmp_path / "pad_src"
    src.mkdir(exist_ok=True)
    paths = {}
    for i in ids:
        paths[i] = src / f"{i}.png"
        paths[i].write_bytes(b"png" + i.encode())
    return paths


class TestRoots:
    def test_d2_sees_the_pool_and_d3_the_pool_plus_the_adapt_images(self, tmp_path):
        pool, pad = _pool(tmp_path), _pad(tmp_path)
        roots = dc.build_roots(pool, pad, tmp_path / "work")
        assert len(config_mod.collect_images(roots["D2"])) == 3
        assert len(config_mod.collect_images(roots["D3"])) == 5
        assert {p.stem for p in config_mod.collect_images(roots["D3"])} >= set(pad)

    def test_the_links_resolve_to_the_original_files(self, tmp_path):
        pool, pad = _pool(tmp_path), _pad(tmp_path)
        roots = dc.build_roots(pool, pad, tmp_path / "work")
        link = roots["D3"] / dc.PAD_DIR / "PAT_1_1_1.png"
        assert link.is_symlink() and link.resolve() == pad["PAT_1_1_1"].resolve()

    def test_a_pad_id_that_collides_with_a_pool_image_is_refused(self, tmp_path):
        pool = _pool(tmp_path)
        (pool / "PAT_1_1_1.jpg").write_bytes(b"x")
        with pytest.raises(dc.PoolCollisionError, match="PAT_1_1_1"):
            dc.build_roots(pool, _pad(tmp_path), tmp_path / "work")

    def test_existing_roots_are_not_reused(self, tmp_path):
        pool, pad = _pool(tmp_path), _pad(tmp_path)
        dc.build_roots(pool, pad, tmp_path / "work")
        with pytest.raises(dc.RootExistsError):
            dc.build_roots(pool, pad, tmp_path / "work")

    def test_a_missing_pool_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            dc.build_roots(tmp_path / "nope", _pad(tmp_path), tmp_path / "work")

    def test_an_empty_adapt_list_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="target_adapt"):
            dc.build_roots(_pool(tmp_path), {}, tmp_path / "work")


def test_the_command_line_exports_the_code_revision_and_quotes_paths():
    line = dc.command_line(["--image-root", "/a b/c"])
    assert line.startswith("GIT_COMMIT=$(git rev-parse HEAD) python scripts/pretrain.py")
    assert "'/a b/c'" in line
