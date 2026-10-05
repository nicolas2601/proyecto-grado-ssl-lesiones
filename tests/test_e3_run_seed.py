"""``--b71-seed``: the pinned F4-vs-B71 comparison on the seed 43 and 44 weights (post-hoc robustness)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import e3_run  # noqa: E402
from test_e3_run import cli, runtime, selected  # noqa: E402

from dermssl import e3_seed as es  # noqa: E402


def seeded_runtime(seed: int, **override):
    """The synthetic runtime with the fingerprint a real build would give for ``seed``."""
    rt = runtime()
    rt.fingerprint = {**rt.fingerprint, "protocol": es.protocol(seed), "weights_sha256": f"{seed}" * 32,
                      **({} if seed == es.BASE_SEED else {"b71_seed": seed}), **override}
    return rt


def pin_of(tmp_path):
    path, sha = selected(tmp_path / "e3")
    return ["--selection", str(path), "--selection-sha256", sha]


def seed_cli(out, seed, stage, *extra):
    return cli(out, stage, "--b71-seed", str(seed), *extra, rt=seeded_runtime(seed))


def test_the_default_seed_is_the_registered_one_and_only_registered_seeds_parse():
    assert e3_run.parse_args(["--stage", "extract", "--out-dir", "x"]).b71_seed == 42
    assert e3_run.parse_args(["--stage", "final", "--out-dir", "x", "--b71-seed", "44"]).b71_seed == 44
    with pytest.raises(SystemExit):
        e3_run.parse_args(["--stage", "final", "--out-dir", "x", "--b71-seed", "45"])


@pytest.mark.parametrize("seed", [43, 44])
def test_the_pinned_selection_runs_on_another_seed_in_its_own_out_dir(tmp_path, seed):
    pin = pin_of(tmp_path)
    out = tmp_path / f"rob_s{seed}"
    assert seed_cli(out, seed, "extract", "--splits", "train", "val", *pin) == 0
    assert seed_cli(out, seed, "final", *pin, "--ci-resamples", "20") == 0
    doc = json.loads((out / "e3_final_eval.json").read_text())
    assert doc["fingerprint"]["b71_seed"] == seed and doc["fingerprint"]["protocol"] == es.protocol(seed)
    assert doc["selection_sha256"] == pin[-1] and (out / "val_read.lock").is_file()
    assert not (tmp_path / "e3" / "val_read.lock").exists()  # the registered out-dir is untouched


def test_cells_of_different_seeds_are_never_mixed(tmp_path, capsys):
    pin = pin_of(tmp_path)
    out = tmp_path / "rob"
    assert seed_cli(out, 43, "extract", "--splits", "val", *pin) == 0
    capsys.readouterr()
    assert seed_cli(out, 43, "extract", "--splits", "val", *pin) == 0
    assert "skipped" in capsys.readouterr().out
    assert seed_cli(out, 44, "extract", "--splits", "val", *pin) == 0
    assert "done" in capsys.readouterr().out
    assert list((out / "cells").glob("extract__val.json.stale-*"))
    assert json.loads((out / "cells" / "extract__val.json").read_text())["key"]["b71_seed"] == 44


def test_a_runtime_for_another_seed_than_the_flag_is_refused(tmp_path):
    pin = pin_of(tmp_path)
    out = tmp_path / "rob"
    assert cli(out, "extract", "--splits", "val", "--b71-seed", "43", *pin, rt=seeded_runtime(44)) == 2
    assert not (out / "features").exists()


def test_select_is_never_rerun_on_another_seed(tmp_path):
    assert seed_cli(tmp_path / "rob", 43, "extract", "--splits", "fit", "holdout") == 2
    assert seed_cli(tmp_path / "rob", 43, "select", "--ci-resamples", "5") == 2
    assert not (tmp_path / "rob" / "e3_selection.json").exists()


def test_another_seed_may_change_only_its_own_weights_not_the_masks(tmp_path, capsys):
    pin = pin_of(tmp_path)
    out = tmp_path / "rob"
    assert seed_cli(out, 43, "extract", "--splits", "train", "val", *pin) == 0
    rt = seeded_runtime(43, masks_sha256="z" * 64)
    assert cli(out, "final", "--b71-seed", "43", *pin, rt=rt) == 2
    assert "masks_sha256" in capsys.readouterr().out and not (out / "val_read.lock").exists()


def test_the_base_seed_still_refuses_other_weights_than_the_selected_ones(tmp_path, capsys):
    pin = pin_of(tmp_path)
    out = tmp_path / "other"
    assert cli(out, "extract", "--splits", "train", "val", *pin) == 0
    rt = runtime()
    rt.fingerprint = {**rt.fingerprint, "weights_sha256": "q" * 64}
    assert cli(out, "final", *pin, rt=rt) == 2
    assert "weights_sha256" in capsys.readouterr().out


def test_missing_weights_abort_with_the_typed_error_before_the_val_lock(tmp_path, capsys, monkeypatch):
    import e3_io

    def no_weights(args):
        raise es.MissingWeightsError("sin pesos de B* semilla 44 en /x")

    monkeypatch.setattr(e3_io, "build_runtime", no_weights)
    pin = pin_of(tmp_path)
    out = tmp_path / "rob"
    assert e3_run.main(["--stage", "final", "--out-dir", str(out), "--b71-seed", "44", *pin], revision="rev0") == 2
    assert "sin pesos" in capsys.readouterr().out and not (out / "val_read.lock").exists()
