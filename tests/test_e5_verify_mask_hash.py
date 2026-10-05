"""The pod diagnostic names the hash definition that reproduces the manifest, on masks from the real exporter."""

import json
import sys
from pathlib import Path

import seg_paths  # noqa: F401
import export_tone_inputs as exporter
from test_seg_export_tone_inputs import _build_cache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import e5_verify_mask_hash as vh  # noqa: E402


def _pool(tmp_path):
    cache_dir, masks_path, _ = _build_cache(tmp_path)
    out = tmp_path / "pool"
    exporter.run(exporter.parse_args(["--cache-dir", str(cache_dir), "--pseudo-masks", str(masks_path),
                                      "--output-root", str(out), "--source-revision", "r"]))
    return out, json.loads((out / "mask_manifest.json").read_text())["masks"]


def test_only_the_manifest_definition_reproduces_the_hash_of_a_real_exported_mask(tmp_path):
    out, entries = _pool(tmp_path)
    result = vh.check_entry(out, next(e for e in entries if e["eligible"]))
    assert result["manifest_def (bool + shape)"] is True and result["png_file_bytes"] is False
    assert [k for k, ok in result.items() if ok] == ["manifest_def (bool + shape)"]


def test_the_sample_is_deterministic_eligible_only_and_capped_per_dataset(tmp_path):
    _, entries = _pool(tmp_path)
    got = vh.sample(entries, 1)
    assert all(e["eligible"] for e in got) and got == vh.sample(entries, 1)
    assert len({e["dataset"] for e in got}) == len(got)


def test_main_prints_a_line_per_dataset(tmp_path, capsys):
    out, _ = _pool(tmp_path)
    assert vh.main(["--pool-root", str(out), "--n", "5"]) == 0
    assert "manifest_def (bool + shape)': '" in capsys.readouterr().out
