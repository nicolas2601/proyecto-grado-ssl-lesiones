from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig_c1_macro_f1 as figure  # noqa: E402


BASELINES = {
    "ham10000": ("results/baselines", ""),
    "isic2019": ("results/baselines_isic2019", "_isic2019"),
    "isic2020": ("results/baselines_isic2020", "_isic2020"),
}


def _json(path: str) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def test_ham_c1_uses_canonical_e3b_rows_and_exact_support():
    data = figure.figure_data(REPO)["datasets"]["ham10000"]
    source = _json("resultados/banco_completo/e3b_bs64/e3b_summary.json")
    proposed = source["proposed"]["rows"]["B71_tta"]
    assert data["n_val"] == 2004
    assert len(data["rows"]) == 9
    assert data["rows"][0]["name"] == "B71 (TTA, canónico)"
    assert data["rows"][0]["f1"] == proposed["macro_f1"]
    assert data["rows"][0]["ci"] == tuple(proposed["ci95_grouped"]["macro_f1"])
    assert all(row["n"] == 2004 for row in data["rows"])
    assert data["resampling_unit"] == "lesion_id"


@pytest.mark.parametrize("dataset", ("isic2019", "isic2020"))
def test_isic_c1_uses_b71_original_and_eight_frozen_baselines(dataset):
    data = figure.figure_data(REPO)["datasets"][dataset]
    baseline_dir, suffix = BASELINES[dataset]
    rows = {row["name"]: row for row in data["rows"]}
    b71 = _json(
        f"resultados/banco_completo/isic_eval/limpio_canonico_bs64/{dataset}/B71__original.json"
    )["result"]["metrics"]
    assert rows["B71 original (canónico)"]["f1"] == b71["macro_f1"]
    assert rows["B71 original (canónico)"]["ci"] == tuple(b71["ci95"]["macro_f1"])
    assert "batch 64" in data["protocol"]
    assert "transductivo" in data["protocol"]
    assert "StratifiedKFold" in data["protocol"] if dataset == "isic2019" else "patient_id" in data["protocol"]
    for model, name in figure.bench_rows.MODEL_NAMES.items():
        probe = _json(f"{baseline_dir}/results_{model}{suffix}.json")["probes"]["logreg"]
        assert rows[name]["f1"] == probe["macro_f1"]
        assert rows[name]["ci"] == tuple(probe["ic95_bootstrap"]["macro_f1"])
    assert data["n_val"] == (5067 if dataset == "isic2019" else 6660)


def test_build_all_exports_each_dataset_without_overwriting(tmp_path):
    outputs = figure.build_all(REPO, tmp_path)
    assert set(outputs) == set(BASELINES)
    for dataset, paths in outputs.items():
        assert {path.suffix for path in paths} == {".pdf", ".svg", ".png"}
        assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
        stem = f"fig_c1_f1_macro_{dataset}"
        assert (tmp_path / f"{stem}.caption.txt").is_file()
        trace = json.loads((tmp_path / f"{stem}.traceability.json").read_text(encoding="utf-8"))
        script_sha = hashlib.sha256((REPO / "scripts/figures/fig_c1_macro_f1.py").read_bytes()).hexdigest()
        assert trace["figure_script"] == {"path": "scripts/figures/fig_c1_macro_f1.py", "sha256": script_sha}
    with pytest.raises(FileExistsError):
        figure.build_all(REPO, tmp_path)
