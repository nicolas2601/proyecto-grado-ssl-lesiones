"""Figure 2 (model comparison): its numbers equal the committed result files, read independently of the bench."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "figures"))

import fig2_models as fig  # noqa: E402

ISIC_CONTRASTS = "resultados/banco_completo/isic_eval/limpio_canonico_bs64/{ds}/contrasts/{arm}__clean__vs__PUBLIC_{arm}.json"
HAM_CLEAN = "resultados/banco_completo/ham_clean/ham_clean_val.json"
BASELINES = {"ham10000": ("results/baselines", ""), "isic2019": ("results/baselines_isic2019", "_isic2019"),
             "isic2020": ("results/baselines_isic2020", "_isic2020")}


def _json(rel: str) -> dict:
    return json.loads((REPO / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data() -> dict:
    return fig.figure_data(REPO)


def test_sources_are_the_bench_loaders(data):
    assert data["sources"] == ["dermssl.bench.rows", "dermssl.bench.contrasts_doc"]
    assert fig.R.dataset_rows.__module__ == "dermssl.bench.rows"
    assert fig.contrasts_doc.build.__module__ == "dermssl.bench.contrasts_doc"


def test_row_order_is_arms_then_public_controls_then_baselines_by_mean_macro_f1(data):
    order = data["order"]
    assert order[:4] == ["A71-limpio", "B71-limpio", "DINOv2 público · prot. A71", "DINOv2 público · prot. B71"]
    names = dict(zip(fig.R.MODELS, fig.R.MODEL_NAMES.values()))
    mean = {}
    for model, name in names.items():
        vals = [_json(f"{d}/results_{model}{s}.json")["probes"]["logreg"]["macro_f1"] for d, s in BASELINES.values()]
        mean[name] = sum(vals) / 3
    assert order[4:] == sorted(mean, key=lambda n: -mean[n])


@pytest.mark.parametrize("ds", list(BASELINES))
def test_baseline_macro_f1_and_ci_match_their_json(data, ds):
    directory, suffix = BASELINES[ds]
    for model, name in fig.R.MODEL_NAMES.items():
        probe = _json(f"{directory}/results_{model}{suffix}.json")["probes"]["logreg"]
        point, low, high = data["datasets"][ds]["macro_f1"][name]
        assert point == probe["macro_f1"] and (low, high) == tuple(probe["ic95_bootstrap"]["macro_f1"])


@pytest.mark.parametrize("ds", ["isic2019", "isic2020"])
@pytest.mark.parametrize("arm", ["A71", "B71"])
def test_isic_deltas_and_points_match_the_contrast_json(data, ds, arm):
    paired = {p["metric"]: p for p in _json(ISIC_CONTRASTS.format(ds=ds, arm=arm))["result"]["paired"]}
    entry = data["datasets"][ds]
    for metric in ("macro_f1", "mcc"):
        assert entry["deltas"][(arm, metric)] == (paired[metric]["point_delta"], paired[metric]["ci95_low"],
                                                  paired[metric]["ci95_high"])
        assert entry[metric][f"{arm}-limpio"][0] == paired[metric]["point_arm"]
        assert entry[metric][f"DINOv2 público · prot. {arm}"][0] == paired[metric]["point_control"]


@pytest.mark.parametrize("arm", ["A71", "B71"])
def test_ham_deltas_match_the_ham_clean_json(data, arm):
    paired = {(p["arm"], p["control"], p["metric"]): p for p in _json(HAM_CLEAN)["paired"]}
    for metric in ("macro_f1", "mcc"):
        p = paired[(f"{arm}_clean", f"public_{arm}", metric)]
        assert data["datasets"]["ham10000"]["deltas"][(arm, metric)] == (p["point_delta"], p["ci95_low"], p["ci95_high"])
    assert data["datasets"]["ham10000"]["macro_f1"][f"{arm}-limpio"][0] == _json(HAM_CLEAN)["results"][f"{arm}_clean"]["macro_f1"]


def test_mcc_interval_only_where_published(data):
    for ds, entry in data["datasets"].items():
        for name, (_, low, high) in entry["mcc"].items():
            published = ds != "ham10000" and (name.endswith("-limpio") or name.startswith("DINOv2 público ·"))
            assert (low is not None) == published and (high is not None) == published, (ds, name)


def test_only_isic2020_has_finetuned_markers_and_they_match_their_json(data):
    assert [k for k, e in data["datasets"].items() if e["finetune"]] == ["isic2020"]
    ft = data["datasets"]["isic2020"]["finetune"]
    assert set(ft) == set(fig.R.MODEL_NAMES.values())
    for model, name in fig.R.MODEL_NAMES.items():
        probe = _json(f"{fig.R.FT_ISIC2020_DIR}/json/results_{model}_ft_isic2020.json")["probes"]["finetune"]
        assert ft[name]["macro_f1"][0] == probe["macro_f1"]
        assert ft[name]["macro_f1"][1:] == tuple(probe["ic95_bootstrap"]["macro_f1"])


def test_build_writes_pdf_svg_and_png(tmp_path):
    paths = fig.build(REPO, tmp_path)
    assert {p.suffix for p in paths} == {".pdf", ".svg", ".png"}
    assert all(p.is_file() and p.stat().st_size > 0 and p.stem == "fig2_models" for p in paths)
