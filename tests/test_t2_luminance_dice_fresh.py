"""The committed T2 result is what the committed inputs give, and its table is generated from it."""

from __future__ import annotations

import json

from scripts import t2_luminance_dice as t2

OUT = t2.RESULTS / "2026-09-28-t2-luminance-dice"


def _committed() -> dict:
    return json.loads((OUT / t2.JSON_FILE).read_text(encoding="utf-8"))


def test_the_json_is_rebuilt_exactly_from_the_committed_inputs():
    committed = _committed()
    params = committed["params"]
    rebuilt = t2.build_document(t2.RESULTS, OUT / t2.TERCILES_FILE, params["n_resamples"], params["seed"])
    assert {k: v for k, v in committed.items() if k not in ("provenance", "environment")} == rebuilt


def test_the_run_used_the_registered_parameters_and_commit():
    committed = _committed()
    assert committed["params"] == {"n_resamples": 1000, "seed": 20260923, "clear_margin": -0.02, "min_n": 100}
    assert committed["provenance"]["registration_commit"] == t2.REGISTRATION_COMMIT


def test_the_table_is_generated_from_the_json():
    assert (OUT / t2.TABLE_FILE).read_text(encoding="utf-8") == t2.render_markdown(_committed())


def test_the_verdicts_quoted_in_docs27_are_the_generated_ones():
    doc = (t2.REPO / t2.REGISTRATION_DOC).read_text(encoding="utf-8")
    result = doc[doc.index("**Resultado T2"):].split("\n\n", 1)[0]
    lead = {"ham10000": "En HAM10000 la predicción de Pau queda", "isic2018": "En ISIC 2018 queda"}
    for dataset, cell in _committed()["datasets"].items():
        verdicts = {r["dark_minus_rest"]["dice"]["verdict"] for r in cell["segmenters"].values()}
        assert len(verdicts) == 1, f"{dataset}: the segmenters disagree, docs/27 must say it per segmenter"
        assert f"{lead[dataset]} **{verdicts.pop()}**" in result
