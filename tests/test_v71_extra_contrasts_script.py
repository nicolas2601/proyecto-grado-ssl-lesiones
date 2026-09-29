"""``scripts/v71_extra_contrasts.py`` reproduces the committed val read and its own JSON.

The point values of the fair operating points must be the per-entry numbers
already published in ``v71_final_eval.json``, and the s42 macro-F1 contrast
must be the published one to the last digit: if not, the script is measuring
something else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import v71_extra_contrasts as xc  # noqa: E402

V71 = REPO / "resultados" / "clasificacion" / "v71"
HAVE_DATA = (V71 / "v71_final_eval.probas.npz").exists() and xc.V7_S43.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="val probas not committed")


@pytest.fixture(scope="module")
def inputs():
    return xc.load_inputs(V71, xc.V7_S43)


@pytest.fixture(scope="module")
def doc(inputs):
    """Computed once: every contrast is 1000 resamples."""
    return xc.document(inputs)


@pytest.fixture(scope="module")
def published():
    return json.loads((V71 / "v71_final_eval.json").read_text(encoding="utf-8"))


@needs_data
def test_fair_mel_points_are_the_published_per_entry_values(doc, published):
    rows = doc["fair_mel"]["contrasts"]
    assert {(r["arm"], r["control"]) for r in rows} == set(xc.FAIR_PAIRS)
    for r in rows:
        assert r["point_arm"] == published["results"][r["arm"]][r["metric"]]
        assert r["point_control"] == published["results"][r["control"]][r["metric"]]


@needs_data
def test_the_s42_macro_f1_contrast_is_the_published_one(doc, published):
    rows = doc["s43"]["contrasts"]
    (mine,) = [r for r in rows if (r["arm"], r["control"], r["metric"]) == ("A71", "A_v7", "macro_f1")]
    (ref,) = [r for r in published["paired"] if (r["arm"], r["control"], r["metric"]) == ("A71", "A_v7", "macro_f1")]
    assert (mine["ci95_low"], mine["ci95_high"]) == (ref["ci95_low"], ref["ci95_high"])
    assert mine["ci95_low"] == pytest.approx(0.0362, abs=1e-4)
    assert mine["ci95_high"] == pytest.approx(0.1215, abs=1e-4)


@needs_data
def test_mel_to_nv_covers_every_entry_and_the_s43_primaries(doc, inputs):
    section = doc["mel_to_nv"]
    assert set(section["entries"]) == {*xc.V71_LABELS, "A_v7_s43", "B_v7_s43"}
    assert {(r["arm"], r["control"]) for r in section["contrasts"]} == set(xc.MEL_TO_NV_PAIRS)
    n_mel = int((inputs["y_true"] == xc.MEL).sum())
    assert all(e["n_mel"] == n_mel for e in section["entries"].values())


@needs_data
def test_the_s43_section_warns_that_the_soup_contains_s43(doc):
    assert "s43" in doc["s43"]["note"] and "sopa" in doc["s43"]["note"]


def test_misaligned_labels_are_refused(tmp_path):
    y = np.arange(10) % 7
    np.savez(tmp_path / "v71.npz", y_true=y, **{k: np.zeros((10, 7)) for k in xc.V71_LABELS},
             **{f"{k}__fit80": np.zeros((10, 7)) for k in xc.V71_LABELS})
    np.savez(tmp_path / "s43.npz", y_true=y[::-1], plain=np.zeros((10, 7)), proposed=np.zeros((10, 7)))
    with pytest.raises(ValueError, match="y_true"):
        xc.check_aligned(np.load(tmp_path / "v71.npz"), np.load(tmp_path / "s43.npz"))


@needs_data
def test_the_committed_json_is_what_the_script_writes(doc):
    committed = V71 / "extra_contrasts.json"
    assert committed.exists(), "falta resultados/clasificacion/v71/extra_contrasts.json"
    assert json.loads(committed.read_text(encoding="utf-8")) == json.loads(json.dumps(doc))


def test_main_writes_the_document(tmp_path, monkeypatch):
    monkeypatch.setattr(xc, "load_inputs", lambda *_: {})
    monkeypatch.setattr(xc, "document", lambda inputs: {"ok": 1})
    out = tmp_path / "extra.json"
    assert xc.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"ok": 1}
