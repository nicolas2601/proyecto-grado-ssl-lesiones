"""docs/31 as data: arms, fixed protocols, reproduction guard, selection, val entries,
the luminance-gap metric and the cost projection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dermssl import metrics
from dermssl import paired_bootstrap as pb
from dermssl import t4_sog

REPO = Path(__file__).resolve().parents[1]
V71 = REPO / "resultados/clasificacion/v71"


def test_fixed_protocols_are_the_committed_v71_winners():
    winners = json.loads((V71 / "selection.json").read_text(encoding="utf-8"))["winners"]
    assert t4_sog.FIXED_PROTOCOLS == winners
    assert t4_sog.WEIGHTS == {"public": "public", "A": "A_soup_a1.0", "B": "B_s42_a1.0"}


def _row(f1, ece=0.1):
    return {m: 0.5 for m in t4_sog.REPRODUCED_METRICS} | {"macro_f1": f1, "ece": ece}


def test_reproduction_passes_within_tolerance_and_fails_beyond_it():
    committed = {cid: _row(0.7) for cid in t4_sog.FIXED_PROTOCOLS.values()}
    close = {cid: _row(0.7 + 5e-7) for cid in committed}
    far = {cid: _row(0.7 + 1e-4) for cid in committed}
    assert t4_sog.reproduction_check(close, committed)["passed"] is True
    report = t4_sog.reproduction_check(far, committed)
    assert report["passed"] is False and report["tolerance"] == 1e-6


def test_reproduction_fails_when_a_fixed_protocol_is_missing():
    committed = {cid: _row(0.7) for cid in t4_sog.FIXED_PROTOCOLS.values()}
    partial = dict(list(committed.items())[:2])
    assert t4_sog.reproduction_check(partial, committed)["passed"] is False


def _grid(weights: str) -> dict:
    return {f"{weights}@224|cls|none|id": _row(0.60), f"{weights}@224|cls4|none|tta": _row(0.70),
            f"{weights}@448|cls4|none|tta": _row(0.90)}


def test_sog_winners_use_the_docs28_rule_per_arm_at_224_only():
    rows = {k: v for w in t4_sog.WEIGHTS.values() for k, v in _grid(w).items()}
    winners = t4_sog.sog_winners(rows)
    assert winners == {arm: f"{w}@224|cls4|none|tta" for arm, w in t4_sog.WEIGHTS.items()}


def test_val_entries_hold_the_protocol_fixed_for_the_primary_contrast():
    winners = {arm: f"{w}@224|cls|none|id" for arm, w in t4_sog.WEIGHTS.items()}
    entries = t4_sog.val_entries(winners)
    for arm, fixed in t4_sog.FIXED_PROTOCOLS.items():
        assert entries[f"{arm}_id"] == (t4_sog.IDENTITY, fixed)
        assert entries[f"{arm}_sog"] == (t4_sog.SOG, fixed)
        assert entries[f"{arm}_sogwin"] == (t4_sog.SOG, winners[arm])
    assert set(t4_sog.PRIMARY_CONTRASTS) == {(f"{a}_sog", f"{a}_id") for a in t4_sog.WEIGHTS}
    assert set(t4_sog.SECONDARY_CONTRASTS) == {(f"{a}_sogwin", f"{a}_id") for a in t4_sog.WEIGHTS}


def test_val_units_group_entries_by_input_and_weights():
    winners = {arm: f"{w}@224|cls|none|id" for arm, w in t4_sog.WEIGHTS.items()}
    units = t4_sog.val_units(t4_sog.val_entries(winners))
    assert set(units) == {(i, w, 224) for i in t4_sog.INPUTS for w in t4_sog.WEIGHTS.values()}
    sog_b = units[(t4_sog.SOG, "B_s42_a1.0", 224)]
    assert sog_b["features"] == {"cls", "cls4"} and sog_b["tta"] is True


def _stratified(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    codes = np.array([0, 1, 2, -1] * (n // 4))
    return y, codes, rng.dirichlet(np.ones(3), n)


def test_the_gap_is_dark_minus_light_macro_f1_in_absolute_value():
    y, codes, proba = _stratified()
    pred = proba.argmax(1)
    dark, light = codes == 0, codes == 2
    expected = abs(metrics.macro_f1(y[dark], pred[dark], 3) - metrics.macro_f1(y[light], pred[light], 3))
    packed = t4_sog.pack_strata(y, codes)
    assert t4_sog.gap_dark_light(packed, proba) == pytest.approx(expected)


def test_the_range_gap_ignores_unavailable():
    y, codes, proba = _stratified()
    pred = proba.argmax(1)
    f1s = [metrics.macro_f1(y[codes == c], pred[codes == c], 3) for c in (0, 1, 2)]
    packed = t4_sog.pack_strata(y, codes)
    assert t4_sog.gap_range(packed, proba) == pytest.approx(max(f1s) - min(f1s))


def test_the_gap_resamples_the_stratum_with_the_image():
    """A paired contrast on packed labels keeps each image's tercile in every draw."""
    y, codes, proba = _stratified(n=80)
    other = np.roll(proba, 1, axis=0)
    packed = t4_sog.pack_strata(y, codes)
    row = pb.contrast_record(arm="sog", control="id", metric=t4_sog.GAP_METRIC, probe="logreg",
                             fraction=1.0, seed=0, y_true=packed, arm_proba=other, control_proba=proba,
                             n_resamples=20, metric_fn=t4_sog.gap_dark_light)
    assert row["point_delta"] == pytest.approx(t4_sog.gap_dark_light(packed, other)
                                               - t4_sog.gap_dark_light(packed, proba))
    assert row["metric"] in pb.LOWER_IS_BETTER


def test_tercile_codes_come_from_the_manifest_labels():
    labels = np.array(["dark", "medium", "light", "unavailable", ""], dtype=object)
    assert t4_sog.tercile_codes(labels).tolist() == [0, 1, 2, -1, -1]
    with pytest.raises(ValueError, match="tercil"):
        t4_sog.tercile_codes(np.array(["III"], dtype=object))


def test_cost_projection_matches_the_registered_estimate():
    cost = t4_sog.cost_projection(n_fit=6409, n_hold=1602, n_train=8011, n_val=2004)
    assert cost["forwards"]["grid"] == 115_350 and cost["forwards"]["val"] == 144_258
    assert cost["gpu_hours"] == pytest.approx(259_608 / 720 / 3600)
    assert 0.4 < cost["cpu_hours"] < 0.8


def _paired(delta_f1: dict, gap: dict, excl_f1=(), excl_gap=()) -> list[dict]:
    rows = []
    for arm in t4_sog.WEIGHTS:
        rows.append({"arm": f"{arm}_sog", "control": f"{arm}_id", "metric": "macro_f1",
                     "point_delta": delta_f1[arm], "excludes_zero": arm in excl_f1})
        rows.append({"arm": f"{arm}_sog", "control": f"{arm}_id", "metric": t4_sog.GAP_METRIC,
                     "point_delta": gap[arm], "excludes_zero": arm in excl_gap})
    return rows


def test_verdicts_hold_when_the_registered_prediction_holds():
    rows = _paired({"A": 0.004, "B": -0.008, "public": 0.015}, {"A": 0.01, "B": -0.02, "public": 0.0})
    verdicts = t4_sog.verdicts(rows)
    assert [verdicts[p]["holds"] for p in ("P1", "P2", "P3")] == [True, True, True]
    assert verdicts["P1"]["improvement"] == {"A": False, "B": False, "public": False}


def test_verdicts_are_falsified_by_size_or_by_an_interval_off_zero():
    big = t4_sog.verdicts(_paired({"A": 0.02, "B": 0.0, "public": 0.0}, {"A": 0.0, "B": 0.0, "public": 0.04}))
    assert big["P1"]["holds"] is False and big["P2"]["holds"] is False and big["P3"]["holds"] is False
    ci = t4_sog.verdicts(_paired({"A": 0.009, "B": 0.0, "public": 0.0}, {"A": -0.02, "B": 0.0, "public": 0.0},
                                 excl_f1=("A",), excl_gap=("A",)))
    assert ci["P1"]["holds"] is False and ci["P3"]["holds"] is False
    assert ci["P1"]["improvement"]["A"] is True and ci["P3"]["reduction"]["A"] is True


def test_verdicts_refuse_a_missing_contrast():
    rows = _paired({"A": 0.0, "B": 0.0, "public": 0.0}, {"A": 0.0, "B": 0.0, "public": 0.0})[1:]
    with pytest.raises(KeyError, match="macro_f1"):
        t4_sog.verdicts(rows)
