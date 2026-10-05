"""The registered E3 adoption rule and the holdout scoring (docs/33 section 3)."""

import csv

import numpy as np
import pytest

from dermssl import e3_selection as sel
from dermssl import t4_sog


def row(f1, mel=0.5, dark=0.5, ece=0.1):
    return {"macro_f1": f1, "mel_recall": mel, "dark_macro_f1": dark, "ece": ece}


def decide(rows, **kw):
    return sel.adoption(rows, **kw)


def test_adopted_when_f1_is_within_the_margin_and_melanoma_recall_improves():
    out = decide({"B71": row(0.80), "F1": row(0.795, mel=0.6)})
    assert out["variants"]["F1"]["adopted"] and out["winner"] == "F1"


def test_adopted_by_the_dark_tercile_alone():
    out = decide({"B71": row(0.80), "F1": row(0.80, mel=0.5, dark=0.51)})
    assert out["variants"]["F1"]["adopted"]


def test_the_f1_margin_is_inclusive_at_exactly_one_point_and_exclusive_below():
    at = decide({"B71": row(0.80), "F1": row(0.79, mel=0.6)})
    below = decide({"B71": row(0.80), "F1": row(0.7899, mel=0.6)})
    assert at["variants"]["F1"]["passes_macro_f1"] and not below["variants"]["F1"]["passes_macro_f1"]
    assert below["winner"] is None


def test_without_a_gain_in_either_subgroup_the_variant_is_not_adopted():
    out = decide({"B71": row(0.80), "F1": row(0.85, mel=0.5, dark=0.4)})
    assert not out["variants"]["F1"]["adopted"] and out["winner"] is None
    assert out["decision"] == "no_adoption"


def test_nan_melanoma_recall_never_counts_as_a_gain():
    out = decide({"B71": row(0.80), "F1": row(0.80, mel=float("nan"), dark=0.4)})
    assert not out["variants"]["F1"]["adopted"]


def test_the_bar_is_the_best_no_mask_row_and_the_delta_is_against_the_reference():
    rows = {"B71": row(0.78), "v71:B_soup": row(0.82), "F1": row(0.80, mel=0.6)}
    out = decide(rows, nomask=("B71", "v71:B_soup"))
    assert out["best_nomask"] == "v71:B_soup" and not out["variants"]["F1"]["passes_macro_f1"]
    assert "v71:B_soup" not in out["variants"]  # no-mask rows are references, not candidates


def test_among_adopted_variants_the_winner_is_the_docs28_rule_with_ece_tie_break():
    rows = {"B71": row(0.80), "F1": row(0.81, mel=0.6, ece=0.2), "F2": row(0.809, mel=0.6, ece=0.05),
            "F3": row(0.70, mel=0.9)}
    out = decide(rows)
    assert out["adopted"] == ["F1", "F2"] and out["winner"] == "F2"


def test_the_reference_must_be_present():
    with pytest.raises(sel.SelectionError):
        decide({"F1": row(0.8)})


def test_dark_tercile_codes_come_from_the_manifest_with_its_sha_checked(tmp_path):
    path = tmp_path / "m.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "tone_tercile"])
        writer.writeheader()
        writer.writerows([{"image_id": "a", "tone_tercile": "dark"}, {"image_id": "b", "tone_tercile": "light"}])
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    codes = sel.tercile_codes(["b", "a", "zzz"], path, sha)
    assert list(codes) == [t4_sog.TERCILES["light"], t4_sog.TERCILES["dark"], t4_sog.UNAVAILABLE_CODE]
    with pytest.raises(ValueError):
        sel.tercile_codes(["a"], path, "0" * 64)


def test_the_registered_manifest_sha_is_the_one_in_docs33():
    assert sel.MANIFEST_SHA256 == "57df1e65e88d3b54ba530d6e026010bfc8ce25d1437e50b4ffe37cfd7d4f87da"


def test_score_variants_adds_the_dark_tercile_macro_f1_and_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 90)
    codes = np.array([0, 1, 2] * 30)
    proba = np.eye(3)[y] * 0.6 + 0.4 / 3
    rows = sel.score_variants(y, {"B71": proba}, codes, mel=1, n_resamples=20)
    r = rows["B71"]
    assert r["macro_f1"] == pytest.approx(1.0) and r["dark_macro_f1"] == pytest.approx(1.0)
    assert r["n_dark"] == 30 and "ci95" in r
