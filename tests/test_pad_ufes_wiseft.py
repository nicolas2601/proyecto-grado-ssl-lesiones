"""dermssl.pad_ufes_wiseft: the docs/30 8 WiSE-FT alpha curve on PAD-UFES-20, synthetic inputs only."""

from __future__ import annotations

import pytest

from dermssl import pad_ufes_readings as pr
from dermssl import pad_ufes_wiseft as pw
from dermssl import v71_protocol as vp


class TestTheGrid:
    def test_the_alphas_are_the_v71_grid_plus_public(self):
        assert pw.ALPHAS == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        assert pw.ALPHAS[1:] == vp.ALPHAS

    def test_the_three_models_are_a_star_s42_s43_and_the_soup(self):
        assert pw.MODELS == ("A_s42", "A_s43", "A_soup")

    @pytest.mark.parametrize("model, alpha, expected", [
        ("A_s42", 0.2, "A_s42_a0.2"), ("A_s43", 0.8, "A_s43_a0.8"), ("A_soup", 1.0, "A_soup_a1.0")])
    def test_a_point_is_named_as_v71_names_its_weights(self, model, alpha, expected):
        assert pw.weights_of(model, alpha) == expected
        assert vp.parse_weights(expected)["alpha"] == alpha

    @pytest.mark.parametrize("model", pw.MODELS)
    def test_alpha_zero_is_exactly_public_for_every_model(self, model):
        assert pw.weights_of(model, 0.0) == vp.PUBLIC == pr.PUBLIC

    def test_an_alpha_off_the_grid_is_refused(self):
        with pytest.raises(ValueError, match="grilla"):
            pw.weights_of("A_s42", 0.5)

    def test_an_unregistered_model_is_refused(self):
        with pytest.raises(ValueError, match="modelo"):
            pw.weights_of("B_s42", 0.8)


class TestAlphaZeroIsShared:
    def test_public_is_one_role_and_appears_once(self):
        roles = pw.roles()
        assert roles.count(vp.PUBLIC) == 1
        assert len(roles) == 1 + len(pw.MODELS) * (len(pw.ALPHAS) - 1)
        assert len(set(roles)) == len(roles)

    @pytest.mark.parametrize("model", pw.MODELS)
    def test_every_curve_starts_at_the_same_public_role(self, model):
        curve = pw.curve(model)
        assert list(curve) == list(pw.ALPHAS)
        assert curve[0.0] == vp.PUBLIC
        assert set(curve.values()) <= set(pw.roles())

    def test_the_w1_pair_is_alpha_08_against_alpha_1_per_model(self):
        assert pw.w1_pair("A_soup") == ("A_soup_a0.8", "A_soup_a1.0")
        assert all(pw.w1_pair(m) in pw.contrast_pairs() for m in pw.MODELS)

    def test_every_contrast_is_against_the_adapted_end_of_its_own_curve(self):
        for arm, control in pw.contrast_pairs():
            model = next(m for m in pw.MODELS if control == pw.weights_of(m, 1.0))
            assert arm in pw.curve(model).values() and arm != control


def _curve(*f1s) -> dict:
    return dict(zip(pw.ALPHAS, f1s))


class TestPadW2:
    """No step from alpha to alpha - 0.2 lowers macro-F1 by more than 0.01; public is the max or within 0.01."""

    def test_the_tolerance_is_one_registered_constant(self):
        assert pw.W2_TOLERANCE == 0.01

    @pytest.mark.parametrize("f1s", [
        (0.5, 0.5, 0.5, 0.5, 0.5, 0.5),                # flat
        (0.60, 0.58, 0.55, 0.52, 0.50, 0.45),          # public best, monotone
        (0.60, 0.59, 0.58, 0.57, 0.56, 0.55),          # every step drops exactly 0.01 going up in alpha
        (0.50, 0.505, 0.51, 0.505, 0.50, 0.49),        # public 0.01 below the max, small steps
    ])
    def test_a_curve_inside_the_tolerance_holds(self, f1s):
        assert pw.w2_model(_curve(*f1s))["holds"] is True

    def test_a_step_down_of_exactly_the_tolerance_is_allowed(self):
        result = pw.w2_model(_curve(0.59, 0.60, 0.60, 0.60, 0.60, 0.60))
        assert result["violations"] == [] and result["holds"] is True

    def test_a_step_down_just_over_the_tolerance_is_a_violation(self):
        result = pw.w2_model(_curve(0.60, 0.6101, 0.60, 0.60, 0.60, 0.60))
        assert result["holds"] is False
        assert [(v["from_alpha"], v["to_alpha"]) for v in result["violations"]] == [(0.2, 0.0)]
        assert result["violations"][0]["drop"] == pytest.approx(0.0101)

    def test_public_more_than_the_tolerance_below_the_max_fails_even_with_small_steps(self):
        result = pw.w2_model(_curve(0.50, 0.506, 0.512, 0.512, 0.512, 0.512))
        assert result["violations"] == []
        assert result["public_gap_to_max"] == pytest.approx(0.012)
        assert result["holds"] is False

    def test_a_missing_alpha_or_a_non_finite_point_is_refused(self):
        with pytest.raises(ValueError, match="alpha"):
            pw.w2_model({a: 0.5 for a in pw.ALPHAS[:-1]})
        with pytest.raises(ValueError, match="finito"):
            pw.w2_model(_curve(0.5, 0.5, float("nan"), 0.5, 0.5, 0.5))

    def test_the_verdict_needs_all_three_models(self):
        good = _curve(0.60, 0.58, 0.55, 0.52, 0.50, 0.45)
        bad = _curve(0.40, 0.45, 0.50, 0.52, 0.55, 0.58)
        assert pw.w2_verdict({m: good for m in pw.MODELS})["PAD-W2"] == pw.MET
        mixed = pw.w2_verdict({"A_s42": good, "A_s43": bad, "A_soup": good})
        assert mixed["PAD-W2"] == pw.NOT_MET
        assert [m for m, r in mixed["by_model"].items() if not r["holds"]] == ["A_s43"]

    def test_the_verdict_refuses_a_missing_model(self):
        with pytest.raises(ValueError, match="modelo"):
            pw.w2_verdict({"A_s42": _curve(*[0.5] * 6)})


def _row(model: str, delta: float, low: float, high: float, metric: str = "macro_f1") -> dict:
    arm, control = pw.w1_pair(model)
    return {"arm": arm, "control": control, "metric": metric, "point_delta": delta,
            "ci95_low": low, "ci95_high": high, "excludes_zero": low > 0 or high < 0}


class TestPadW1:
    """alpha 0.8 - alpha 1.0 on primary macro-F1: point > 0 in all three; "mejora" only if the CI is above 0."""

    def _rows(self, **over) -> list[dict]:
        base = {m: (0.02, -0.01, 0.05) for m in pw.MODELS}
        base.update(over)
        other = [_row(m, -9.0, -9.5, -8.5, metric="ece") for m in pw.MODELS]  # never read
        return [_row(m, *base[m]) for m in pw.MODELS] + other

    def test_positive_points_in_all_three_meet_it_without_any_mejora(self):
        result = pw.w1_verdict(self._rows())
        assert result["PAD-W1"] == pw.MET
        assert {m: r["mejora"] for m, r in result["by_model"].items()} == dict.fromkeys(pw.MODELS, False)

    def test_mejora_only_where_the_interval_is_above_zero(self):
        result = pw.w1_verdict(self._rows(A_s43=(0.04, 0.01, 0.07)))
        assert result["PAD-W1"] == pw.MET
        assert [m for m, r in result["by_model"].items() if r["mejora"]] == ["A_s43"]

    @pytest.mark.parametrize("point", [0.0, -0.001])
    def test_a_point_at_or_below_zero_in_one_model_refutes_it(self, point):
        result = pw.w1_verdict(self._rows(A_soup=(point, -0.03, 0.02)))
        assert result["PAD-W1"] == pw.NOT_MET
        assert result["by_model"]["A_soup"]["holds"] is False

    def test_an_interval_below_zero_is_never_mejora(self):
        result = pw.w1_verdict(self._rows(A_s42=(-0.04, -0.07, -0.01)))
        assert result["by_model"]["A_s42"]["mejora"] is False
        assert result["PAD-W1"] == pw.NOT_MET

    def test_a_missing_w1_row_is_refused(self):
        rows = [r for r in self._rows() if r["arm"] != pw.w1_pair("A_s43")[0]]
        with pytest.raises(ValueError, match="A_s43"):
            pw.w1_verdict(rows)


def _grid(sources=("s42", "soup"), drop=()) -> dict:
    rows = {}
    for k, weights in enumerate([vp.PUBLIC] + [vp.weights_name("A", s, a) for s in sources for a in vp.ALPHAS]):
        for feature, weighting, tta in (("cls", "none", False), ("cls_attn", "la0.5", True)):
            cid = vp.config_id(weights, 224, feature, weighting, tta)
            value = 0.5 + k / 100 + (0.2 if tta else 0.0)
            rows[cid] = {m: value for m in pw.HAM_METRICS} | {
                "mel_auc": 0.0, "ci95": {m: [value - 0.01, value + 0.01] for m in pw.HAM_METRICS}}
    for cid in drop:
        rows.pop(cid)
    return {"code_revision": "c" * 40, "rows": rows}


class TestTheHamHoldoutCurve:
    def test_s42_and_the_soup_are_read_from_the_cls_none_id_rows(self):
        curves = pw.ham_curves(_grid())
        s42 = curves["A_s42"]
        assert s42["available"] is True
        assert list(s42["points"]) == [pw.alpha_key(a) for a in pw.ALPHAS]
        assert s42["points"]["0.0"]["config"] == "public@224|cls|none|id"
        assert s42["points"]["0.8"]["config"] == "A_s42_a0.8@224|cls|none|id"
        assert s42["points"]["0.8"]["macro_f1"] == pytest.approx(0.54)  # never the tta row
        assert s42["points"]["0.8"]["ci95"]["ece"] == pytest.approx([0.53, 0.55])
        assert curves["A_soup"]["points"]["1.0"]["config"] == "A_soup_a1.0@224|cls|none|id"

    def test_alpha_zero_is_the_same_public_row_for_every_curve(self):
        curves = pw.ham_curves(_grid())
        assert curves["A_s42"]["points"]["0.0"] == curves["A_soup"]["points"]["0.0"]

    def test_s43_has_no_ham_curve_and_says_so(self):
        s43 = pw.ham_curves(_grid())["A_s43"]
        assert s43["available"] is False and "points" not in s43
        assert "s43" in s43["note"]

    def test_only_the_registered_metrics_are_carried(self):
        point = pw.ham_curves(_grid())["A_s42"]["points"]["0.4"]
        assert set(point) == {"config", *pw.HAM_METRICS, "ci95"}
        assert pw.HAM_METRICS == ("macro_f1", "balanced_accuracy", "auc", "ece")

    def test_a_curve_with_a_missing_point_is_refused(self):
        with pytest.raises(ValueError, match="A_soup_a0.6"):
            pw.ham_curves(_grid(drop=("A_soup_a0.6@224|cls|none|id",)))


N_PAD = 90


@pytest.fixture(scope="module")
def document():
    """A document on synthetic probabilities: the curve peaks at alpha 0.8 for every model."""
    import numpy as np

    from dermssl import pad_ufes as pu

    mp = pytest.MonkeyPatch()
    mp.setattr(pr, "BOOTSTRAP", {"n_resamples": 12, "rng_seed": 20260923})
    cycle = ["bcc", "mel", "nv", "akiec", "bkl"]
    labels = np.array([cycle[i % 5] for i in range(N_PAD)])
    truth = np.array([pr.HAM_CLASSES.index(x) for x in labels])
    rng = np.random.default_rng(0)
    probas = {}
    for role in pw.roles():
        alpha = vp.parse_weights(role)["alpha"]
        signal = 2.0 - abs(alpha - 0.8) * 2
        raw = rng.random((N_PAD, len(pr.HAM_CLASSES)))
        raw[np.arange(N_PAD), truth] += signal
        probas[role] = raw / raw.sum(axis=1, keepdims=True)
    pad = {"ham_labels": labels, "groups": np.array([f"P{i // 3}" for i in range(N_PAD)]),
           "fst_bins": np.array([pu.FST_BINS[i % 4] for i in range(N_PAD)]), "fst": [2] * N_PAD}
    doc = pw.build_document(probas, pad, {"code_revision": "f" * 40}, _grid())
    mp.undo()
    return doc


class TestTheDocument:
    def test_every_reading_scores_every_role_once(self, document):
        from dermssl import pad_ufes as pu

        assert set(document["readings"]) == set(pu.READINGS)
        for entry in document["readings"].values():
            assert list(entry["arms"]) == pw.roles()

    def test_the_verdicts_live_on_the_primary_reading_only(self, document):
        assert set(document["verdicts"]) == {"PAD-W1", "PAD-W2"}
        assert all("verdicts" not in entry for entry in document["readings"].values())

    def test_w2_reads_the_primary_macro_f1_points_of_each_curve(self, document):
        arms = document["readings"]["primary"]["arms"]
        for model in pw.MODELS:
            points = document["verdicts"]["PAD-W2"]["by_model"][model]["macro_f1"]
            assert points == {pw.alpha_key(a): arms[role]["macro_f1"] for a, role in pw.curve(model).items()}

    def test_w1_reads_the_primary_paired_row(self, document):
        rows = document["readings"]["primary"]["paired"]
        for model in pw.MODELS:
            arm, control = pw.w1_pair(model)
            row = next(r for r in rows if (r["arm"], r["control"], r["metric"]) == (arm, control, "macro_f1"))
            assert document["verdicts"]["PAD-W1"]["by_model"][model]["point_delta"] == row["point_delta"]

    def test_the_strata_carry_rule_b_and_no_verdict(self, document):
        strata = document["fst_strata"]
        assert strata["exploratory"] is True and strata["reading"] == "primary"
        assert all("verdicts" not in entry for entry in strata["bins"].values())

    def test_the_ham_side_is_the_committed_curve_and_s43_has_none(self, document):
        assert document["ham_holdout"]["curves"]["A_s43"]["available"] is False
        assert document["ham_holdout"]["curves"]["A_s42"]["points"]["0.0"]["config"] == "public@224|cls|none|id"

    def test_the_grid_and_the_provenance_are_recorded(self, document):
        assert document["alphas"] == list(pw.ALPHAS)
        assert document["curves"]["A_soup"]["0.8"] == "A_soup_a0.8"
        assert document["code_revision"] == "f" * 40

    def test_a_missing_role_is_refused(self):
        with pytest.raises(ValueError, match="A_s43_a0.4"):
            pw.check_roles({r: None for r in pw.roles() if r != "A_s43_a0.4"})


class TestAlphaKeys:
    """Every alpha-keyed mapping of the document uses one explicit text key, never a float's repr."""

    def test_the_key_is_one_decimal(self):
        assert [pw.alpha_key(a) for a in pw.ALPHAS] == ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]

    def test_every_alpha_mapping_is_keyed_by_text_already_in_memory(self, document):
        keys = [pw.alpha_key(a) for a in pw.ALPHAS]
        assert list(document["verdicts"]["PAD-W2"]["by_model"]["A_s42"]["macro_f1"]) == keys
        assert list(document["ham_holdout"]["curves"]["A_soup"]["points"]) == keys
        assert all(list(points) == keys for points in document["curves"].values())
