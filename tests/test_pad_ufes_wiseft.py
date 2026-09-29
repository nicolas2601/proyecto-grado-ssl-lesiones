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
