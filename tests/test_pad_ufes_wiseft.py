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
