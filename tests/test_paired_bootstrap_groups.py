"""docs/30 2.4: the PAD-UFES paired bootstrap resamples patients, not images.

A patient contributes several images, and those images are not independent
evidence. Drawing images one by one would narrow the interval with that
dependence. The pairing still holds: both arms see the same drawn patients.
"""

from __future__ import annotations

import numpy as np
import pytest

from dermssl import paired_bootstrap as pb


def _proba(n_rows: int, n_classes: int, seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).random((n_rows, n_classes))
    return raw / raw.sum(axis=1, keepdims=True)


class TestTheGroupedDraw:
    def test_a_drawn_group_enters_with_all_its_rows(self):
        groups = np.array(["p1", "p1", "p1", "p2", "p3", "p3"])
        draw = pb.index_sampler(len(groups), groups)
        rng = np.random.RandomState(0)
        for _ in range(20):
            picks = draw(rng)
            drawn = groups[picks]
            for patient in set(drawn):
                size = int((groups == patient).sum())
                assert int((drawn == patient).sum()) % size == 0, (
                    f"{patient} entro partido: una remuestra agrupada toma el paciente entero"
                )

    def test_it_draws_as_many_groups_as_there_are(self):
        groups = np.repeat(np.arange(7), [1, 2, 3, 1, 1, 4, 2])
        draw = pb.index_sampler(len(groups), groups)
        rng = np.random.RandomState(3)
        picks = draw(rng)
        starts = sum(1 for i in range(len(picks)) if i == 0 or groups[picks[i]] != groups[picks[i - 1]]
                     or picks[i] <= picks[i - 1])
        assert starts == 7

    def test_without_groups_the_draw_is_the_published_one(self):
        """Every interval already published used ``randint(0, n, n)``; the
        grouped option must not move a single one of them."""
        draw = pb.index_sampler(40)
        assert np.array_equal(draw(np.random.RandomState(5)), np.random.RandomState(5).randint(0, 40, 40))

    def test_a_group_label_per_row_is_required(self):
        with pytest.raises(ValueError, match="cada fila necesita su grupo"):
            pb.index_sampler(5, ["a", "b"])


class TestTheGroupedContrast:
    def test_both_arms_see_the_same_patients(self):
        seen: list[np.ndarray] = []

        def spy(y_true, proba):
            seen.append(y_true.copy())
            return float(len(y_true))

        y_true = np.arange(30) % 3
        groups = np.arange(30) // 3
        pb.paired_contrast(y_true=y_true, arm_proba=_proba(30, 3, 1),
                           control_proba=_proba(30, 3, 2), metric_fn=spy,
                           n_resamples=4, seed=0, groups=groups)
        pairs = seen[2:]
        for index in range(0, len(pairs), 2):
            assert np.array_equal(pairs[index], pairs[index + 1])

    def test_one_patient_repeated_widens_nothing_it_should_not(self):
        """Duplicating every image of a row-level dataset makes a row bootstrap
        twice as confident; grouping the copies gives the original interval."""
        y_true = np.arange(80) % 4
        arm, control = _proba(80, 4, 7), _proba(80, 4, 8)
        base = pb.paired_contrast(y_true=y_true, arm_proba=arm, control_proba=control,
                                  metric_fn=pb.metric_macro_f1, n_resamples=300, seed=1,
                                  groups=np.arange(80))
        twice = pb.paired_contrast(y_true=np.tile(y_true, 2), arm_proba=np.tile(arm, (2, 1)),
                                   control_proba=np.tile(control, (2, 1)),
                                   metric_fn=pb.metric_macro_f1, n_resamples=300, seed=1,
                                   groups=np.tile(np.arange(80), 2))
        rows = pb.paired_contrast(y_true=np.tile(y_true, 2), arm_proba=np.tile(arm, (2, 1)),
                                  control_proba=np.tile(control, (2, 1)),
                                  metric_fn=pb.metric_macro_f1, n_resamples=300, seed=1)
        width = lambda r: r["ci95_high"] - r["ci95_low"]  # noqa: E731
        assert width(twice) == pytest.approx(width(base), rel=0.35)
        assert width(rows) < 0.85 * width(twice)

    def test_the_record_passes_groups_through_and_keeps_its_schema(self):
        y_true = np.arange(40) % 2
        arm, control = _proba(40, 2, 4), _proba(40, 2, 5)
        grouped = pb.contrast_record(arm="a", control="c", metric="macro_f1", probe="logreg",
                                     fraction=1.0, seed=0, y_true=y_true, arm_proba=arm,
                                     control_proba=control, n_resamples=50, groups=np.arange(40) // 4)
        plain = pb.contrast_record(arm="a", control="c", metric="macro_f1", probe="logreg",
                                   fraction=1.0, seed=0, y_true=y_true, arm_proba=arm,
                                   control_proba=control, n_resamples=50)
        assert set(grouped) == set(plain)
        assert grouped["point_delta"] == plain["point_delta"]
        assert (grouped["ci95_low"], grouped["ci95_high"]) != (plain["ci95_low"], plain["ci95_high"])
