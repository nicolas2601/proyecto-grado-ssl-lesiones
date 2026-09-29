"""Closing-table rows of the exploratory blocks T1-T7.

Each row takes its verdict from the experiment's own JSON and never
re-derives it; its key number is the JSON field, which the perturbation
tests pin: change the field in a copy of the file and the row follows.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_closing_table as ct  # noqa: E402
import closing_rows_exploratory as ex  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def _row(fragment: str) -> ct.Row:
    matches = [r for r in ct.build_rows() if fragment in r.pedido]
    assert len(matches) == 1, f"{fragment!r} matches {len(matches)} rows"
    return matches[0]


def _spec(fragment: str) -> ct.RowSpec:
    return next(s for s in ex.rows() if fragment in s.pedido)


def test_t1_verdict_is_the_one_its_json_records():
    words = set(_load(ex.T1)["verdicts"].values())
    assert words == {ct.MET}
    assert _row("T1:").veredicto == ct.MET


@pytest.mark.parametrize(("ref", "fragment"), [("ham10000", "T2 (HAM10000)"), ("isic2018", "T2 (ISIC 2018)")])
def test_t2_verdict_per_dataset_is_the_one_its_json_records(ref, fragment):
    segs = _load(ex.T2)["datasets"][ref]["segmenters"]
    words = {s["dark_minus_rest"]["dice"]["verdict"] for s in segs.values()}
    assert len(words) == 1
    assert _row(fragment).veredicto == words.pop()


def test_t3_t6_t7_verdicts_are_the_ones_their_json_records():
    assert _row("T3:").veredicto == _load(ex.T3)["prediction"]["verdict"]
    assert _row("T6:").veredicto == _load(ex.T6)["datasets"]["ham10000"]["evaluation"]["verdict"]["prediction"]
    assert _row("T7:").veredicto == _load(ex.T7)["datasets"]["ham10000"]["verdict"]["conclusion"]


def test_t4_verdict_reads_the_holds_flags_of_its_json():
    holds = {v["holds"] for v in _load(ex.T4)["verdicts"].values()}
    assert holds == {False}
    assert _row("T4:").veredicto == ct.NOT_MET


def test_a_row_whose_json_verdicts_disagree_fails_instead_of_picking_one():
    assert ex.registered_verdict([ct.MET, ct.MET]) == ct.MET
    with pytest.raises(ValueError):
        ex.registered_verdict([ct.MET, ct.NOT_MET])
    with pytest.raises(ValueError):
        ex.registered_verdict(["se sostiene"])


def test_the_distinguishable_verdict_is_explained_in_the_legend():
    assert ct.DISTINGUISHABLE == "distinguible pero no clara"
    assert ct.DISTINGUISHABLE in "\n".join(ct.rules())


def _set(doc: dict, keys: tuple, value) -> None:
    for k in keys[:-1]:
        doc = doc[k]
    doc[keys[-1]] = value


def _perturbed_row(fragment: str, path: Path, keys: tuple, value, tmp_path: Path) -> ct.Row:
    spec = _spec(fragment)
    for source in spec.sources:
        doc = copy.deepcopy(_load(source))
        if source == path:
            _set(doc, keys, value)
        (tmp_path / source).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / source).write_text(json.dumps(doc), encoding="utf-8")
    return ct.build_row(spec, root=tmp_path)


PERTURBATIONS = [
    ("T1:", ex.T1, ("metrics", "whole_image", "accuracy"), 0.123456, "0,1235"),
    ("T1:", ex.T1, ("metrics", "masked", "linear_kappa"), 0.987654, "0,9877"),
    ("T2 (HAM10000)", ex.T2, ("datasets", "ham10000", "segmenters", "A2", "dark_minus_rest", "dice", "delta"),
     -0.4321, "−0,4321"),
    ("T2 (ISIC 2018)", ex.T2,
     ("datasets", "isic2018", "segmenters", "student_r3", "dark_minus_rest", "dice", "ci95_high"), -0.3333, "−0,3333"),
    ("T3:", ex.T3, ("segmenters", "a2", "v7_astar_s42", "dark", "contrasts", "mask_pool", "macro_f1", "ci95_low"),
     0.4444, "+0,4444"),
    ("T4:", ex.T4, ("paired", 0, "point_delta"), -0.5555, "−0,5555"),
    ("T6:", ex.T6, ("datasets", "ham10000", "evaluation", "gap", "reduction", "delta"), 0.6666, "+0,6666"),
    ("T7:", ex.T7, ("datasets", "ham10000", "reduction", "T7", "ci95_low"), -0.7777, "−0,7777"),
    ("T7:", ex.T7_R0, ("analysis", "teacher_gap_binary", "delta"), -0.8888, "−0,8888"),
]


@pytest.mark.parametrize(("fragment", "path", "keys", "value", "expected"), PERTURBATIONS,
                         ids=[f"{p[0]}{'.'.join(map(str, p[2][-2:]))}" for p in PERTURBATIONS])
def test_each_new_row_number_is_its_json_field(fragment, path, keys, value, expected, tmp_path):
    original = ct.build_row(_spec(fragment))
    assert expected not in original.resultado
    assert expected in _perturbed_row(fragment, path, keys, value, tmp_path).resultado


def test_a_perturbed_json_verdict_moves_the_row_verdict(tmp_path):
    row = _perturbed_row("T6:", ex.T6, ("datasets", "ham10000", "evaluation", "verdict", "prediction"),
                         ct.MET, tmp_path)
    assert row.veredicto == ct.MET


def test_the_new_rows_follow_the_existing_ones_unchanged():
    import closing_rows_classifier as clf
    import closing_rows_segmentation as segrows

    existing = [s.pedido for s in clf.rows() + segrows.rows()]
    pedidos = [s.pedido for s in ct.specs()]
    assert pedidos[:len(existing)] == existing
    assert [p.split(" ")[0].rstrip(":") for p in pedidos[len(existing):]] == [
        "T1", "T2", "T2", "T3", "T4", "T6", "T7"]


def test_every_new_row_cites_the_commits_its_json_records():
    for fragment, path, keys in [("T1:", ex.T1, ("registration_commit",)),
                                 ("T3:", ex.T3, ("provenance", "registration_commit")),
                                 ("T4:", ex.T4, ("registration_commit",)),
                                 ("T7:", ex.T7, ("provenance", "registration_commit"))]:
        doc = _load(path)
        for k in keys:
            doc = doc[k]
        assert f"`{doc[:7]}`" in _row(fragment).resultado
