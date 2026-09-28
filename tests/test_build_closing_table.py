"""The closing table decides every verdict by rule from committed JSON.

These tests pin the rules (paired CI, registered thresholds, the two-seed
reading of docs/28) and the fail-honest path: a registered item whose data
file is missing renders as pending instead of inventing a number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import build_closing_table as ct  # noqa: E402


@pytest.mark.parametrize(
    ("low", "high", "higher_is_better", "expected"),
    [
        (0.01, 0.05, True, ct.BETTER),
        (-0.05, -0.01, True, ct.WORSE),
        (-0.01, 0.02, True, ct.SAME),
        (0.01, 0.05, False, ct.WORSE),
        (-0.05, -0.01, False, ct.BETTER),
        (0.0, 0.03, True, ct.SAME),
    ],
)
def test_paired_verdict_requires_the_interval_to_exclude_zero(low, high, higher_is_better, expected):
    assert ct.paired_verdict(low, high, higher_is_better) == expected


def test_every_verdict_word_is_in_the_closed_vocabulary():
    assert {ct.BETTER, ct.SAME, ct.WORSE, ct.MET, ct.NOT_MET} == {
        "mejora", "no cambia", "empeora", "cumplida", "no cumplida"}


def test_s4_threshold_uses_the_lower_bound_of_the_paired_interval():
    assert ct.s4_threshold_met({"ci95_low": -0.0099}) is True
    assert ct.s4_threshold_met({"ci95_low": -0.0101}) is False


def test_sdiag_below_student_needs_non_overlapping_intervals():
    student = {"ci95_low": 0.70, "ci95_high": 0.72}
    assert ct.below_without_overlap({"ci95_low": 0.40, "ci95_high": 0.69}, student) is True
    assert ct.below_without_overlap({"ci95_low": 0.60, "ci95_high": 0.705}, student) is False


def _row(arm: str, direction: str) -> dict:
    return {"arm": f"{arm}__bbox_crop", "control": f"{arm}__none", "metric": "macro_f1",
            "direction": direction}


def test_mask_helps_only_if_both_seeds_of_the_same_arm_win():
    one_seed = [_row("v7_astar_s42", "arm_better"), _row("v7_astar_s43", "indistinguishable")]
    both = [_row("v7_astar_s42", "arm_better"), _row("v7_astar_s43", "arm_better")]
    assert ct.mask_helps(one_seed) is False
    assert ct.mask_helps(both) is True


def test_an_unseeded_backbone_needs_its_single_run_to_win():
    assert ct.mask_helps([_row("dinov2_public_vitb14", "arm_better")]) is True
    assert ct.mask_helps([_row("dinov2_public_vitb14", "control_better")]) is False


def test_seed_verdict_is_met_only_when_every_seed_meets_it():
    assert ct.seeds_verdict({42: "cumplida", 43: "cumplida"}) == ct.MET
    assert ct.seeds_verdict({42: "no cumplida", 43: "cumplida"}) == ct.NOT_MET


def test_a_missing_data_file_renders_the_row_as_pending(tmp_path):
    spec = ct.RowSpec("Profesor", "pedido", "hecho", (Path("no/existe.json"),), lambda d: ("x", ct.MET))
    row = ct.build_row(spec, root=tmp_path)
    assert row.resultado == ct.PENDING
    assert row.veredicto == ct.PENDING


def test_a_present_data_file_feeds_the_builder(tmp_path):
    path = Path("datos.json")
    (tmp_path / path).write_text(json.dumps({"v": 0.5}), encoding="utf-8")
    spec = ct.RowSpec("Pau", "p", "h", (path,), lambda d: (str(d[path]["v"]), ct.MET))
    row = ct.build_row(spec, root=tmp_path)
    assert (row.resultado, row.veredicto) == ("0.5", ct.MET)


def test_an_item_without_any_data_source_is_pending():
    row = ct.build_row(ct.RowSpec("Profesor", "D1", "no corrido", (), None))
    assert row.veredicto == ct.PENDING


def test_render_has_one_verdict_per_row_from_the_allowed_set():
    allowed = {ct.BETTER, ct.SAME, ct.WORSE, ct.MET, ct.NOT_MET, ct.PENDING, ct.DONE}
    rows = ct.build_rows()
    assert rows, "the table must not be empty"
    assert {r.veredicto for r in rows if not r.veredicto.startswith(ct.MEETS_THRESHOLD)} <= allowed


def _by_pedido(fragment: str):
    return next(r for r in ct.build_rows() if fragment in r.pedido)


def test_an_action_row_says_done_not_met():
    assert ct.DONE == "hecho (no es mejora medida)"
    assert _by_pedido("S3: reportar aparte").veredicto == ct.DONE
    assert _by_pedido("FST con las máscaras de A2").veredicto == ct.DONE


def test_the_oversegmenting_coverage_row_carries_the_computed_dice_count():
    import closing_rows_segmentation as segrows

    data = {p: json.loads((REPO / p).read_text(encoding="utf-8"))
            for p in segrows.SDIAG_ALL + tuple(segrows.STUDENT.values())}
    below = sum(segrows._below_student(data, p) for p in segrows.SDIAG_ALL)
    expected = f"cumple umbral (sobre-segmenta; Dice < student_r3 {below}/{len(segrows.SDIAG_ALL)})"
    assert _by_pedido("S-diag: cobertura").veredicto == expected


def test_the_legend_explains_every_special_verdict():
    legend = "\n".join(ct.rules())
    assert ct.DONE in legend
    assert ct.MEETS_THRESHOLD in legend


def test_the_a2_favourable_count_is_computed_not_transcribed():
    rows = {r.pedido: r for r in ct.build_rows()}
    cells = next(r for p, r in rows.items() if "A2 contra student_r3" in p)
    doc = json.loads((REPO / ct.A2_VS_STUDENT).read_text(encoding="utf-8"))
    masked = [c for c in doc["results"] if not c["cell"].endswith("__none")]
    favourable = sum(c["delta_a2_minus_student"] > 0 for c in masked)
    assert f"{favourable} de {len(masked)}" in cells.resultado


def test_a_mixed_family_of_contrasts_fails_instead_of_picking_a_side():
    import closing_rows_classifier as clf

    assert clf.family_verdict([ct.SAME, ct.SAME]) == ct.SAME
    assert clf.family_verdict([ct.SAME, ct.BETTER]) == ct.BETTER
    with pytest.raises(ValueError):
        clf.family_verdict([ct.BETTER, ct.WORSE])


def _scope(diff: int) -> dict:
    bins = {"x": {"a_count": 5, "b_count": 5 + diff, "diff_count": diff, "diff_prop": 0.0}}
    dist = {"total": {"a": 5, "b": 5 + diff}, "bins": bins}
    return {"n": {"a": 5, "b": 5}, "tone_distribution": dist, "FST_distribution": dist}


def test_the_hair_sanity_requires_every_difference_to_be_zero():
    import closing_rows_segmentation as segrows

    assert segrows.fst_sanity_clean({"scopes": {"pool": _scope(0)}}) is True
    assert segrows.fst_sanity_clean({"scopes": {"pool": _scope(1)}}) is False


def test_the_hair_free_fst_row_is_filled_from_its_files():
    import closing_rows_segmentation as segrows

    row = _by_pedido("FST con la caché sin pelo")
    doc = json.loads((REPO / segrows.FST_HAIR_CMP).read_text(encoding="utf-8"))
    pool = doc["scopes"]["pool"]
    assert row.veredicto == ct.DONE
    assert f"{pool['FST_distribution']['total']['b']:,}".replace(",", ".") in row.resultado
    assert "aísla la remoción de pelo" in row.resultado
