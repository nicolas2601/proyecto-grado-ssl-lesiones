"""The PAD phototype table computes its own caveat (Paula, 2026-09-28).

III-VI is mostly type III in PAD, so the grouped bin does not stand for dark
skin. The note is derived from the counts, never typed by hand: a transcribed
84 % would survive a change of data unnoticed.
"""

from __future__ import annotations

from dermssl import pad_ufes as pu

#: Per-image counts of the published metadata, as the pod diagnosis reported them.
PUBLISHED_LIKE = [1] * 153 + [2] * 876 + [3] * 392 + [4] * 62 + [5] * 10 + [6] * 1 + [None] * 804


def test_composition_counts_every_type_and_the_blanks():
    comp = pu.fst_composition(PUBLISHED_LIKE)
    assert comp["by_type"] == {1: 153, 2: 876, 3: 392, 4: 62, 5: 10, 6: 1}
    assert comp["n_missing"] == 804
    assert comp["iii_vi_total"] == 465
    assert round(comp["iii_share_of_iii_vi"], 3) == 0.843


def test_the_note_is_computed_from_the_counts():
    note = pu.fst_note(pu.fst_composition(PUBLISHED_LIKE))
    assert "84 % tipo III (392/465)" in note
    assert "NO representa piel oscura" in note
    assert "IV (62), V (10) y VI (1)" in note
    assert pu.THESIS_FST_SENTENCE in note
    assert pu.THESIS_FST_SENTENCE == "PAD-UFES-20 no permite concluir sobre FST V-VI"


def test_other_counts_give_another_note():
    note = pu.fst_note(pu.fst_composition([3, 3, 4, 4, 5, 6, 1]))
    assert "33 % tipo III (2/6)" in note


def test_the_table_reports_the_three_bins_and_iv_to_vi_as_descriptive():
    table = pu.fst_table_markdown(pu.fst_composition(PUBLISHED_LIKE))
    assert "| I | 153 |" in table and "| II | 876 |" in table and "| III-VI | 465 |" in table
    for label, n in (("IV", 62), ("V", 10), ("VI", 1)):
        assert f"| {label} (descriptivo, sin veredicto) | {n} |" in table
    assert f"| {pu.FST_MISSING} | 804 |" in table
    assert pu.fst_note(pu.fst_composition(PUBLISHED_LIKE)) in table


def test_the_manifest_summary_carries_the_note(tmp_path):
    records = [pu.PadRecord(f"i{k}", f"p{k}", "1", "BCC", "bcc", v, "ok" if v else "missing")
               for k, v in enumerate([3, 3, 3, 4, None])]
    summary = pu.fst_summary(records)
    assert "75 % tipo III (3/4)" in summary["note"]
    assert summary["composition"]["by_type"][4] == 1
