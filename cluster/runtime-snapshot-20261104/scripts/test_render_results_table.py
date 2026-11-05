from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

ROWS = [
    {'name': 'SimSiam sobre DINOv2 (base)', 'internal': 'baseline-current-recipe', 'type': 'SSL puro',
     'pad_macro_f1': 0.6063, 'ham_macro_f1': 0.5140, 'delta_pad': None, 'gpu_h': None, 'seeds': '42', 'status': 'referencia', 'section': 'SSL puro'},
    {'name': 'Recorte aleatorio escala 0,4', 'internal': 'crop-scale-0.4', 'type': 'SSL puro',
     'pad_macro_f1': 0.5911, 'ham_macro_f1': 0.5068, 'delta_pad': (-0.0152, -0.0277, -0.0043),
     'gpu_h': 1.1328, 'seeds': '42', 'status': 'evaluado', 'section': 'SSL puro'},
]


def test_delta_cell_shows_point_and_paired_interval():
    from render_results_table import delta_cell
    assert delta_cell((-0.0152, -0.0277, -0.0043)) == '−0,015 [−0,028; −0,004]'


def test_missing_delta_marks_reference_row():
    from render_results_table import delta_cell
    assert delta_cell(None) == 'referencia'


def test_markdown_lists_every_row_with_clear_name_and_internal_column():
    from render_results_table import to_markdown
    md = to_markdown(ROWS)
    assert '| SimSiam sobre DINOv2 (base) | baseline-current-recipe |' in md
    assert '| Recorte aleatorio escala 0,4 | crop-scale-0.4 |' in md
    assert '−0,015 [−0,028; −0,004]' in md


def test_json_roundtrip_keeps_all_rows(tmp_path):
    from render_results_table import write_json
    out = tmp_path / 'tabla.json'
    write_json(ROWS, out)
    assert json.loads(out.read_text())['rows'][1]['internal'] == 'crop-scale-0.4'


def test_rejects_row_without_pad_score():
    from render_results_table import to_markdown
    try:
        to_markdown([{**ROWS[0], 'pad_macro_f1': None}])
    except ValueError as error:
        assert 'pad_macro_f1' in str(error)
    else:
        raise AssertionError('row without PAD macro-F1 must be rejected')


def test_verdict_is_green_only_when_interval_excludes_zero_above():
    from render_results_table import verdict
    assert verdict((0.02, 0.005, 0.03)) == 'mejora'
    assert verdict((-0.02, -0.03, -0.005)) == 'empeora'
    assert verdict((0.01, -0.004, 0.02)) == 'sin evidencia'
    assert verdict(None) == 'referencia'


def test_markdown_groups_rows_by_section_in_fixed_order_with_footer():
    from render_results_table import to_markdown, FOOTER
    rows = [{**ROWS[1], 'section': 'supervisado', 'internal': 'sup-x', 'name': 'Lector supervisado X'},
            {**ROWS[1], 'section': 'SSL puro'}]
    md = to_markdown(rows)
    assert md.index('SSL puro') < md.index('supervisado')
    assert md.rstrip().endswith(FOOTER)


def test_sections_outside_the_fixed_list_are_rejected():
    from render_results_table import to_markdown
    try:
        to_markdown([{**ROWS[1], 'section': 'otra'}])
    except ValueError as error:
        assert 'section' in str(error)
    else:
        raise AssertionError('unknown section must be rejected')


def test_markdown_has_separate_ham_delta_column():
    from render_results_table import to_markdown
    row = {**ROWS[1], 'delta_ham': (0.0331, 0.0041, 0.0660)}
    md = to_markdown([row])
    assert '| +0,033 [+0,004; +0,066] |' in md
