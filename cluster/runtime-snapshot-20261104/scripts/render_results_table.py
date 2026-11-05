"""Master SSL results table (MD, JSON, PNG). Δ columns: seed 20261025 with paired IC95 AND mean of 3 CV seeds (no IC)."""
import json
from pathlib import Path
import sys
import textwrap

SECTIONS = ["SSL puro", "SSL con muestreo guiado por etiquetas", "supervisado", "imagen-texto y +metadatos", "clásico", "semi-supervisado", "no factible"]
FOOTER = "Una semilla SSL; bootstrap condicional; sin validación clínica."
COLORS = {"mejora": "#c8e6c9", "empeora": "#ffcdd2", "sin evidencia": "#eeeeee", "referencia": "#ffffff"}
HEADERS = ["Método", "Tipo", "Semillas", "PAD F1 / AUC", "Δ PAD media 3 CV [IC95]", "Δ PAD semilla 20261025 (puntual)", "HAM F1", "Δ HAM media 3 CV [IC95]", "Δ HAM semilla 20261025 (puntual)", "GPU-h", "Estado"]
WIDTHS = [0.16, 0.08, 0.07, 0.07, 0.10, 0.08, 0.07, 0.10, 0.08, 0.06, 0.13]


def _num(value):
    return f'{value:.3f}'.replace('.', ',')


def _signed(value):
    return ('−' if value < 0 else '+') + f'{abs(value):.3f}'.replace('.', ',')


def delta_cell(delta):
    if delta is None:
        return 'referencia'
    point, low, high = delta
    return f'{_signed(point)} [{_signed(low)}; {_signed(high)}]'


def mean_cell(value):
    return 'referencia' if value is None else _signed(value)


def verdict(delta):
    if delta is None:
        return 'referencia'
    point, low, high = delta
    if low > 0:
        return 'mejora'
    if high < 0:
        return 'empeora'
    return 'sin evidencia'


def _check_section(rows):
    unknown = {row.get('section') for row in rows} - set(SECTIONS)
    if unknown:
        raise ValueError(f'unknown section {sorted(map(str, unknown))}')


def _metric(value, measured_flag_text):
    if isinstance(value, (int, float)):
        return _num(value)
    return measured_flag_text


NOT_RUN_SECTIONS = {'semi-supervisado', 'no factible'}


def _missing_text(row):
    """'no ejecutado' = never run; 'pendiente' = run exists but figure still to compute."""
    never_run = row.get('section') in NOT_RUN_SECTIONS or row['status'].startswith(('no ejecutado', 'no factible'))
    return 'no ejecutado' if never_run else 'pendiente'


def _cells(row):
    pad, ham = row.get('pad_macro_f1'), row.get('ham_macro_f1')
    gpu = row.get('gpu_h')
    gpu_text = _missing_text(row) if gpu is None else ('≈ ' if 'estimad' in row['status'] else '') + f"{gpu:.2f}".replace('.', ',')
    ham_text = _metric(ham, 'no medido en HAM' if pad is not None else _missing_text(row))
    return [row['name'], row['type'], row.get('seeds', '—'),
            _metric(pad, _missing_text(row)), delta_cell(row.get('delta_pad')), mean_cell(row.get('delta_pad_mean')),
            ham_text, delta_cell(row.get('delta_ham')), mean_cell(row.get('delta_ham_mean')),
            gpu_text, row['status']]


def to_markdown(rows):
    _check_section(rows)
    header = '| ' + ' | '.join(HEADERS) + ' |'
    sep = '|' + '---|' * len(HEADERS)
    lines = []
    for section in SECTIONS:
        group = [row for row in rows if row.get('section') == section]
        if not group:
            continue
        lines += ['', f'### {section}', '', header, sep]
        lines += ['| ' + ' | '.join(_cells(row)) + ' |' for row in group]
    return '\n'.join(lines).lstrip('\n') + f'\n\n{FOOTER}\n'


def write_json(rows, path):
    Path(path).write_text(json.dumps({'rows': rows}, indent=2, ensure_ascii=False) + '\n')


LINE_IN = 0.135
PAD_IN = 0.08
SECTION_IN = 0.34
CHAR_IN = 0.075
FIG_W_IN = 18.0


def _wrap_cell(text, col):
    width = max(6, int(WIDTHS[col] * FIG_W_IN * 0.98 / CHAR_IN))
    return textwrap.fill(str(text), width=width).split('\n')


def _layout(rows):
    blocks = [('header', [_wrap_cell(text, i) for i, text in enumerate(HEADERS)])]
    for section in SECTIONS:
        group = [row for row in rows if row.get('section') == section]
        if not group:
            continue
        blocks.append(('section', section))
        for row in group:
            cells = [_wrap_cell(cell, i) for i, cell in enumerate(_cells(row))]
            blocks.append(('row', (row, cells)))
    return blocks


def _block_height(kind, payload):
    if kind == 'section':
        return SECTION_IN
    lines = max(len(cell) for cell in payload[1]) if kind == 'row' else max(len(cell) for cell in payload)
    return lines * LINE_IN + PAD_IN


def write_png_table(rows, path, notes=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _check_section(rows)
    assert abs(sum(WIDTHS) - 1.0) < 1e-9
    xs = [sum(WIDTHS[:i]) for i in range(len(WIDTHS))]
    blocks = _layout(rows)
    note_lines = [textwrap.fill(note, width=280).split('\n') for note in (FOOTER, *notes)]
    heights = [_block_height(kind, payload) for kind, payload in blocks]
    total = sum(heights) + 0.3 + LINE_IN * sum(len(lines) for lines in note_lines) + 0.3
    fig = plt.figure(figsize=(FIG_W_IN, total))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(total, 0)
    y = 0.15
    for (kind, payload), h in zip(blocks, heights):
        if kind == 'header':
            for i, lines in enumerate(payload):
                ax.text(xs[i] + 0.003, y, '\n'.join(lines), fontsize=8, weight='bold', va='top', linespacing=1.2)
            ax.plot([0, 1], [y + h, y + h], color='black', lw=0.8)
        elif kind == 'section':
            ax.text(0.003, y + 0.02, payload, fontsize=9, weight='bold', color='#1a237e', va='top')
        else:
            row, cells = payload
            for col, key in ((4, 'delta_pad'), (7, 'delta_ham')):
                colour = COLORS[verdict(row.get(key))]
                ax.add_patch(plt.Rectangle((xs[col], y), WIDTHS[col], h, color=colour, zorder=0))
            for i, lines in enumerate(cells):
                ax.text(xs[i] + 0.003, y + PAD_IN / 2, '\n'.join(lines), fontsize=7.5, va='top', zorder=1, linespacing=1.2)
        y += h
    y += 0.3
    ax.plot([0, 1], [y, y], color='black', lw=0.5)
    for lines in note_lines:
        ax.text(0.003, y + 0.05, '\n'.join(lines), fontsize=7.5, va='top', style='italic', linespacing=1.2)
        y += LINE_IN * len(lines)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(registry, outdir, notes=()):
    data = json.loads(Path(registry).read_text())
    rows = data['rows']
    notes = list(notes) + list(data.get('notes', []))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tabla-maestra.md').write_text(to_markdown(rows))
    write_json(rows, out / 'tabla-maestra.json')
    write_png_table(rows, out / 'tabla-maestra.png', notes=notes)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
