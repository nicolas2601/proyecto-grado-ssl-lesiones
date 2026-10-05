"""Table model: Markdown pipes are escaped, LaTeX is booktabs and switches to longtable for long tables."""

from __future__ import annotations

from dermssl.bench import tables as T

SMALL = T.Table("Δ de prueba", ["A", "B"], [["`x|y`", "0,1000 [+0,0100; −0,2000]"]], "nota", "demo")


def test_markdown_escapes_pipes_inside_cells():
    lines = T.markdown(SMALL)
    assert lines[0] == "### Δ de prueba" and "nota" in lines
    assert r"`x\|y`" in lines[-2] and lines[-2].count("|") - lines[-2].count(r"\|") == 3


def test_latex_small_table_is_booktabs_with_unicode_and_textbar():
    tex = "\n".join(T.latex(SMALL))
    assert r"\toprule" in tex and r"\bottomrule" in tex and r"\resizebox" in tex
    assert r"\textbar{}" in tex and r"$\Delta$" in tex and "$-$" in tex and r"\label{tab:demo}" in tex


def test_latex_long_table_uses_longtable():
    big = T.Table("grande", ["A"], [[str(i)] for i in range(T.LONG_ROWS + 1)], "", "big")
    tex = "\n".join(T.latex(big))
    assert r"\begin{longtable}" in tex and r"\begin{table}" not in tex
