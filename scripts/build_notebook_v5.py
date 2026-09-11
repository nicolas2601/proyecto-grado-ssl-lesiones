#!/usr/bin/env python3
"""Generate ``notebooks/proyecto_ssl_lesiones_v5.ipynb``.

Why generate instead of editing a notebook by hand
--------------------------------------------------
A Kaggle notebook must be self-contained: the session cannot import
``src/dermssl`` from this repository. The naive workaround is to copy the logic
into the notebook, which immediately drifts away from the tested copy.

So the notebook is *built*: the numpy-only core in ``src/dermssl/`` is inlined
verbatim into one cell, and the torch-dependent cells come from
``scripts/notebook_cells_v5.py``. Result: the code that runs on Kaggle is exactly
the code covered by ``tests/``, and the notebook is a build artefact - never
edited by hand.

Usage
-----
    python scripts/build_notebook_v5.py            # write the notebook
    python scripts/build_notebook_v5.py --check    # verify it is up to date
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
OUT = REPO / "notebooks" / "proyecto_ssl_lesiones_v5.ipynb"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dermssl import MODULE_ORDER  # noqa: E402
from notebook_cells_v5 import CELLS  # noqa: E402

CORE_MD = """## 0. Nucleo reutilizable (generado, no editar aqui)

Las funciones de abajo son **copia exacta** de `src/dermssl/` en este repositorio, donde
estan cubiertas por `tests/` (83 pruebas que corren sin GPU, sin Kaggle y sin internet).

Se pegan aca porque una sesion de Kaggle no puede importar codigo del repositorio. El
notebook se genera con `python scripts/build_notebook_v5.py`, de modo que **el codigo que
corre aca y el codigo probado son el mismo codigo**. Para cambiar algo: se edita el modulo,
se corre `pytest`, y se regenera el notebook.

Que hay aca:

| Modulo | Responsabilidad |
|---|---|
| `patch_guard` | evita la trampa de resolucion vs tamano de parche (256 no es multiplo de 14) |
| `data_resolver` | localiza los datasets en Kaggle, por kagglehub o en disco local |
| `label_regimes` | muestreo de regimenes de pocas etiquetas, anidados y sin clases vacias |
| `metrics` | macro-F1, AUC, ECE, bootstrap - implementados en numpy y verificados a mano |
| `probes` | kNN, prototipos y linear probe sobre features congeladas |
| `feature_cache` | cache en disco de las features (resume + rejilla de experimentos barata) |
"""

BANNER = (
    "# ============================================================================\n"
    "# {name}.py  --  inlined from src/dermssl/{name}.py (do not edit here)\n"
    "# ============================================================================"
)


def module_source(name: str) -> str:
    """Read one module, dropping the ``__future__`` import and any
    intra-package ``dermssl`` import.

    Two reasons the ``__future__`` import cannot survive inlining:

    1. ``from __future__ import annotations`` must be the *first* statement of a
       compilation unit, and a notebook cell is one unit - several inlined
       modules cannot each carry it.
    2. It turns annotations into strings, and ``@dataclass`` then resolves them
       by looking the defining module up in ``sys.modules``. Inside a notebook
       cell (or any ``exec``) that lookup is not guaranteed to work.

    Dropping it is safe: every annotation used here (``Path | None``,
    ``dict[str, Path]``) is native syntax from Python 3.10 onwards, and Kaggle
    runs 3.10+.

    ``from dermssl...`` imports (e.g. ``fitzpatrick.py``'s
    ``from dermssl.pretrain import lesion_guided``) are dropped for a
    different reason: on real Kaggle ``src/dermssl`` is never installed, so
    that import raises ``ModuleNotFoundError`` in the inlined cell - that is
    the whole reason this inlining mechanism exists. Dropping it is only safe
    because ``MODULE_ORDER`` guarantees the dependency (e.g.
    ``pretrain/lesion_guided``) is inlined into the same cell *before* any
    module that needs it, so the names it defines (e.g. ``lesion_mask``) are
    already plain names in the shared exec namespace by the time this
    module's code runs. Callers must therefore use bare names
    (``lesion_mask(...)``), not the package-qualified form
    (``lesion_guided.lesion_mask(...)``).
    """
    text = (SRC / "dermssl" / f"{name}.py").read_text(encoding="utf-8")
    lines = [
        ln
        for ln in text.splitlines()
        if ln.strip() != "from __future__ import annotations"
        and not ln.lstrip().startswith(("from dermssl", "import dermssl"))
    ]
    return "\n".join(lines).strip()


def build_core_cell() -> str:
    parts = [
        "# Inlined core of src/dermssl/ -- single source of truth, covered by tests/.",
        "# Regenerate with: python scripts/build_notebook_v5.py",
        "# NOTE: no 'from __future__ import annotations' on purpose - see module_source().",
        "",
    ]
    for name in MODULE_ORDER:
        parts.extend([BANNER.format(name=name), "", module_source(name), "", ""])
    parts.append(
        'print("nucleo dermssl inlined:", '
        f'{list(MODULE_ORDER)!r})'
    )
    return "\n".join(parts)


def to_cell(cell_type: str, source: str, index: int) -> dict:
    lines = source.rstrip("\n").splitlines(keepends=True)
    cell = {
        "cell_type": cell_type,
        "id": f"cell-{index:02d}",
        "metadata": {},
        "source": lines,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build_notebook() -> dict:
    """Assemble the notebook: header, config, inlined core, then the pipeline."""
    ordered = list(CELLS)
    # The core must be defined after the config cell and before any cell using it.
    ordered.insert(2, ("markdown", CORE_MD))
    ordered.insert(3, ("code", build_core_cell()))

    cells = [to_cell(kind, src, i) for i, (kind, src) in enumerate(ordered)]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate(notebook: dict) -> list[str]:
    """Compile every code cell. A syntax error here is a build failure.

    This is the cheapest possible proof that the notebook is runnable: it cannot
    catch logic errors (the unit tests do that) but it guarantees Kaggle will not
    die on cell 1 with a SyntaxError after a 40-minute queue.
    """
    problems = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        try:
            compile(source, f"<cell {i}>", "exec")
        except SyntaxError as exc:
            problems.append(f"cell {i} ({cell['id']}): {exc}")
    return problems


def serialize(notebook: dict) -> str:
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed notebook differs from a fresh build",
    )
    args = parser.parse_args()

    notebook = build_notebook()
    problems = validate(notebook)
    if problems:
        print("BUILD FAILED - syntax errors in generated cells:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    payload = serialize(notebook)
    if args.check:
        if not OUT.is_file():
            print(f"{OUT} is missing; run without --check", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != payload:
            print(f"{OUT} is stale; run: python scripts/build_notebook_v5.py", file=sys.stderr)
            return 1
        print(f"OK - {OUT.name} is up to date ({len(notebook['cells'])} cells)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload, encoding="utf-8")
    code_cells = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    print(
        f"wrote {OUT.relative_to(REPO)} | {len(notebook['cells'])} cells "
        f"({code_cells} code) | all code cells compile"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
