#!/usr/bin/env python3
"""Genera ``notebooks/proyecto_ssl_lesiones_v5.ipynb``.

Por qué se genera en vez de editar un notebook a mano
-----------------------------------------------------
Un notebook de Kaggle debe ser autocontenido: la sesión no puede importar
``src/dermssl`` desde este repositorio. Copiar la lógica al notebook provoca
que se aleje inmediatamente de la copia probada.

Por eso el notebook se *construye*: el núcleo solo-Numpy de ``src/dermssl/`` se
inserta literalmente en una celda y las celdas dependientes de torch provienen
de ``scripts/notebook_cells_v5.py``. El código que corre en Kaggle coincide con
el cubierto por ``tests/``; el notebook es un artefacto de construcción y no se
edita a mano.

Uso
---
    python scripts/build_notebook_v5.py            # escribe el notebook
    python scripts/build_notebook_v5.py --check    # verifica que esté actualizado
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
    "# {name}.py  --  insertado desde src/dermssl/{name}.py (no editar aquí)\n"
    "# ============================================================================"
)


def module_source(name: str) -> str:
    """Lee un módulo y descarta la importación ``__future__``.

    Dos motivos por los que no puede sobrevivir a la inserción:

    1. ``from __future__ import annotations`` debe ser la *primera* sentencia de
       una unidad de compilación, y una celda de notebook es una unidad: varios
       módulos insertados no pueden contenerla individualmente.
    2. Convierte las anotaciones en strings y ``@dataclass`` las resuelve al
       buscar el módulo definidor en ``sys.modules``. Esa búsqueda no está
       garantizada dentro de una celda de notebook (ni en un ``exec``).

    Descartarla es seguro: cada anotación usada aquí (``Path | None``,
    ``dict[str, Path]``) es sintaxis nativa desde Python 3.10, y Kaggle usa 3.10+.
    """
    text = (SRC / "dermssl" / f"{name}.py").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() != "from __future__ import annotations"]
    return "\n".join(lines).strip()


def build_core_cell() -> str:
    parts = [
        "# Núcleo insertado de src/dermssl/: fuente única de verdad, cubierto por tests/.",
        "# Regenerar con: python scripts/build_notebook_v5.py",
        "# NOTA: se omite 'from __future__ import annotations' intencionalmente; ver module_source().",
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
    """Ensamblar el notebook: encabezado, configuración, núcleo insertado y pipeline."""
    ordered = list(CELLS)
    # El núcleo debe definirse después de la celda de configuración y antes de usarla.
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
    """Compila cada celda de código. Un error de sintaxis implica fallo de construcción.

    Es la comprobación más económica de que el notebook es ejecutable: no puede
    detectar errores lógicos (lo hacen las pruebas unitarias), pero garantiza que
    Kaggle no falle en la celda 1 con SyntaxError tras una cola de 40 minutos.
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
        help="falla si el notebook versionado difiere de una construcción nueva",
    )
    args = parser.parse_args()

    notebook = build_notebook()
    problems = validate(notebook)
    if problems:
        print("FALLÓ LA CONSTRUCCIÓN: errores de sintaxis en celdas generadas:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    payload = serialize(notebook)
    if args.check:
        if not OUT.is_file():
            print(f"falta {OUT}; ejecute sin --check", file=sys.stderr)
            return 1
        if OUT.read_text(encoding="utf-8") != payload:
            print(f"{OUT} está desactualizado; ejecute: python scripts/build_notebook_v5.py", file=sys.stderr)
            return 1
        print(f"CORRECTO: {OUT.name} está actualizado ({len(notebook['cells'])} celdas)")
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
