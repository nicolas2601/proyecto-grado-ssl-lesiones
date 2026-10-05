"""Orden de celdas para el notebook v5.

Único lugar que decide la estructura del notebook: prosa de
``notebook_prose_v5`` y celdas ejecutables de ``notebook_code_v5``. El núcleo
insertado de ``src/dermssl/`` se inyecta desde ``build_notebook_v5.py``.
"""

from notebook_code_v5 import (
    BACKBONE_CODE,
    CONFIG_CODE,
    DATA_CODE,
    EXTRACT_CODE,
    PROBE_CODE,
    REPORT_CODE,
    SPLIT_CODE,
)
from notebook_prose_v5 import (
    BACKBONE_MD,
    DATA_MD,
    EXTRACT_MD,
    HEADER_MD,
    PROBE_MD,
    REPORT_MD,
    SPLIT_MD,
)

CELLS = [
    ("markdown", HEADER_MD),
    ("code", CONFIG_CODE),
    ("markdown", DATA_MD),
    ("code", DATA_CODE),
    ("markdown", SPLIT_MD),
    ("code", SPLIT_CODE),
    ("markdown", BACKBONE_MD),
    ("code", BACKBONE_CODE),
    ("markdown", EXTRACT_MD),
    ("code", EXTRACT_CODE),
    ("markdown", PROBE_MD),
    ("code", PROBE_CODE),
    ("markdown", REPORT_MD),
    ("code", REPORT_CODE),
]
