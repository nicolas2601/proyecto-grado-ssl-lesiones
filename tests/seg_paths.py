"""Make the hyphen-named plan-segmentacion folders importable from tests.

The folder names contain hyphens, so they are not Python packages. Each folder
holds plain, uniquely named modules; this helper just puts them on sys.path.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plan-segmentacion"
COMUN = PLAN / "comun"
CAMINO_B = PLAN / "camino-b-cnn"

SEG_DIRS = (
    COMUN / "nucleo",
    COMUN / "datos",
    COMUN / "estudiante",
    COMUN / "evaluacion",
    CAMINO_B / "preentrenamiento-contrastivo",
    CAMINO_B / "pseudo-mascaras",
    CAMINO_B / "entrenamiento",
)

for _dir in reversed(SEG_DIRS):
    if str(_dir) not in sys.path:
        sys.path.insert(0, str(_dir))
