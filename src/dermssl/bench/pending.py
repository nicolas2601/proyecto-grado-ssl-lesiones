"""Pending cells of the bench: the technical reason, in one place.

A cell is pending only while its JSON is missing: every row, matrix and contrast is derived from
the files on disk, so a stale reason cannot outlive its data. The only reason written by hand is
the HAM10000 clean-weights one, which names the missing file. No ETA is invented: ``sin ETA``.
"""

from __future__ import annotations

from dataclasses import dataclass

from dermssl import ham_clean

NO_ETA = "sin ETA"


@dataclass(frozen=True)
class Pending:
    reason: str
    eta: str = NO_ETA

    @property
    def note(self) -> str:
        return f"{self.reason}; {self.eta}"


HAM_CLEAN = Pending(f"la sensibilidad de pesos limpios sobre la val de HAM10000 (`scripts/ham_clean_run.py`) no se "
                    f"ha ejecutado: todavía no existe `{ham_clean.RESULT_JSON}`; el val de HAM ya estaba excluido del "
                    "pool original, así que no es una corrección de transductividad sino la medida de cuánto cambian "
                    "las cifras con los pesos que se publican")


def missing(rel: str) -> Pending:
    """Reason for a cell whose expected file is not in the repository."""
    return Pending(f"todavía no existe `{rel}` en el repositorio")
