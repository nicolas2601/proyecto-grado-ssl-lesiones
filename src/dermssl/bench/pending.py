"""Pending cells of the bench: the technical reason, in one place.

A cell is pending only while its JSON is missing: every row, matrix and contrast is derived from
the files on disk, so a stale reason cannot outlive its data. The only reason written by hand is
the HAM10000 clean-weights one, a decision and not a wait. No ETA is invented: ``sin ETA``.
"""

from __future__ import annotations

from dataclasses import dataclass

NO_ETA = "sin ETA"


@dataclass(frozen=True)
class Pending:
    reason: str
    eta: str = NO_ETA

    @property
    def note(self) -> str:
        return f"{self.reason}; {self.eta}"


HAM_CLEAN = Pending("no se evaluaron los pesos limpios sobre la val/holdout de HAM10000: el val de HAM ya estaba "
                    "excluido del pool, por lo que no hay transductividad que corregir; el costo/beneficio no "
                    "justificó una corrida más antes del corte")


def missing(rel: str) -> Pending:
    """Reason for a cell whose expected file is not in the repository."""
    return Pending(f"todavía no existe `{rel}` en el repositorio")
