"""Pending cells of the bench: the technical reason and the declared ETA, in one place.

The ETAs are declared by the operator, they are not measured from the repository. A pending
row turns into a measured row by itself when its JSON appears; nothing here is a number.
"""

from __future__ import annotations

from dataclasses import dataclass

ETA_SOURCE = "ETA declarada por el operador el 2026-10-05; no se mide desde el repositorio"


@dataclass(frozen=True)
class Pending:
    reason: str
    eta: str

    @property
    def note(self) -> str:
        return f"{self.reason}; ETA {self.eta}"


E2 = Pending("los pesos de D2 y D3 (GPU) y la única lectura de `target_test` de PAD no han terminado; el split "
             "de PAD ya existe (`resultados/banco_completo/pad_da/target_split.json`)", "~10:30 UTC")
A71_CLEAN = Pending("falta A* s43 limpio (GPU0 en curso); A71-limpio es la sopa de A* s42 + s43 limpios", "~15:00 UTC")
ORIGINAL_CANONICAL = Pending("re-extracción de B71/A71 originales con el cómputo canónico (batch 64 + autocast) "
                             "en GPU0 después de A* s43; hoy solo existe la corrida de batch 16", "~15:00 UTC")
FT_ISIC2020 = Pending("el fine-tuning de los 8 baselines en ISIC 2020 corre en GPU1 después de E2",
                      "inicio al terminar E2 (~10:30 UTC); duración no medida")
HAM_CLEAN = Pending("E1 de B71-limpio en HAM10000 todavía no se corrió (no hay JSON en el repositorio)",
                    "no declarada por el operador")
CLEAN_VS_ORIGINAL = Pending("el contraste B71-limpio contra B71 original exige el original con el mismo cómputo "
                            "canónico; la corrida de batch 16 no es comparable (cambia el cómputo, no solo los "
                            "datos)", ORIGINAL_CANONICAL.eta)
