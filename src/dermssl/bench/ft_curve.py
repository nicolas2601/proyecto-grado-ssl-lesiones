"""Per-epoch curve of the ISIC 2020 fine-tuning (val balanced accuracy and train loss).

Source order: ``curva_epocas`` in the result JSON, else the committed ``ft_<model>.epochs.log``,
else the curve is declared not recorded. Nothing is typed in: values come from those files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dermssl.bench.contrast import num
from dermssl.bench.rows import FT_ISIC2020_DIR, MODEL_NAMES, MODELS, _load

LOG_DIR = f"{FT_ISIC2020_DIR}/logs"
NOT_RECORDED = "curva no registrada"
EPOCH_LINE = re.compile(r"^epoca (\d+)/\d+ loss=([0-9.]+) val_bal_acc=([0-9.]+) mejor=")
RECIPE_NOTE = ("Receta existente sin cambios; con 1,8 % de positivos la bal_acc en val oscila entre épocas; "
               "se reporta la mejor época en val (optimista).")
POD_NOTE = "Esta ronda corrió en nuestro pod (GPU RTX PRO 4000), no en el banco de Karen."


@dataclass(frozen=True)
class EpochPoint:
    epoch: int
    train_loss: float
    val_balanced_accuracy: float


def parse_log(text: str) -> list[EpochPoint]:
    """Epoch points from lines ``epoca N/M loss=X val_bal_acc=Y mejor=Z (Ts)``; other lines are ignored."""
    out = []
    for line in text.splitlines():
        if match := EPOCH_LINE.match(line.strip()):
            out.append(EpochPoint(int(match[1]), float(match[2]), float(match[3])))
    return out


def curve_for(repo: Path, model: str, doc: dict | None) -> tuple[list[EpochPoint], str]:
    """(curve, source) with source in ``json`` / ``log`` / ``no registrada``."""
    if doc and doc.get("curva_epocas"):
        return [EpochPoint(c["epoca"], c["train_loss"], c["val_balanced_accuracy"]) for c in doc["curva_epocas"]], "json"
    log = repo / LOG_DIR / f"ft_{model}.epochs.log"
    if log.is_file() and (points := parse_log(log.read_text(encoding="utf-8"))):
        return points, "log"
    return [], "no registrada"


def _doc(repo: Path, model: str) -> dict | None:
    rel = f"{FT_ISIC2020_DIR}/json/results_{model}_ft_isic2020.json"
    return _load(repo, rel) if (repo / rel).is_file() else None


def _cells(points: list[EpochPoint]) -> str:
    return "; ".join(f"{p.epoch}: {num(p.val_balanced_accuracy)} / {num(p.train_loss)}" for p in points)


def section(repo: Path) -> list[str]:
    """Markdown section: notes and one line per model (curve, or why there is none)."""
    out = ["### Curva por época del fine-tuning (ISIC 2020)", "", RECIPE_NOTE, "", POD_NOTE, "",
           "Formato: época: bal_acc en val / pérdida de entrenamiento. La curva sale de `curva_epocas` del JSON o, "
           "si no existe, del log por época commiteado.", "", "| Modelo | Fuente de la curva | Curva |", "|---|---|---|"]
    for model in MODELS:
        doc = _doc(repo, model)
        points, source = curve_for(repo, model, doc)
        shown = _cells(points) if points else (NOT_RECORDED if doc else "pendiente: el fine-tuning no ha terminado")
        out.append(f"| {MODEL_NAMES[model]} | {source if points else '—'} | {shown} |")
    return out + [""]
