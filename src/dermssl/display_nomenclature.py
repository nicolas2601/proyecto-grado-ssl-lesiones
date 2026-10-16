"""Translate presentation text only; experiment keys and scientific data stay unchanged."""
from __future__ import annotations

import re
from functools import wraps

NAMES = {
    "B71/A71": "Proposed sin segmentación / control SSL sin consistencia de tono",
    "A71/B71": "control SSL sin consistencia de tono / Proposed sin segmentación",
    "B71 + F4": "Proposed con segmentación",
    "B71+F4": "Proposed con segmentación",
    "B71-limpio": "Proposed sin segmentación, pesos con exclusión de validación",
    "A71-limpio": "control SSL sin consistencia de tono, pesos con exclusión de validación",
    "B71": "Proposed sin segmentación",
    "A71": "control SSL sin consistencia de tono",
    "B* v7.1": "Proposed sin segmentación",
    "A* v7.1": "control SSL sin consistencia de tono",
    "B* v7": "experimento SSL con consistencia de tono, protocolo v7",
    "A* v7": "experimento de control SSL sin consistencia de tono, protocolo v7",
    "A*": "experimento de control SSL sin consistencia de tono",
    "B*": "experimento SSL con consistencia de tono",
    "student_r3": "experimento de segmentación U-Net, ronda final 3",
    "student_r2": "experimento de segmentación U-Net, ronda 2",
    "student_r1": "experimento de segmentación U-Net, ronda 1",
    "S4-A2": "experimento de segmentación DINOv2 con pseudo-máscaras CCAM",
    "S4-UNet": "experimento de segmentación DINOv2 con decoder tipo U-Net",
    "Estudiante": "Experimento de segmentación",
    "estudiante": "experimento de segmentación",
    "student": "experimento de segmentación",
    "F4": "Proposed con segmentación (experimental)",
    "D0": "referencia pública sin adaptación",
    "D1": "control de adaptación SSL en el dominio fuente",
    "D2": "control de continuación SSL solo con imágenes fuente",
    "D3": "experimento de adaptación SSL con imágenes fuente y destino",
}
_PATTERN = re.compile(
    r"(?<![\w/\\.-])(" + "|".join(re.escape(k) for k in sorted(NAMES, key=len, reverse=True))
    + r")(?![\w/\\-]|\.\w)"
)
_TABLE_NAMES = {
    "F1": "experimento con imagen completa y recorte",
    "F2": "experimento con recorte de resolución original",
    "F2_448": "experimento con recorte de resolución original a 448 píxeles",
    "F3": "experimento con vistas guiadas por la máscara",
    "F5": "experimento con imagen, recorte y anillo perilesional",
}
_TABLE_PATTERN = re.compile(r"(?<=\| )(" + "|".join(_TABLE_NAMES) + r")(?= \|)")
_LITERAL = re.compile(r"(`[^`]*`|!?\[[^\]]*\]\([^)]*\)|https?://\S+|<[^>]*>)")
_SEGMENTATION_NAMES = {
    **NAMES,
    "A2-s43": "experimento de segmentación DINOv2 con pseudo-máscaras CCAM, semilla 43",
    "A2": "experimento de segmentación DINOv2 con pseudo-máscaras CCAM",
    "A1": "experimento de segmentación DINOv2 con pseudo-máscaras de saliencia DINOv2",
}
_SEGMENTATION_PATTERN = re.compile(
    r"(?<![\w/\\.-])(" + "|".join(re.escape(k) for k in sorted(_SEGMENTATION_NAMES, key=len, reverse=True))
    + r")(?![\w/\\-]|\.\w)"
)


def display_text(text: str, *, segmentation: bool = False) -> str:
    """Keep literal references and executable fenced blocks byte-for-byte intact."""
    lines = []
    pattern = _SEGMENTATION_PATTERN if segmentation else _PATTERN
    names = _SEGMENTATION_NAMES if segmentation else NAMES
    fenced = False
    for line in text.splitlines(keepends=True):
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
            lines.append(line)
            continue
        if fenced:
            lines.append(line)
            continue
        pieces = _LITERAL.split(line)
        for index in range(0, len(pieces), 2):
            pieces[index] = pattern.sub(lambda match: names[match.group()], pieces[index])
            pieces[index] = _TABLE_PATTERN.sub(lambda match: _TABLE_NAMES[match.group()], pieces[index])
        lines.append("".join(pieces))
    return "".join(lines)


def segmentation_text(text: str) -> str:
    """Resolve A1/A2 only where those codes refer to segmentation experiments."""
    return display_text(text, segmentation=True)


def segmentation_presentation(function):
    """Translate a segmentation report without modifying its input dictionary."""
    @wraps(function)
    def render(*args, **kwargs):
        return segmentation_text(function(*args, **kwargs))

    return render


def presentation(function):
    """Apply vocabulary at the final renderer boundary, never to input records."""
    @wraps(function)
    def render(*args, **kwargs):
        value = function(*args, **kwargs)
        if isinstance(value, str):
            return display_text(value)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [display_text(item) for item in value]
        raise TypeError("Presentation renderers must return text or a list of text lines")

    return render
