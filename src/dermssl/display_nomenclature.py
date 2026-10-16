"""Translate presentation text only; experiment keys and scientific data stay unchanged."""
from __future__ import annotations

import re
from functools import wraps

NAMES = {
    r"A\*/B\*": "Modelo de control / Proposed sin segmentación",
    r"A\*/B\*,": "Modelo de control / Proposed sin segmentación,",
    "B71/A71": "Proposed sin segmentación / Modelo de control",
    "A71/B71": "Modelo de control / Proposed sin segmentación",
    "B71 + F4": "Proposed con segmentación",
    "B71+F4": "Proposed con segmentación",
    "B71-limpio": "Proposed sin segmentación (pesos limpios)",
    "A71-limpio": "Modelo de control (pesos limpios)",
    "B71": "Proposed sin segmentación",
    "A71": "Modelo de control",
    "B* v7.1": "Proposed sin segmentación",
    "A* v7.1": "Modelo de control",
    "B* v7": "Proposed sin segmentación, protocolo v7",
    "A* v7": "Modelo de control, protocolo v7",
    "A*": "Modelo de control",
    "B*": "Proposed sin segmentación",
    "student_r3": "Experimento de segmentación 3",
    "student_r2": "Experimento de segmentación 2",
    "student_r1": "Experimento de segmentación 1",
    "S4-A2": "Experimento de segmentación 5",
    "S4-UNet": "Segmentador DINOv2",
    "Estudiante": "Experimento de segmentación",
    "estudiante": "experimento de segmentación",
    "student": "experimento de segmentación",
    "F4": "Proposed con segmentación (experimental)",
    "D0": "Modelo de referencia",
    "D1": "Adaptación inicial",
    "D2": "Control de entrenamiento",
    "D3": "Adaptación al nuevo conjunto",
}
NAMES.update({'experimento de segmentación DINOv2 con pseudo-máscaras CCAM, semilla 43': 'Experimento de '
                                                                            'segmentación 5, '
                                                                            'semilla 43',
 'experimento de segmentación DINOv2 con pseudo-máscaras de saliencia DINOv2': 'Experimento de '
                                                                               'segmentación 4',
 'experimento de segmentación DINOv2 con pseudo-máscaras CCAM': 'Experimento de segmentación 5',
 'experimento de segmentación DINOv2 con decoder tipo U-Net': 'Segmentador DINOv2',
 'experimento de segmentación U-Net, ronda final 3': 'Experimento de segmentación 3',
 'experimento de segmentación U-Net, ronda 2': 'Experimento de segmentación 2',
 'experimento de segmentación U-Net, ronda 1': 'Experimento de segmentación 1',
 'experimento de control SSL sin consistencia de tono, protocolo v7': 'Modelo de control, '
                                                                      'protocolo v7',
 'experimento de control SSL sin consistencia de tono': 'Modelo de control',
 'control SSL sin consistencia de tono, pesos con exclusión de validación': 'Modelo de control '
                                                                            '(pesos limpios)',
 'control SSL sin consistencia de tono': 'Modelo de control',
 'experimento SSL con consistencia de tono, protocolo v7': 'Proposed sin segmentación, protocolo '
                                                           'v7',
 'experimento SSL con consistencia de tono': 'Proposed sin segmentación',
 'Proposed sin segmentación, pesos con exclusión de validación': 'Proposed sin segmentación (pesos '
                                                                 'limpios)',
 'control de adaptación SSL en el dominio fuente': 'Adaptación inicial',
 'control de continuación SSL solo con imágenes fuente': 'Control de entrenamiento',
 'experimento de adaptación SSL con imágenes fuente y destino': 'Adaptación al nuevo conjunto',
 'referencia pública sin adaptación': 'Modelo de referencia',
 'experimento con imagen completa y recorte': 'Imagen y recorte',
 'experimento con recorte de resolución original a 448 píxeles': 'Recorte a 448 píxeles',
 'experimento con recorte de resolución original': 'Recorte original',
 'experimento con vistas guiadas por la máscara': 'Vistas de la lesión',
 'experimento con imagen, recorte y anillo perilesional': 'Imagen y piel alrededor'})
# Markdown-escaped stars have the same display meaning as literal stars.
NAMES.update({key.replace("*", r"\*"): value for key, value in list(NAMES.items()) if "*" in key})
_PATTERN = re.compile(
    r"(?<![\w/\\.-])(" + "|".join(re.escape(k) for k in sorted(NAMES, key=len, reverse=True))
    + r")(?![\w/\\-]|\.\w)"
)
_TABLE_NAMES = {
    "F1": "Imagen y recorte",
    "F2": "Recorte original",
    "F2_448": "Recorte a 448 píxeles",
    "F3": "Vistas de la lesión",
    "F5": "Imagen y piel alrededor",
}
_TABLE_PATTERN = re.compile(r"(?<=\| )(" + "|".join(_TABLE_NAMES) + r")(?= \|)")
_LITERAL = re.compile(r"(`[^`]*`|!?\[[^\]]*\]\([^)]*\)|https?://\S+|</?[A-Za-z][A-Za-z0-9-]*(?:\s+[^>]*)?/?>|<!--[^>]*-->)")
_SEGMENTATION_NAMES = {
    **NAMES,
    "A2-s43": "Experimento de segmentación 5, semilla 43",
    "A2": "Experimento de segmentación 5",
    "A1": "Experimento de segmentación 4",
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
