"""Translate presentation text only; experiment keys and scientific data stay unchanged."""
from __future__ import annotations

import re
from functools import wraps

NAMES = {
    "F1-F5 con segmentación": "Las cinco variantes con segmentación",
    "B71 sin segmentación": "Proposed sin segmentación",
    "T1–T7": "las pruebas exploratorias de tono y luminancia",
    "T1-T7": "las pruebas exploratorias de tono y luminancia",
    "F1–F5": "las cinco variantes con segmentación",
    "F1-F5": "las cinco variantes con segmentación",
    "semilla 42": "repetición 42",
    "semilla 43": "repetición 43",
    "semilla 44": "repetición 44",
    "E3b": "Prueba con segmentación en los modelos de comparación",
    "E1": "Prueba de pesos limpios",
    "E2": "Prueba de adaptación a fotos clínicas",
    "E3": "Prueba con segmentación",
    "E4": "Prueba de matrices de confusión",
    "E5": "Prueba descriptiva del tono de piel",
    "T7-aug": "Prueba de oscurecimiento (solo aumento de datos)",
    "T7": "Prueba de oscurecimiento",
    "T6": "Prueba de umbral de máscara por luminancia",
    "T4": "Prueba de constancia de color",
    "T3": "Prueba de clasificación por luminancia",
    "T2": "Prueba de segmentación por luminancia",
    "T1": "Prueba del estimador de tono",
    "S-diag": "Prueba de segmentación sin entrenamiento",
    "s42/s43/s44": "repeticiones 42, 43 y 44",
    "s42/s43": "repeticiones 42 y 43",
    "s42": "repetición 42",
    "s43": "repetición 43",
    "s44": "repetición 44",
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
_CONTEXT_NAMES = {
    "segmentation": {
        "A2-s43": "Experimento de segmentación 5, repetición 43",
        "A2": "Experimento de segmentación 5",
        "A1": "Experimento de segmentación 4",
        "S2'": "Mejora de máscaras: quitar pelo con filtro de forma",
        "S1": "Mejora de máscaras: unir regiones",
        "S2": "Mejora de máscaras: quitar pelo",
        "S3": "Mejora de máscaras: usar la confianza del tono",
        "S4": "Segmentación con DINO y cabezal entrenado",
        "S5": "Segmentación con red entrenada con pseudo-máscaras",
        "r0": "iniciales (ronda 0)",
        "r1": "de la ronda 1",
        "r2": "de la ronda 2",
    },
    "bench": {
        "S0": "lote 16",
        "S1": "lote 64",
        "S2": "lote 64 en precisión completa",
    },
    "v7": {
        "P1-NI": "Predicción de anclaje (no inferioridad estricta)",
        "P3-IC": "Predicción de objetivo (intervalo de confianza)",
        "P1": "Predicción de anclaje",
        "P2": "Predicción de tono",
        "P3": "Predicción de objetivo",
    },
}
_COMPILED: dict[str | None, tuple[re.Pattern[str], dict[str, str]]] = {}


def _vocabulary(context: str | None) -> tuple[re.Pattern[str], dict[str, str]]:
    """Global names plus the names that are only unambiguous in one kind of report."""
    if context not in _COMPILED:
        names = {**NAMES, **_CONTEXT_NAMES.get(context or "", {})}
        pattern = re.compile(
            r"(?<![\w/\\.-])(" + "|".join(re.escape(k) for k in sorted(names, key=len, reverse=True))
            + r")(?![\w/\\-]|\.\w)"
        )
        _COMPILED[context] = (pattern, names)
    return _COMPILED[context]


CODE_LABELS = {
    "A71": "Modelo de control",
    "B71": "Proposed sin segmentación",
    "public_A71": "DINOv2 público con el protocolo del modelo de control",
    "public_B71": "DINOv2 público con el protocolo de Proposed sin segmentación",
}
_CODE_SPAN = re.compile(r"`([^`]*)`")


def _label_code_spans(piece: str) -> str:
    """A code span that is exactly a known label is a display label, not a path or command."""
    return _CODE_SPAN.sub(lambda m: CODE_LABELS.get(m.group(1), m.group(0)), piece)


def display_text(text: str, *, context: str | None = None, segmentation: bool = False, code_labels: bool = False) -> str:
    """Keep literal references and executable fenced blocks byte-for-byte intact.

    ``code_labels`` also translates whole-token code spans; generated reports use it, prose does not.
    """
    lines = []
    pattern, names = _vocabulary("segmentation" if segmentation else context)
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
        if code_labels:
            pieces = [_label_code_spans(piece) if piece.startswith("`") else piece for piece in pieces]
        for index in range(0, len(pieces), 2):
            pieces[index] = pattern.sub(lambda match: names[match.group()], pieces[index])
            pieces[index] = _TABLE_PATTERN.sub(lambda match: _TABLE_NAMES[match.group()], pieces[index])
        lines.append("".join(pieces))
    return "".join(lines)


def segmentation_text(text: str) -> str:
    """Resolve A1/A2 only where those codes refer to segmentation experiments."""
    return display_text(text, context="segmentation", code_labels=True)


def segmentation_presentation(function):
    """Translate a segmentation report without modifying its input dictionary."""
    @wraps(function)
    def render(*args, **kwargs):
        return segmentation_text(function(*args, **kwargs))

    return render


def context_presentation(context: str):
    """Decorator factory: translate a generated report with the names that fit its context."""
    def decorate(function):
        @wraps(function)
        def render(*args, **kwargs):
            value = function(*args, **kwargs)
            if isinstance(value, str):
                return display_text(value, context=context, code_labels=True)
            return [display_text(item, context=context, code_labels=True) for item in value]

        return render

    return decorate


def presentation(function):
    """Apply vocabulary at the final renderer boundary, never to input records."""
    @wraps(function)
    def render(*args, **kwargs):
        value = function(*args, **kwargs)
        if isinstance(value, str):
            return display_text(value, code_labels=True)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [display_text(item, code_labels=True) for item in value]
        raise TypeError("Presentation renderers must return text or a list of text lines")

    return render
