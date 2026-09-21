"""Mitigación de etapa 2: transformador LAB estilo CIRCLe, muestreo de
desplazamiento FST y política de mezcla sintética al nivel del probe entrenable.

Este módulo aporta la mecánica de la etapa 2, no evidencia de equidad: una
transformación determinista de color, un muestreador FST acotado y una política
de mezcla. :func:`should_run_stage2` impide usarlo con datos reales hasta que
la etapa 1 informe una brecha medible. Se usa solo NumPy para conservar el
notebook autocontenido sin dependencias opcionales en tiempo de ejecución.
"""

from __future__ import annotations

import numpy as np

from dermssl.fitzpatrick import (
    FST_THRESHOLDS,
    MAX_MASK_AREA_FRACTION,
    MIN_MASK_AREA_FRACTION,
    srgb_to_lab,
)
from dermssl.gates import FAIL
from dermssl.mask_manifest import mask_sha256

# Máximo desplazamiento FST que puede solicitar una muestra sintética.
MAX_FST_SHIFT = 2

# Unidades L* desplazadas por nivel FST; un valor heurístico para hacer visible
# un cambio de bucket sin sacar L* de [0, 100] en una muestra típica.
FST_SHIFT_L_STEP = 8.0

# Iluminante D65 y observador de 2 grados: deben coincidir bit a bit con las
# constantes privadas de fitzpatrick.py para que lab_to_srgb siga siendo inversa.
_D65_XN = 0.95047
_D65_YN = 1.0
_D65_ZN = 1.08883

_SRGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)


def _f_inv(t: np.ndarray) -> np.ndarray:
    """Inversa de ``_f`` de fitzpatrick.py con el mismo punto de corte CIE Lab."""
    cube = t**3
    linear = (t - 16.0 / 116.0) / 7.787
    return np.where(cube > 0.008856, cube, linear)


def _gamma_encode(linear: np.ndarray) -> np.ndarray:
    """Convierte luz lineal [0, 1] a sRGB [0, 1]; inversa de ``_inverse_gamma``."""
    low = linear * 12.92
    high = 1.055 * np.power(np.clip(linear, 1e-12, None), 1.0 / 2.4) - 0.055
    return np.where(linear <= 0.0031308, low, high)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convierte CIE Lab a sRGB, con flotantes en [0, 1] y forma (..., 3).

    La salida se limita al rango válido porque no toda terna Lab se representa
    como un color sRGB.
    """
    lab = np.asarray(lab, dtype=float)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    xr, yr, zr = _f_inv(fx), _f_inv(fy), _f_inv(fz)
    xyz = np.stack([xr * _D65_XN, yr * _D65_YN, zr * _D65_ZN], axis=-1)

    linear = xyz @ _XYZ_TO_SRGB.T
    linear = np.clip(linear, 0.0, None)  # Protege de ruido flotante negativo.
    srgb = _gamma_encode(linear)
    return np.clip(srgb, 0.0, 1.0)


def clamp_fst_shift(requested_shift: int, *, max_shift: int = MAX_FST_SHIFT) -> int:
    """Limita un desplazamiento FST solicitado a ``[-max_shift, max_shift]``."""
    if requested_shift > max_shift:
        return max_shift
    if requested_shift < -max_shift:
        return -max_shift
    return requested_shift


def target_fst_bucket(
    source_bucket: int, requested_shift: int, *, max_shift: int = MAX_FST_SHIFT
) -> int:
    """Devuelve el bucket FST (1..6) tras limitar el desplazamiento solicitado."""
    shift = clamp_fst_shift(requested_shift, max_shift=max_shift)
    n_buckets = len(FST_THRESHOLDS) + 1
    return int(min(max(source_bucket + shift, 1), n_buckets))


def _is_degenerate_mask(mask: np.ndarray) -> bool:
    """Usa los mismos límites que ``_is_degenerate`` de fitzpatrick.py."""
    area = float(mask.mean())
    return not (MIN_MASK_AREA_FRACTION <= area <= MAX_MASK_AREA_FRACTION)


def shift_skin_tone(
    img: np.ndarray, mask: np.ndarray | None, shift_buckets: int, *, max_shift: int = MAX_FST_SHIFT
) -> np.ndarray:
    """Desplaza de forma determinista L* de la piel por niveles FST.

    Los píxeles de lesión (``mask`` True) se conservan píxel a píxel. ``img``
    es sRGB flotante (H, W, 3) y ``mask`` usa el contrato bool (H, W), donde
    True representa lesión. Una máscara ausente o degenerada impide transformar
    un marco completo sin protección.
    """
    if mask is None:
        raise ValueError("shift_skin_tone requiere una máscara explícita de lesión")
    mask = np.asarray(mask, dtype=bool)
    if _is_degenerate_mask(mask):
        raise ValueError(
            f"máscara de lesión degenerada (área={float(mask.mean()):.4f}); "
            "no es posible proteger la región de lesión"
        )

    shift = clamp_fst_shift(shift_buckets, max_shift=max_shift)
    img = np.asarray(img, dtype=float)
    lab = srgb_to_lab(img)

    shifted_L = np.clip(lab[..., 0] - shift * FST_SHIFT_L_STEP, 0.0, 100.0)
    shifted_lab = lab.copy()
    shifted_lab[..., 0] = np.where(mask, lab[..., 0], shifted_L)

    out = lab_to_srgb(shifted_lab)
    out = np.where(mask[..., None], img, out)  # Conserva la lesión exactamente.
    return np.clip(out, 0.0, 1.0)


def shift_skin_tone_from_mask_record(
    img: np.ndarray,
    mask: np.ndarray | None,
    mask_record: dict,
    shift_buckets: int,
    *,
    max_shift: int = MAX_FST_SHIFT,
) -> np.ndarray:
    """Aplica un cambio de tono solo si la máscara pasa el QA aceptado.

    El adaptador vincula los píxeles al contrato JSON para impedir usar una
    máscara obsoleta, en cuarentena o alterada como evidencia de preservación.
    """
    if not bool(mask_record.get("eligible")):
        raise ValueError("el registro de máscara no es elegible para la transformación de tono")
    if mask_record.get("mask_semantics") != "true=lesion":
        raise ValueError("el registro de máscara debe declarar la semántica true=lesion")
    if mask is None:
        raise ValueError("no se puede comprobar la procedencia sin una máscara")
    lesion_mask = np.asarray(mask, dtype=bool)
    if mask_record.get("mask_sha256") != mask_sha256(lesion_mask):
        raise ValueError("la procedencia de la máscara no coincide con los píxeles entregados")
    return shift_skin_tone(
        img, lesion_mask, shift_buckets, max_shift=max_shift
    )


def synthetic_blend_mask(
    n_samples: int, *, target_ratio: float = 0.5, seed: int = 42
) -> np.ndarray:
    """Máscara booleana que selecciona las muestras con contraparte sintética FST."""
    if not (0.0 <= target_ratio <= 1.0):
        raise ValueError(f"target_ratio debe pertenecer a [0, 1]; se recibió {target_ratio!r}")

    n_synthetic = int(round(n_samples * target_ratio))
    mask = np.zeros(n_samples, dtype=bool)
    if n_synthetic == 0 or n_samples == 0:
        return mask
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_samples, size=n_synthetic, replace=False)
    mask[idx] = True
    return mask


def should_run_stage2(gate_status: str) -> bool:
    """La etapa 2 solo se ejecuta cuando la puerta de equidad de la etapa 1 falla."""
    return gate_status == FAIL
