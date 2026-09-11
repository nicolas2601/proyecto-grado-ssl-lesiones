"""Mitigacion Stage 2: transformador de color estilo CIRCLe, muestreo de
shift de fototipo y politica de blend sintetico, replicado solo a nivel del
probe entrenable (Pakzad et al., arXiv:2208.13528; Corbin & Marques, IEEE
Access 2023).

Esto es una transformacion de color determinista,
un muestreador de shift de fototipo acotado, y una politica de blend -
construidos y probados de antemano para estar listos apenas exista un
fst_manifest.csv real. Nada aca mide ni afirma una mejora de equidad real;
should_run_stage2 es el freno explicito que evita correr esto contra datos
reales hasta que Stage 1 diga que vale la pena.
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

# Shift maximo de bucket FST que puede pedir una muestra sintetica.
MAX_FST_SHIFT = 2

# Unidades de L* desplazadas por nivel de bucket FST. Magnitud heuristica,
# no medida: buckets mas oscuros (numero mayor) bajan L*. Elegida para que
# un shift de 1 bucket se note sin sacar L* de [0, 100] en un shift de 2.
FST_SHIFT_L_STEP = 8.0

# Blanco de referencia D65, observador 2 grados - debe coincidir bit a bit
# con las constantes privadas de fitzpatrick.py, o lab_to_srgb deja de ser
# la inversa de srgb_to_lab. Duplicado en vez de importado porque
# fitzpatrick.py las mantiene privadas a proposito.
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
    """Inversa de la no linealidad de CIE Lab de fitzpatrick.py, mismo
    punto de quiebre (0.008856) para que la conversion de ida y vuelta
    cierre dentro de la misma tolerancia que usa srgb_to_lab."""
    cube = t**3
    linear = (t - 16.0 / 116.0) / 7.787
    return np.where(cube > 0.008856, cube, linear)


def _gamma_encode(linear: np.ndarray) -> np.ndarray:
    """Lineal [0, 1] -> sRGB con correccion gamma. Inversa de
    fitzpatrick._inverse_gamma."""
    low = linear * 12.92
    high = 1.055 * np.power(np.clip(linear, 1e-12, None), 1.0 / 2.4) - 0.055
    return np.where(linear <= 0.0031308, low, high)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convierte CIE Lab de vuelta a sRGB, float en [0, 1], shape (..., 3).

    Inversa de fitzpatrick.srgb_to_lab. La salida se recorta a [0, 1]: no
    todo triple Lab arbitrario es un color sRGB representable, y devolver
    valores fuera de gamut en silencio rompe el contrato de rango valido
    contra el que se testea este modulo.
    """
    lab = np.asarray(lab, dtype=float)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    xr, yr, zr = _f_inv(fx), _f_inv(fy), _f_inv(fz)
    xyz = np.stack([xr * _D65_XN, yr * _D65_YN, zr * _D65_ZN], axis=-1)

    linear = xyz @ _XYZ_TO_SRGB.T
    linear = np.clip(linear, 0.0, None)  # ruido flotante negativo minimo
    srgb = _gamma_encode(linear)
    return np.clip(srgb, 0.0, 1.0)


def clamp_fst_shift(requested_shift: int, *, max_shift: int = MAX_FST_SHIFT) -> int:
    """Acota un shift de bucket FST pedido al rango [-max_shift, max_shift].

    Nunca deja pasar un pedido fuera de rango en silencio, siempre lo
    recorta explicitamente.
    """
    if requested_shift > max_shift:
        return max_shift
    if requested_shift < -max_shift:
        return -max_shift
    return requested_shift


def target_fst_bucket(
    source_bucket: int, requested_shift: int, *, max_shift: int = MAX_FST_SHIFT
) -> int:
    """Bucket FST resultante (1..6) tras un shift, ya acotado.

    Ejemplo: bucket origen 2, shift pedido 2 -> bucket destino 4. Un shift
    mas alla de max_shift se recorta primero, y el resultado se limita al
    rango valido de buckets para que un shift cerca de los extremos no de
    bucket 0 o 7.
    """
    shift = clamp_fst_shift(requested_shift, max_shift=max_shift)
    n_buckets = len(FST_THRESHOLDS) + 1
    return int(min(max(source_bucket + shift, 1), n_buckets))


def _is_degenerate_mask(mask: np.ndarray) -> bool:
    """Mismos limites que el _is_degenerate privado de fitzpatrick.py (un
    solo contrato de mascara, dos usos)."""
    area = float(mask.mean())
    return not (MIN_MASK_AREA_FRACTION <= area <= MAX_MASK_AREA_FRACTION)


def shift_skin_tone(
    img: np.ndarray, mask: np.ndarray | None, shift_buckets: int, *, max_shift: int = MAX_FST_SHIFT
) -> np.ndarray:
    """Transformador de color LAB determinista: desplaza el L* de los
    pixeles de piel segun shift_buckets niveles FST; los pixeles de lesion
    (mask True) pasan sin cambios, pixel a pixel.

    img es sRGB float (H, W, 3) en [0, 1]. mask sigue el mismo contrato que
    fitzpatrick.estimate_fst: bool (H, W), True = lesion; una mascara no
    booleana se convierte (mismo resguardo de dtype que fitzpatrick, para
    que ~mask siga siendo negacion logica).

    Lanza ValueError si falta la mascara o es degenerada (area fuera de
    1%-90%) en vez de transformar todo el cuadro en silencio.
    """
    if mask is None:
        raise ValueError("shift_skin_tone requiere una mascara de lesion explicita")
    mask = np.asarray(mask, dtype=bool)
    if _is_degenerate_mask(mask):
        raise ValueError(
            f"mascara de lesion degenerada (area={float(mask.mean()):.4f}); "
            "no se puede proteger la region de la lesion"
        )

    shift = clamp_fst_shift(shift_buckets, max_shift=max_shift)
    img = np.asarray(img, dtype=float)
    lab = srgb_to_lab(img)

    shifted_L = np.clip(lab[..., 0] - shift * FST_SHIFT_L_STEP, 0.0, 100.0)
    shifted_lab = lab.copy()
    shifted_lab[..., 0] = np.where(mask, lab[..., 0], shifted_L)

    out = lab_to_srgb(shifted_lab)
    out = np.where(mask[..., None], img, out)  # passthrough exacto sobre la lesion
    return np.clip(out, 0.0, 1.0)


def synthetic_blend_mask(
    n_samples: int, *, target_ratio: float = 0.5, seed: int = 42
) -> np.ndarray:
    """Mascara booleana (largo n_samples) que elige que muestras reciben su
    contraparte sintetica con shift de FST, dimensionada para que
    target_ratio del lote sea sintetico (por defecto ~50/50). Deterministica
    para una misma seed.
    """
    if not (0.0 <= target_ratio <= 1.0):
        raise ValueError(f"target_ratio debe estar en [0, 1], se recibio {target_ratio!r}")

    n_synthetic = int(round(n_samples * target_ratio))
    mask = np.zeros(n_samples, dtype=bool)
    if n_synthetic == 0 or n_samples == 0:
        return mask
    rng = np.random.RandomState(seed)
    idx = rng.choice(n_samples, size=n_synthetic, replace=False)
    mask[idx] = True
    return mask


def should_run_stage2(gate_status: str) -> bool:
    """Dependencia de Stage 2 respecto a Stage 1: Stage 2 solo hace falta
    correrlo cuando el gate de equidad de Stage 1 da FAIL. Es opcional, no
    obligatorio, cuando Stage 1 da PASS o SKIP.
    """
    return gate_status == FAIL
