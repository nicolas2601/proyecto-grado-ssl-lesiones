"""Estimacion de fototipo Fitzpatrick (FST) por angulo ITA, con mascara.

El repo documenta el sesgo de tono de piel como una limitacion conocida
(HAM10000 sesga a FST I-III) pero nunca lo mide. Este modulo lo convierte
en un numero: cada imagen recibe un bucket FST (1-6) o un estado explicito
de "no disponible", calculado solo sobre pixeles de piel (la lesion se
excluye via mascara).

Numpy puro, sin scikit-image en runtime. scikit-image se usa unicamente en
los tests como oraculo para validar la conversion a LAB contra
skimage.color.rgb2lab (atol=1e-4) - nunca se importa aca, porque este
modulo se inlinea en el notebook y un import de skimage rompe esa celda.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dermssl.pretrain import lesion_guided

# Umbrales ITA de Groh et al. (grados), de mayor a menor. Bucket 1 = piel
# mas clara, bucket 6 = mas oscura.
FST_THRESHOLDS = (55.0, 41.0, 28.0, 19.0, 10.0)

# Por debajo de esta fraccion de piel, la estimacion no es confiable.
MIN_SKIN_FRACTION = 0.05

# Mismo rango que usa pretrain.lesion_guided para descartar una mascara
# degenerada (demasiado chica, o tan grande que probablemente tomo todo
# el cuadro en vez de una sola lesion).
MIN_MASK_AREA_FRACTION = 0.01
MAX_MASK_AREA_FRACTION = 0.90

# Pixeles casi negros se tratan como borde/vineta de la captura dermoscopica,
# no como piel, sin importar lo que diga la mascara de lesion.
BORDER_LUMINANCE_THRESHOLD = 0.02

# Blanco de referencia D65, observador 2 grados (igual que skimage).
_D65_XN = 0.95047
_D65_YN = 1.0
_D65_ZN = 1.08883

# sRGB -> XYZ (D65), misma matriz que usa skimage.
_SRGB_TO_XYZ = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ]
)


def _inverse_gamma(channel: np.ndarray) -> np.ndarray:
    """Quita la correccion gamma sRGB: [0, 1] gamma -> [0, 1] lineal."""
    linear_low = channel / 12.92
    linear_high = ((channel + 0.055) / 1.055) ** 2.4
    return np.where(channel <= 0.04045, linear_low, linear_high)


def _f(t: np.ndarray) -> np.ndarray:
    """No linealidad de CIE Lab: raiz cubica con piso lineal cerca de 0."""
    cube_root = np.cbrt(t)
    linear = 7.787 * t + 16.0 / 116.0
    return np.where(t > 0.008856, cube_root, linear)


def srgb_to_lab(img: np.ndarray) -> np.ndarray:
    """Convierte una imagen sRGB (float [0, 1], H, W, 3) a CIE Lab (D65)."""
    linear = _inverse_gamma(img)
    xyz = linear @ _SRGB_TO_XYZ.T

    xr = xyz[..., 0] / _D65_XN
    yr = xyz[..., 1] / _D65_YN
    zr = xyz[..., 2] / _D65_ZN

    fx, fy, fz = _f(xr), _f(yr), _f(zr)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def ita_degrees(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angulo ITA en grados: arctan((L*-50)/b*) * 180/pi.

    Arctan simple, no arctan2: arctan2 envolveria lecturas con b* negativo
    fuera del rango [-90, 90] que exige la formula original.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.degrees(np.arctan((np.asarray(L) - 50.0) / np.asarray(b)))


def fst_bucket(ita: float) -> int:
    """Mapea un ITA agregado (grados) a un bucket Fitzpatrick (1-6).

    Un valor justo en el umbral cae en el bucket mas oscuro.
    """
    bucket = len(FST_THRESHOLDS) + 1
    for i, threshold in enumerate(FST_THRESHOLDS, start=1):
        if ita > threshold:
            bucket = i
            break
    return bucket


@dataclass
class FSTEstimate:
    """Resultado de una estimacion FST. Nunca se descarta en silencio:
    siempre trae un bucket o un status explicando por que no hay uno."""

    bucket: int | None  # 1..6, None si no esta disponible
    ita: float  # NaN si no esta disponible
    skin_fraction: float
    status: str  # "ok" | "no_skin" | "degenerate_mask" | "ita_out_of_range"


def _is_degenerate(mask: np.ndarray) -> bool:
    """True si el area de la mascara esta fuera del rango plausible para
    una sola lesion (1%-90%)."""
    area = float(mask.mean())
    return not (MIN_MASK_AREA_FRACTION <= area <= MAX_MASK_AREA_FRACTION)


def estimate_fst(img: np.ndarray, mask: np.ndarray | None = None) -> FSTEstimate:
    """Estimacion FST guiada por mascara, para una imagen sRGB (H, W, 3) en [0, 1].

    mask es la segmentacion de la lesion (bool (H, W), True = lesion). Si no
    se pasa, o si es degenerada, se usa el fallback Otsu de
    pretrain.lesion_guided.

    Una mascara no booleana se convierte a bool: el resto del codigo depende
    de que ~mask sea negacion logica, no negacion bit a bit.
    """
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    gray = img.mean(axis=-1)
    border = gray < BORDER_LUMINANCE_THRESHOLD

    if mask is None or _is_degenerate(mask):
        mask = lesion_guided.lesion_mask(gray)

    if _is_degenerate(mask):
        skin = ~mask & ~border
        return FSTEstimate(
            bucket=None,
            ita=float("nan"),
            skin_fraction=float(skin.mean()),
            status="degenerate_mask",
        )

    skin = ~mask & ~border
    skin_fraction = float(skin.mean())

    if skin_fraction < MIN_SKIN_FRACTION:
        return FSTEstimate(
            bucket=None, ita=float("nan"), skin_fraction=skin_fraction, status="no_skin"
        )

    lab = srgb_to_lab(img)
    skin_lab = lab[skin]
    L_med = np.median(skin_lab[:, 0])
    b_med = np.median(skin_lab[:, 2])
    ita = float(ita_degrees(np.asarray(L_med), np.asarray(b_med)))

    # Un ITA finito no deberia salir de [-90, 90], pero una piel acromatica
    # exacta en L*=50 da una division 0/0 = NaN. Se chequea explicito (no con
    # un assert, que rompe toda la corrida por una imagen atipica y ademas
    # desaparece con python -O) y se degrada con un status, no una excepcion.
    if not (-90.0 <= ita <= 90.0):
        return FSTEstimate(
            bucket=None, ita=ita, skin_fraction=skin_fraction, status="ita_out_of_range"
        )

    return FSTEstimate(
        bucket=fst_bucket(ita), ita=ita, skin_fraction=skin_fraction, status="ok"
    )
