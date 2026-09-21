# Referencias de comparación

Estado: pendiente.

Pisos que dan escala para juzgar los caminos A y B. Ninguno se entrena con máscaras.

| Carpeta | Método | Referencia |
|---|---|---|
| `otsu/` | Umbral de Otsu | IoU 0,477 en ISIC 2018 (Nagaoka 2025) |
| `umbral-adaptativo/` | Umbral global con normalización de color RGB/XYZ | Thanh et al., ICEEE 2019 |
| `sam/` | SAM con prompt central automático | IoU 0,578 en ISIC 2018 (Nagaoka 2025) |

El techo supervisado (U-Net con máscaras) se usa solo como referencia y no se implementa aquí.

Candidato del repositorio: `src/dermssl/pretrain/lesion_guided.py` (fallback de Otsu).
