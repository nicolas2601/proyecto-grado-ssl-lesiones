# T7 — Chequeo del maestro r0 por tercil de luminancia (HAM10000)

Generado por `scripts/t7_r0_tercile_check.py` desde `t7_r0_tercile_check.json`; no editar a mano. Registro y regla condicional: docs/27, bloque T7. Los bins son `tone_tercile` del manifiesto publicado (luminancia, no es tono de piel). Oráculo diagnóstico: no entra al entrenamiento.

Cobertura: 9822 imágenes con tercil; 193 sin tercil; 0 fuera del pool.

| Tercil | n | excluidas por r0 | Dice r0 (a) binaria | Dice r0 (b) confiable | Dice experimento de segmentación DINOv2 con pseudo-máscaras CCAM (mismas imágenes) |
|---|---:|---:|---:|---:|---:|
| oscuro | 4082 | 299 (0,073) | 0,7376 | 0,7949 | 0,7680 |
| medio | 4876 | 206 (0,042) | 0,7825 | 0,8395 | 0,8168 |
| claro | 864 | 45 (0,052) | 0,8374 | 0,8837 | 0,8644 |

| Contraste oscuro − resto | Δ [IC 95 %] |
|---|---|
| G(r0), forma (a) | -0,0531 [-0,0617; -0,0446] |
| G(r0), forma (b) | -0,0512 [-0,0599; -0,0428] |
| G(experimento de segmentación DINOv2 con pseudo-máscaras CCAM), mismas imágenes | -0,0560 [-0,0629; -0,0492] |
| \|G(experimento de segmentación DINOv2 con pseudo-máscaras CCAM)\| − \|G(r0)\| (pareado, descriptivo) | +0,0029 [-0,0010; +0,0069] |

Regla condicional (\|G(r0)\| ≥ \|G(experimento de segmentación DINOv2 con pseudo-máscaras CCAM)\|, estimaciones puntuales, forma (a)): **no**.
