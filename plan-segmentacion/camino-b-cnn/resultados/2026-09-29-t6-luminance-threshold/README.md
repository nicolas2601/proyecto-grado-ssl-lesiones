# T6 — Umbral de máscara por tercil de luminancia sobre A2 (exploratorio)

Generado por `scripts/t6_luminance_threshold.py` desde `t6_luminance_threshold.json`; no editar a mano. Registro, predicción y salvedades: docs/27, bloque T6. Análogo barato de *Fitzpatrick Thresholding for Skin Image Segmentation* (arXiv 2510.06655).

Los bins son terciles de la mediana de L\* de la piel (`tone_tercile` del manifiesto publicado): no son tono de piel ni fototipo. Los umbrales se eligen con máscaras de referencia de HAM10000 fold-0 train (oráculo diagnóstico declarado) y se evalúan en fold-0 val (el test de C1-v7); ISIC 2018 es solo descriptivo.

Grilla 0,30-0,70, paso 0,05. Bootstrap pareado por imagen: 1000 remuestreos, semilla 20260923, IC percentil 95 %.

## Umbrales elegidos (HAM10000 fold-0 train, dentro de la muestra)

| Tercil | Umbral | n ajuste | Dice ajuste a 0,5 | Dice ajuste con el umbral |
|---|---:|---:|---:|---:|
| oscuro | 0,30 | 3262 | 0,7620 | 0,7647 |
| medio | 0,30 | 3919 | 0,8182 | 0,8195 |
| claro | 0,40 | 681 | 0,8555 | 0,8557 |

## Dice por tercil en evaluación

| Conjunto | Tercil | n | Dice a 0,5 | Dice adaptado |
|---|---|---:|---:|---:|
| HAM10000 | oscuro | 820 | 0,7635 | 0,7664 |
| HAM10000 | medio | 957 | 0,7934 | 0,7951 |
| HAM10000 | claro | 183 | 0,8840 | 0,8837 |
| ISIC 2018 | oscuro | 158 | 0,7808 | 0,7858 |
| ISIC 2018 | medio | 304 | 0,8154 | 0,8173 |
| ISIC 2018 | claro | 260 | 0,8367 | 0,8370 |

## Brecha oscuro − resto y Dice global (predicción de Pau)

| Conjunto | n oscuro | n resto | Brecha a 0,5 [IC 95 %] | Brecha adaptada [IC 95 %] | Reducción R [IC 95 %] | Parte 1 | ΔDice global [IC 95 %] | Parte 2 | Predicción |
|---|---:|---:|---|---|---|---|---|---|---|
| HAM10000 | 820 | 1140 | -0,0444 [-0,0619; -0,0286] | -0,0429 [-0,0598; -0,0273] | +0,0015 [+0,0005; +0,0026] | se achica, pero menos de 0,02 | +0,0020 [+0,0015; +0,0026] | se sostiene | **no cumplida** |
| ISIC 2018 | 158 | 564 | -0,0444 [-0,0725; -0,0159] | -0,0406 [-0,0678; -0,0126] | +0,0039 [+0,0018; +0,0061] | descriptivo | +0,0020 [+0,0013; +0,0027] | descriptivo | **sin veredicto** |
