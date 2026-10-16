# T2 — Dice por tercil de luminancia (exploratorio)

Generado por `scripts/t2_luminance_dice.py` desde `t2_luminance_dice.json`; no editar a mano. Registro, predicción y salvedades: docs/27, bloque T2.

Los bins son terciles de la mediana de L\* de la piel sobre el pool (`tone_tercile` del manifiesto publicado, sha256 `57df1e65e88d…`): luminancia, no es tono de piel. La luminancia está confundida con el diagnóstico, el tamaño de la lesión, el dermatoscopio y la iluminación, así que ningún contraste se atribuye a una causa.

Bootstrap: 1000 remuestreos, semilla 20260923, IC percentil 95 %. Contraste oscuro − resto no pareado y estratificado por grupo. Veredicto registrado: **cumplida** si el IC queda entero bajo -0,02; sin veredicto si un grupo tiene n < 100.

## Cobertura del cruce por `image_id`

| Conjunto | oscuro | medio | claro | cuarentena | fuera del manifiesto | total |
|---|---:|---:|---:|---:|---:|---:|
| ISIC 2018 | 158 | 304 | 260 | 50 | 1822 | 2594 |
| HAM10000 | 4082 | 4876 | 864 | 193 | 0 | 10015 |

## Dice y HD95 por tercil

| Conjunto | Segmentador | Tercil | n | Dice [IC 95 %] | HD95 px [IC 95 %] | HD95 no finito |
|---|---|---|---:|---|---|---:|
| ISIC 2018 | Experimento de segmentación 5 | oscuro | 158 | 0,7808 [0,7556; 0,8050] | 24,58 [20,75; 28,33] | 0 |
| ISIC 2018 | Experimento de segmentación 5 | medio | 304 | 0,8154 [0,7978; 0,8310] | 19,77 [17,92; 22,00] | 0 |
| ISIC 2018 | Experimento de segmentación 5 | claro | 260 | 0,8367 [0,8195; 0,8524] | 17,81 [15,70; 20,03] | 0 |
| ISIC 2018 | Experimento de segmentación 3 | oscuro | 158 | 0,7651 [0,7380; 0,7921] | 26,43 [22,73; 30,17] | 0 |
| ISIC 2018 | Experimento de segmentación 3 | medio | 304 | 0,8024 [0,7835; 0,8193] | 21,65 [19,38; 23,98] | 0 |
| ISIC 2018 | Experimento de segmentación 3 | claro | 260 | 0,8205 [0,8009; 0,8380] | 19,22 [16,95; 21,83] | 0 |
| HAM10000 | Experimento de segmentación 5 | oscuro | 4082 | 0,7623 [0,7568; 0,7675] | 25,30 [24,64; 26,02] | 0 |
| HAM10000 | Experimento de segmentación 5 | medio | 4876 | 0,8134 [0,8088; 0,8176] | 20,12 [19,57; 20,70] | 0 |
| HAM10000 | Experimento de segmentación 5 | claro | 864 | 0,8615 [0,8533; 0,8703] | 14,57 [13,63; 15,50] | 0 |
| HAM10000 | Experimento de segmentación 3 | oscuro | 4082 | 0,7503 [0,7443; 0,7555] | 25,83 [25,09; 26,58] | 0 |
| HAM10000 | Experimento de segmentación 3 | medio | 4876 | 0,8030 [0,7983; 0,8075] | 20,61 [20,05; 21,18] | 0 |
| HAM10000 | Experimento de segmentación 3 | claro | 864 | 0,8616 [0,8535; 0,8696] | 14,34 [13,41; 15,21] | 0 |

## Contraste oscuro − resto (predicción de Pau)

| Conjunto | Segmentador | n oscuro | n resto | ΔDice oscuro − resto [IC 95 %] | Veredicto | ΔDice oscuro − global [IC 95 %] (descriptivo) | ΔHD95 oscuro − resto px [IC 95 %] (descriptivo) |
|---|---|---:|---:|---|---|---|---|
| ISIC 2018 | Experimento de segmentación 5 | 158 | 564 | -0,0444 [-0,0715; -0,0161] | **distinguible pero no clara** | -0,0347 [-0,0599; -0,0082] | +5,71 [+1,62; +9,68] |
| ISIC 2018 | Experimento de segmentación 3 | 158 | 564 | -0,0456 [-0,0762; -0,0140] | **distinguible pero no clara** | -0,0356 [-0,0647; -0,0079] | +5,90 [+1,79; +9,88] |
| HAM10000 | Experimento de segmentación 5 | 4082 | 5740 | -0,0583 [-0,0652; -0,0515] | **cumplida** | -0,0341 [-0,0409; -0,0278] | +6,02 [+5,10; +6,92] |
| HAM10000 | Experimento de segmentación 3 | 4082 | 5740 | -0,0616 [-0,0688; -0,0542] | **cumplida** | -0,0360 [-0,0430; -0,0293] | +6,17 [+5,22; +7,06] |
