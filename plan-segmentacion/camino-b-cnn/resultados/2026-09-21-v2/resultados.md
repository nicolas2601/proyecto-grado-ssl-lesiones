# Resultados de segmentacion sin etiquetas

## Configuraciones

### isic2018

| configuracion | n | Dice | IC95 Dice | mediana | p10 | HD95 | mediana HD95 |
|---|---|---|---|---|---|---|---|
| final | 2594 | 0.7177 | [0.7090, 0.7260] | 0.7925 | 0.4015 | 26.05 | 13.60 |
| final_tta | 2594 | 0.7180 | [0.7094, 0.7263] | 0.7917 | 0.4024 | 25.74 | 13.42 |
| v2 | 2594 | 0.7161 | [0.7076, 0.7242] | 0.7878 | 0.3970 | 25.87 | 13.93 |
| v2_tta | 2594 | 0.7173 | [0.7087, 0.7256] | 0.7886 | 0.3964 | 25.57 | 13.93 |
| rnorm | 2594 | 0.7162 | [0.7073, 0.7245] | 0.7891 | 0.3475 | 33.47 | 22.85 |
| xnorm | 2594 | 0.6954 | [0.6866, 0.7037] | 0.7678 | 0.3178 | 33.05 | 22.13 |

### ham10000

| configuracion | n | Dice | IC95 Dice | mediana | p10 | HD95 | mediana HD95 |
|---|---|---|---|---|---|---|---|
| final | 10015 | 0.7847 | [0.7812, 0.7883] | 0.8519 | 0.5075 | 22.44 | 12.65 |
| final_tta | 10015 | 0.7849 | [0.7814, 0.7885] | 0.8527 | 0.5083 | 22.38 | 12.53 |
| v2 | 10015 | 0.7825 | [0.7790, 0.7862] | 0.8512 | 0.5017 | 22.63 | 12.88 |
| v2_tta | 10015 | 0.7830 | [0.7796, 0.7867] | 0.8515 | 0.5031 | 22.57 | 12.73 |
| rnorm | 10015 | 0.7650 | [0.7615, 0.7682] | 0.8193 | 0.5300 | 22.66 | 15.04 |
| xnorm | 10015 | 0.7326 | [0.7288, 0.7359] | 0.7878 | 0.4848 | 22.96 | 16.00 |

## Bootstrap pareado por imagen (10000 remuestreos, semilla 42)

Diferencia = primera configuracion menos la segunda. En HD95 un valor
negativo es mejor. La columna veredicto ya tiene en cuenta el sentido
de cada metrica: gana significa mejor, no mayor.

### dice

| conjunto | comparacion | n | diferencia | IC95 | veredicto |
|---|---|---|---|---|---|
| isic2018 | v2 - final | 2594 | -0.0015 | [-0.0027, -0.0004] | pierde |
| isic2018 | final_tta - final | 2594 | +0.0003 | [-0.0002, +0.0007] | empate |
| isic2018 | v2_tta - v2 | 2594 | +0.0012 | [+0.0007, +0.0018] | gana |
| isic2018 | final - rnorm | 2594 | +0.0015 | [-0.0078, +0.0106] | empate |
| isic2018 | final - xnorm | 2594 | +0.0223 | [+0.0127, +0.0318] | gana |
| ham10000 | v2 - final | 10015 | -0.0021 | [-0.0026, -0.0017] | pierde |
| ham10000 | final_tta - final | 10015 | +0.0002 | [+0.0000, +0.0004] | gana |
| ham10000 | v2_tta - v2 | 10015 | +0.0005 | [+0.0004, +0.0007] | gana |
| ham10000 | final - rnorm | 10015 | +0.0197 | [+0.0164, +0.0230] | gana |
| ham10000 | final - xnorm | 10015 | +0.0521 | [+0.0485, +0.0555] | gana |

### hd95

| conjunto | comparacion | n | diferencia | IC95 | veredicto |
|---|---|---|---|---|---|
| isic2018 | v2 - final | 2589 | -0.2218 | [-0.5731, +0.1241] | empate |
| isic2018 | final_tta - final | 2588 | -0.3183 | [-0.6066, -0.0399] | gana |
| isic2018 | v2_tta - v2 | 2591 | -0.3396 | [-0.5881, -0.0999] | gana |
| isic2018 | final - rnorm | 2589 | -7.3663 | [-8.6183, -6.1211] | gana |
| isic2018 | final - xnorm | 2589 | -6.9471 | [-8.1539, -5.7178] | gana |
| ham10000 | v2 - final | 10006 | +0.1741 | [+0.0789, +0.2693] | pierde |
| ham10000 | final_tta - final | 10007 | -0.0826 | [-0.1561, -0.0113] | gana |
| ham10000 | v2_tta - v2 | 10008 | -0.0463 | [-0.0987, +0.0056] | empate |
| ham10000 | final - rnorm | 10007 | -0.2056 | [-0.6956, +0.2864] | empate |
| ham10000 | final - xnorm | 10007 | -0.5060 | [-0.9892, -0.0341] | gana |

## Dice por banda de cobertura de la lesion de referencia

La banda se define con ref_coverage del conjunto final, igual para todas
las configuraciones, para que las filas sean comparables.

### isic2018

| banda ref_coverage | n | final | final_tta | v2 | v2_tta |
|---|---|---|---|---|---|
| [0.00, 0.10) | 1056 | 0.6455 | 0.6464 | 0.6457 | 0.6481 |
| [0.10, 0.25) | 685 | 0.7933 | 0.7936 | 0.7878 | 0.7878 |
| [0.25, 0.40) | 405 | 0.7822 | 0.7823 | 0.7781 | 0.7790 |
| [0.40, 0.60) | 273 | 0.7328 | 0.7318 | 0.7307 | 0.7306 |
| [0.60, 1.01) | 175 | 0.6838 | 0.6834 | 0.6949 | 0.6956 |

### ham10000

| banda ref_coverage | n | final | final_tta | v2 | v2_tta |
|---|---|---|---|---|---|
| [0.00, 0.10) | 1956 | 0.7496 | 0.7495 | 0.7499 | 0.7501 |
| [0.10, 0.25) | 3623 | 0.8500 | 0.8503 | 0.8458 | 0.8462 |
| [0.25, 0.40) | 2203 | 0.7941 | 0.7945 | 0.7906 | 0.7912 |
| [0.40, 0.60) | 1583 | 0.7238 | 0.7242 | 0.7224 | 0.7231 |
| [0.60, 1.01) | 650 | 0.6422 | 0.6419 | 0.6474 | 0.6484 |

