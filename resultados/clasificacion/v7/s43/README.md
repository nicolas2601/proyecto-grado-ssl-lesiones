# Evaluación única v7 (val)

Probe `logreg`, n = 2004, `code_revision` f562ea9d8ad0ee9d262289a8dd0b924c05e1f63b. Generado desde `final_eval.json`. No editar a mano: `--check` lo compara en CI.

| Brazo | Macro-F1 | Bal-acc | AUC | ECE |
|---|---|---|---|---|
| Modelo de control | 0,6850 [0,6353; 0,7258] | 0,6457 [0,5968; 0,6942] | 0,9662 [0,9603; 0,9711] | 0,0202 [0,0169; 0,0377] |
| Proposed sin segmentación | 0,6638 [0,6152; 0,7052] | 0,6119 [0,5657; 0,6607] | 0,9623 [0,9563; 0,9677] | 0,0178 [0,0166; 0,0377] |
| DINOv2 público | 0,6158 [0,5690; 0,6550] | 0,5924 [0,5459; 0,6376] | 0,9257 [0,9123; 0,9370] | 0,1307 [0,1159; 0,1476] |

| Contraste | Δ Macro-F1 | Δ Bal-acc | Δ AUC | Δ ECE |
|---|---|---|---|---|
| Modelo de control − DINOv2 público | 0,0692 [0,0292; 0,1125] | 0,0533 [0,0167; 0,0952] | 0,0405 [0,0307; 0,0516] | -0,1105 [-0,1241; -0,0854] |
| Proposed sin segmentación − DINOv2 público | 0,0479 [0,0051; 0,0920] | 0,0195 [-0,0248; 0,0627] | 0,0366 [0,0265; 0,0480] | -0,1129 [-0,1237; -0,0857] |
| Proposed sin segmentación − Modelo de control | -0,0213 [-0,0584; 0,0156] | -0,0338 [-0,0739; 0,0044] | -0,0039 [-0,0074; -0,0003] | -0,0024 [-0,0152; 0,0141] |

| Predicción (docs/28 §5) | Criterio | Veredicto |
|---|---|---|
| P1 | macro-F1 Modelo de control - público >= -0.01 | cumplida |
| P2 | ECE Proposed sin segmentación - Modelo de control < 0 con IC que excluye el cero y macro-F1 >= -0.01 | no cumplida |
| P3 | macro-F1 Proposed sin segmentación - público >= 0 y ECE Proposed sin segmentación - público < 0 con IC que excluye el cero | cumplida |
| P1-NI | límite inferior IC macro-F1 Modelo de control - público >= -0.01 | cumplida |
| P3-IC | límite inferior IC macro-F1 Proposed sin segmentación - público > 0 (permite decir que le gana) | cumplida |

## ECE con IC basic (lectura secundaria)

IC basic (reverse-percentile) de ECE, lectura secundaria. El ECE remuestreado tiene sesgo hacia arriba, así que el IC percentil puede dejar el punto afuera (Modelo de control en val v7). El basic refleja los cuantiles alrededor del punto y puede salir de [0, 1]. Los veredictos primarios usan el IC percentil registrado y no cambian.

| Brazo o contraste | ECE / Δ ECE | IC basic | ¿Punto fuera del IC percentil? |
|---|---|---|---|
| Modelo de control | 0,0202 | [0,0028; 0,0235] | no |
| Proposed sin segmentación | 0,0178 | [-0,0020; 0,0191] | no |
| DINOv2 público | 0,1307 | [0,1138; 0,1456] | no |
| Modelo de control − DINOv2 público | -0,1105 | [-0,1356; -0,0969] | no |
| Proposed sin segmentación − DINOv2 público | -0,1129 | [-0,1402; -0,1021] | no |
| Proposed sin segmentación − Modelo de control | -0,0024 | [-0,0189; 0,0104] | no |

P2 leída con el IC basic: no cumplida (el veredicto primario no cambia).
