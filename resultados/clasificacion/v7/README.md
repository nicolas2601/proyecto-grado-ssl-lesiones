# Evaluación única v7 (val)

Probe `logreg`, n = 2004, `code_revision` 7fd6245c317ac6ea68ef4f32821ddc82c94ebeea. Generado desde `final_eval.json`. No editar a mano: `--check` lo compara en CI.

| Brazo | Macro-F1 | Bal-acc | AUC | ECE |
|---|---|---|---|---|
| Modelo de control | 0,6464 [0,5980; 0,6894] | 0,6021 [0,5565; 0,6488] | 0,9619 [0,9563; 0,9671] | 0,0135 [0,0138; 0,0336] |
| Proposed sin segmentación | 0,6606 [0,6157; 0,7040] | 0,6327 [0,5871; 0,6838] | 0,9629 [0,9563; 0,9690] | 0,0325 [0,0253; 0,0497] |
| DINOv2 público | 0,6158 [0,5690; 0,6550] | 0,5924 [0,5459; 0,6376] | 0,9257 [0,9123; 0,9370] | 0,1307 [0,1159; 0,1476] |

| Contraste | Δ Macro-F1 | Δ Bal-acc | Δ AUC | Δ ECE |
|---|---|---|---|---|
| Modelo de control − DINOv2 público | 0,0306 [-0,0137; 0,0744] | 0,0097 [-0,0346; 0,0549] | 0,0362 [0,0258; 0,0482] | -0,1172 [-0,1260; -0,0919] |
| Proposed sin segmentación − DINOv2 público | 0,0447 [0,0080; 0,0837] | 0,0403 [0,0024; 0,0777] | 0,0372 [0,0289; 0,0469] | -0,0982 [-0,1119; -0,0771] |
| Proposed sin segmentación − Modelo de control | 0,0141 [-0,0190; 0,0539] | 0,0306 [-0,0055; 0,0691] | 0,0010 [-0,0045; 0,0063] | 0,0190 [0,0003; 0,0270] |

| Predicción (docs/28 §5) | Criterio | Veredicto |
|---|---|---|
| Predicción de anclaje | macro-F1 Modelo de control - público >= -0.01 | cumplida |
| Predicción de tono | ECE Proposed sin segmentación - Modelo de control < 0 con IC que excluye el cero y macro-F1 >= -0.01 | no cumplida |
| Predicción de objetivo | macro-F1 Proposed sin segmentación - público >= 0 y ECE Proposed sin segmentación - público < 0 con IC que excluye el cero | cumplida |
| Predicción de anclaje (no inferioridad estricta) | límite inferior IC macro-F1 Modelo de control - público >= -0.01 | no cumplida |
| Predicción de objetivo (intervalo de confianza) | límite inferior IC macro-F1 Proposed sin segmentación - público > 0 (permite decir que le gana) | cumplida |

## ECE con IC basic (lectura secundaria)

IC basic (reverse-percentile) de ECE, lectura secundaria. El ECE remuestreado tiene sesgo hacia arriba, así que el IC percentil puede dejar el punto afuera (Modelo de control en val v7). El basic refleja los cuantiles alrededor del punto y puede salir de [0, 1]. Los veredictos primarios usan el IC percentil registrado y no cambian.

| Brazo o contraste | ECE / Δ ECE | IC basic | ¿Punto fuera del IC percentil? |
|---|---|---|---|
| Modelo de control | 0,0135 | [-0,0065; 0,0133] | sí |
| Proposed sin segmentación | 0,0325 | [0,0153; 0,0397] | no |
| DINOv2 público | 0,1307 | [0,1138; 0,1456] | no |
| Modelo de control − DINOv2 público | -0,1172 | [-0,1425; -0,1084] | no |
| Proposed sin segmentación − DINOv2 público | -0,0982 | [-0,1194; -0,0846] | no |
| Proposed sin segmentación − Modelo de control | 0,0190 | [0,0110; 0,0376] | no |

Predicción de tono leída con el IC basic: no cumplida (el veredicto primario no cambia).
