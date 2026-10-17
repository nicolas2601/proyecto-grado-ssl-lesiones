# v7: las tres semillas

Generado desde `final_eval.json` (semilla 42), `s43/final_eval.json` (semilla 43) y `s44/final_eval.json` (semilla 44). No editar a mano: `--check` lo compara en CI. Regla de lectura en docs/28 §4.1.

| Brazo | Métrica | repetición 42 | repetición 43 | repetición 44 | Media | Rango (máx − mín) |
|---|---|---|---|---|---|---|
| Modelo de control | Macro-F1 | 0,6464 | 0,6850 | 0,6843 | 0,6719 | +0,0386 |
| Modelo de control | Bal-acc | 0,6021 | 0,6457 | 0,6445 | 0,6308 | +0,0436 |
| Modelo de control | AUC | 0,9619 | 0,9662 | 0,9639 | 0,9640 | +0,0043 |
| Modelo de control | ECE | 0,0135 | 0,0202 | 0,0239 | 0,0192 | +0,0103 |
| Proposed sin segmentación | Macro-F1 | 0,6606 | 0,6638 | 0,6725 | 0,6656 | +0,0119 |
| Proposed sin segmentación | Bal-acc | 0,6327 | 0,6119 | 0,6405 | 0,6284 | +0,0286 |
| Proposed sin segmentación | AUC | 0,9629 | 0,9623 | 0,9643 | 0,9632 | +0,0019 |
| Proposed sin segmentación | ECE | 0,0325 | 0,0178 | 0,0204 | 0,0236 | +0,0147 |
| DINOv2 público | Macro-F1 | 0,6158 | 0,6158 | 0,6158 | 0,6158 | 0,0000 |
| DINOv2 público | Bal-acc | 0,5924 | 0,5924 | 0,5924 | 0,5924 | 0,0000 |
| DINOv2 público | AUC | 0,9257 | 0,9257 | 0,9257 | 0,9257 | 0,0000 |
| DINOv2 público | ECE | 0,1307 | 0,1307 | 0,1307 | 0,1307 | 0,0000 |

| Contraste | Métrica | repetición 42 [IC 95 %] | repetición 43 [IC 95 %] | repetición 44 [IC 95 %] | Media |
|---|---|---|---|---|---|
| Modelo de control − DINOv2 público | Macro-F1 | +0,0306 [-0,0137; +0,0744] | +0,0692 [+0,0292; +0,1125] | +0,0684 [+0,0245; +0,1105] | +0,0561 |
| Modelo de control − DINOv2 público | Bal-acc | +0,0097 [-0,0346; +0,0549] | +0,0533 [+0,0167; +0,0952] | +0,0521 [+0,0096; +0,0943] | +0,0384 |
| Modelo de control − DINOv2 público | AUC | +0,0362 [+0,0258; +0,0482] | +0,0405 [+0,0307; +0,0516] | +0,0382 [+0,0286; +0,0482] | +0,0383 |
| Modelo de control − DINOv2 público | ECE | -0,1172 [-0,1260; -0,0919] | -0,1105 [-0,1241; -0,0854] | -0,1069 [-0,1215; -0,0832] | -0,1115 |
| Proposed sin segmentación − DINOv2 público | Macro-F1 | +0,0447 [+0,0080; +0,0837] | +0,0479 [+0,0051; +0,0920] | +0,0567 [+0,0142; +0,0998] | +0,0498 |
| Proposed sin segmentación − DINOv2 público | Bal-acc | +0,0403 [+0,0024; +0,0777] | +0,0195 [-0,0248; +0,0627] | +0,0481 [+0,0017; +0,0953] | +0,0360 |
| Proposed sin segmentación − DINOv2 público | AUC | +0,0372 [+0,0289; +0,0469] | +0,0366 [+0,0265; +0,0480] | +0,0385 [+0,0292; +0,0495] | +0,0375 |
| Proposed sin segmentación − DINOv2 público | ECE | -0,0982 [-0,1119; -0,0771] | -0,1129 [-0,1237; -0,0857] | -0,1103 [-0,1218; -0,0879] | -0,1071 |
| Proposed sin segmentación − Modelo de control | Macro-F1 | +0,0141 [-0,0190; +0,0539] | -0,0213 [-0,0584; +0,0156] | -0,0117 [-0,0476; +0,0271] | -0,0063 |
| Proposed sin segmentación − Modelo de control | Bal-acc | +0,0306 [-0,0055; +0,0691] | -0,0338 [-0,0739; +0,0044] | -0,0040 [-0,0409; +0,0385] | -0,0024 |
| Proposed sin segmentación − Modelo de control | AUC | +0,0010 [-0,0045; +0,0063] | -0,0039 [-0,0074; -0,0003] | +0,0004 [-0,0049; +0,0065] | -0,0008 |
| Proposed sin segmentación − Modelo de control | ECE | +0,0190 [+0,0003; +0,0270] | -0,0024 [-0,0152; +0,0141] | -0,0034 [-0,0180; +0,0136] | +0,0044 |

| Predicción | repetición 42 | repetición 43 | repetición 44 |
|---|---|---|---|
| Predicción de anclaje | cumplida | cumplida | cumplida |
| Predicción de tono | no cumplida | no cumplida | no cumplida |
| Predicción de objetivo | cumplida | cumplida | cumplida |
| Predicción de anclaje (no inferioridad estricta) | no cumplida | cumplida | cumplida |
| Predicción de objetivo (intervalo de confianza) | cumplida | cumplida | cumplida |

Predicción de objetivo (intervalo de confianza) con las tres semillas: robusto a la semilla.
