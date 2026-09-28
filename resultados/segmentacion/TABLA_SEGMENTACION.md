# Segmentación: estudiante final y pseudo-máscaras v7

Generado por `scripts/build_segmentation_table.py` desde JSON commiteados. No editar a mano: `--check` lo compara en CI.

## Estudiante final contra las máscaras de referencia

Checkpoint `student_r3/best.pt`, umbral 0,5, resolución 224. HD95 en píxeles a esa resolución. Las máscaras de referencia solo se leen para evaluar.

| Variante | Referencia | Métrica | Dirección | Media [IC 95 %] | Mediana | n |
|---|---|---|---|---|---|---:|
| student_r3 | ISIC 2018 | Dice | ↑ mayor es mejor | 0,7177 [0,7090; 0,7260] | 0,7925 | 2.594 |
| student_r3 | ISIC 2018 | IoU | ↑ mayor es mejor | 0,6017 [0,5924; 0,6104] | 0,6563 | 2.594 |
| student_r3 | ISIC 2018 | HD95 (px) | ↓ menor es mejor | 26,05 [25,04; 27,09] | 13,60 | 2.589 |
| student_r3 | HAM10000 | Dice | ↑ mayor es mejor | 0,7847 [0,7812; 0,7883] | 0,8519 | 10.015 |
| student_r3 | HAM10000 | IoU | ↑ mayor es mejor | 0,6784 [0,6745; 0,6827] | 0,7420 | 10.015 |
| student_r3 | HAM10000 | HD95 (px) | ↓ menor es mejor | 22,44 [21,95; 22,90] | 12,65 | 10.007 |
| student_r3 + TTA | ISIC 2018 | Dice | ↑ mayor es mejor | 0,7180 [0,7094; 0,7263] | 0,7917 | 2.594 |
| student_r3 + TTA | ISIC 2018 | IoU | ↑ mayor es mejor | 0,6020 [0,5926; 0,6109] | 0,6552 | 2.594 |
| student_r3 + TTA | ISIC 2018 | HD95 (px) | ↓ menor es mejor | 25,74 [24,69; 26,79] | 13,42 | 2.589 |
| student_r3 + TTA | HAM10000 | Dice | ↑ mayor es mejor | 0,7849 [0,7814; 0,7885] | 0,8527 | 10.015 |
| student_r3 + TTA | HAM10000 | IoU | ↑ mayor es mejor | 0,6788 [0,6748; 0,6832] | 0,7432 | 10.015 |
| student_r3 + TTA | HAM10000 | HD95 (px) | ↓ menor es mejor | 22,38 [21,89; 22,83] | 12,53 | 10.009 |

## Pseudo-máscaras v7 contra los umbrales registrados (docs/27, FASE 2)

Cobertura = `mean_foreground_fraction`. Δ contra la r0 publicada. Umbrales y veredictos calculados por el script desde los JSON.

| Variante | Cobertura media | Δ vs r0 | Sin región | Excluidas por área | Excluidas total | Umbral registrado | Veredicto |
|---|---:|---:|---:|---:|---:|---|---|
| r0 publicada | 0,13045 | +0,0000 | 670 | 4.252 | 4.922 | referencia | — |
| S1 central (reproduce r0) | 0,13045 | +0,0000 | 670 | 4.252 | 4.922 | igual a r0 | reproduce r0 |
| S1 unión | 0,13376 | +0,0033 | 670 | 4.258 | 4.928 | cobertura en 0,21-0,27 | no cumplida |
| S2(a) sin pelo | 0,13328 | +0,0028 | 659 | 4.893 | 5.552 | Δ ≥ 0,02 | no cumplida |
| S2' filtro por forma | 0,13119 | +0,0007 | 658 | 4.380 | 5.038 | (i) marcadas < 50 % y pintada < 0,03; (ii) Δ ≥ 0,02 | no se interpreta (falla la sanity (i)) |

## Control de precisión (500 imágenes fijas)

| Saliencia r0 | Cobertura media | Excluidas |
|---|---:|---:|
| bf16 (publicada, GPU) | 0,121653 | 47 |
| fp32 (recalculada, CPU) | 0,121637 | 47 |

Δ fp32 − bf16 = −0,000016; umbral |Δ| < 0,005: la r0 publicada vale como comparación.

## Remoción de pelo: S2(a) contra S2'

| Variante | Imágenes marcadas / total | % marcadas | Fracción media pintada | Inpaint masivo |
|---|---:|---:|---:|---:|
| S2(a) sin pelo | 58.453 / 58.457 | 100,0 % | 0,0663 | — |
| S2' filtro por forma | 48.200 / 58.457 | 82,5 % | 0,0103 | 0 |

«—» = el `cache_info.json` de esa variante no registra el conteo.
