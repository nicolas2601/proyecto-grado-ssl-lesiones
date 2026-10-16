# Ranking por macro-F1 — todos los clasificadores de HAM10000

Generado por `scripts/build_ranking_table.py` desde los JSON del repositorio. No editar a mano.

## Cómo leer esta tabla

- Métrica: macro-F1 sobre HAM10000 val (mayor es mejor). IC 95 % bootstrap de cada fuente; «—» = la fuente no publica IC.
- Hay dos particiones de val, ambas agrupadas por `lesion_id`: la de Kaggle (baselines y fine-tuning) y la de v6/v7. Entre particiones las filas no son pareadas; «Frente al 1.º» compara intervalos, no es un test pareado. DINOv2 ViT-B/14 público congelado sirve de ancla: aparece en ambas.
- El fine-tuning supervisado elige su mejor época sobre val, así que su número es optimista.
- Parte de la ganancia de v7.1 viene del protocolo del probe (TTA, pooling, ajuste de logits), no del backbone: compare control SSL sin consistencia de tono y Proposed sin segmentación con «DINOv2 público con el protocolo de …».

## Ranking (mejor a peor)

| # | Modelo | Familia | Macro-F1 [IC 95 %] | Frente al 1.º | Val | Fuente |
|---|---|---|---|---|---|---|
| 1 | control SSL sin consistencia de tono (sopa experimento de control SSL sin consistencia de tono s42+s43, cls_attn, la0.5, TTA) | propio | 0,7229 [0,6795; 0,7592] | referencia (1.º) | v6/v7, val n = 2004 | `resultados/clasificacion/v71/v71_final_eval.json` |
| 2 | Proposed sin segmentación (experimento SSL con consistencia de tono s42, cls4, balanced, TTA) | propio | 0,7141 [0,6703; 0,7547] | IC se solapa | v6/v7, val n = 2004 | `resultados/clasificacion/v71/v71_final_eval.json` |
| 3 | ViT-B/16 fine-tuning | fine-tuning supervisado | 0,7058 [0,6593; 0,7431] | IC se solapa | Kaggle, val n = 2014 | `results/finetune/ham10000/results_vit_b16_ft.json` |
| 4 | ResNet-50 fine-tuning | fine-tuning supervisado | 0,6867 [0,6476; 0,7226] | IC se solapa | Kaggle, val n = 2014 | `results/finetune/ham10000/results_resnet50_ft.json` |
| 5 | DINOv2 público, su mejor protocolo v7.1 | congelado (pesos públicos) | 0,6857 [0,6412; 0,7208] | IC se solapa | v6/v7, val n = 2004 | `resultados/clasificacion/v71/v71_final_eval.json` |
| 6 | experimento de control SSL sin consistencia de tono (SSL sin término de tono), semilla 43 | propio | 0,6850 [0,6353; 0,7258] | IC se solapa | v6/v7, val n = 2004 | `resultados/clasificacion/v7/s43/final_eval.json` |
| 7 | experimento de control SSL sin consistencia de tono (SSL sin término de tono), semilla 44 | propio | 0,6843 [0,6398; 0,7248] | IC se solapa | v6/v7, val n = 2004 | `resultados/clasificacion/v7/s44/final_eval.json` |
| 8 | DINOv2 público con el protocolo de Proposed sin segmentación | congelado (pesos públicos) | 0,6837 [0,6381; 0,7202] | IC se solapa | v6/v7, val n = 2004 | `resultados/clasificacion/v71/v71_final_eval.json` |
| 9 | experimento SSL con consistencia de tono (SSL propuesto), semilla 44 | propio | 0,6725 [0,6267; 0,7139] | IC se solapa | v6/v7, val n = 2004 | `resultados/clasificacion/v7/s44/final_eval.json` |
| 10 | experimento SSL con consistencia de tono (SSL propuesto), semilla 43 | propio | 0,6638 [0,6152; 0,7052] | IC se solapa | v6/v7, val n = 2004 | `resultados/clasificacion/v7/s43/final_eval.json` |
| 11 | experimento SSL con consistencia de tono (SSL propuesto), semilla 42 | propio | 0,6606 [0,6157; 0,7040] | IC se solapa | v6/v7, val n = 2004 | `resultados/clasificacion/v7/final_eval.json` |
| 12 | EfficientNet-B0 fine-tuning | fine-tuning supervisado | 0,6503 [0,6074; 0,6870] | IC se solapa | Kaggle, val n = 2014 | `results/finetune/ham10000/results_efficientnet_b0_ft.json` |
| 13 | experimento de control SSL sin consistencia de tono (SSL sin término de tono), semilla 42 | propio | 0,6464 [0,5980; 0,6894] | IC se solapa | v6/v7, val n = 2004 | `resultados/clasificacion/v7/final_eval.json` |
| 14 | DINOv2 público con el protocolo de control SSL sin consistencia de tono | congelado (pesos públicos) | 0,6320 [0,5875; 0,6710] | IC separado | v6/v7, val n = 2004 | `resultados/clasificacion/v71/v71_final_eval.json` |
| 15 | VGG19 fine-tuning | fine-tuning supervisado | 0,6246 [0,5874; 0,6591] | IC separado | Kaggle, val n = 2014 | `results/finetune/ham10000/results_vgg19_ft.json` |
| 16 | DINOv2 público, probe v7 base | congelado (pesos públicos) | 0,6158 [0,5690; 0,6550] | IC separado | v6/v7, val n = 2004 | `resultados/clasificacion/v7/final_eval.json` |
| 17 | DINOv2 ViT-B/14 congelado + logreg | congelado (pesos públicos) | 0,5951 [0,5508; 0,6351] | IC separado | Kaggle, val n = 2014 | `results/baselines/results_dinov2_vitb14.json` |
| 18 | v6 SSL sin término de tono (ViT-B/14) | propio | 0,5856 — | sin IC | v6/v7, val n = 2004 | `resultados/clasificacion/bank.json` |
| 19 | AlexNet fine-tuning | fine-tuning supervisado | 0,5794 [0,5470; 0,6103] | IC separado | Kaggle, val n = 2014 | `results/finetune/ham10000/results_alexnet_ft.json` |
| 20 | DINOv2 ViT-S/14 congelado + logreg | congelado (pesos públicos) | 0,5474 [0,5037; 0,5838] | IC separado | Kaggle, val n = 2014 | `results/baselines/results_dinov2_vits14.json` |
| 21 | EfficientNet-B0 congelado + logreg | congelado (pesos públicos) | 0,5464 [0,5089; 0,5786] | IC separado | Kaggle, val n = 2014 | `results/baselines/results_efficientnet_b0.json` |
| 22 | ViT-B/16 congelado + logreg | congelado (pesos públicos) | 0,5402 [0,4971; 0,5795] | IC separado | Kaggle, val n = 2014 | `results/baselines/results_vit_b16.json` |
| 23 | SimSiam ResNet-50 fine-tuning | fine-tuning supervisado | 0,5323 [0,5007; 0,5634] | IC separado | Kaggle, val n = 2014 | `results/finetune/ham10000/results_simsiam_r50_ft.json` |
| 24 | ResNet-50 congelado + logreg | congelado (pesos públicos) | 0,5280 [0,4917; 0,5633] | IC separado | Kaggle, val n = 2014 | `results/baselines/results_resnet50.json` |
| 25 | v6 SSL propuesto (ViT-B/14) | propio | 0,5237 — | sin IC | v6/v7, val n = 2004 | `resultados/clasificacion/bank.json` |
| 26 | DINOv2 ViT-S/14 fine-tuning | fine-tuning supervisado | 0,4953 [0,4643; 0,5252] | IC separado | Kaggle, val n = 2014 | `results/finetune/ham10000/results_dinov2_vits14_ft.json` |
| 27 | AlexNet congelado + logreg | congelado (pesos públicos) | 0,4929 [0,4584; 0,5215] | IC separado | Kaggle, val n = 2014 | `results/baselines/results_alexnet.json` |
| 28 | DINOv2 ViT-B/14 fine-tuning | fine-tuning supervisado | 0,4268 [0,3970; 0,4545] | IC separado | Kaggle, val n = 2014 | `results/finetune/ham10000/results_dinov2_vitb14_ft.json` |
| 29 | VGG19 congelado + logreg | congelado (pesos públicos) | 0,4151 [0,3858; 0,4404] | IC separado | Kaggle, val n = 2014 | `results/baselines/results_vgg19.json` |
| 30 | SimSiam ResNet-50 congelado + logreg | congelado (pesos públicos) | 0,3169 [0,2939; 0,3405] | IC separado | Kaggle, val n = 2014 | `results/baselines/results_simsiam_r50.json` |

## Lectura

- 1.º: control SSL sin consistencia de tono (sopa experimento de control SSL sin consistencia de tono s42+s43, cls_attn, la0.5, TTA), 0,7229 [0,6795; 0,7592].
- Mejor supervisado: ViT-B/16 fine-tuning, 0,7058 [0,6593; 0,7431]. Los IC del 1.º y del mejor supervisado se solapan: empate estadístico, no victoria.
- Modelos externos con IC separado del 1.º (15): DINOv2 público con el protocolo de control SSL sin consistencia de tono, VGG19 fine-tuning, DINOv2 público, probe v7 base, DINOv2 ViT-B/14 congelado + logreg, AlexNet fine-tuning, DINOv2 ViT-S/14 congelado + logreg, EfficientNet-B0 congelado + logreg, ViT-B/16 congelado + logreg, SimSiam ResNet-50 fine-tuning, ResNet-50 congelado + logreg, DINOv2 ViT-S/14 fine-tuning, AlexNet congelado + logreg, DINOv2 ViT-B/14 fine-tuning, VGG19 congelado + logreg, SimSiam ResNet-50 congelado + logreg.

## Fuera del ranking: fine-tuning con LR de backbone reducido

Re-corrida exploratoria con otra tasa de aprendizaje; no forma parte del banco registrado.

| # | Modelo | Familia | Macro-F1 [IC 95 %] | Frente al 1.º | Val | Fuente |
|---|---|---|---|---|---|---|
| 1 | DINOv2 ViT-S/14 fine-tuning, LR backbone 5e-06 | fine-tuning supervisado | 0,6899 [0,6453; 0,7289] | — | Kaggle, val n = 2014 | `results/finetune_lrprobe/results_dinov2_vits14_ft_lrprobe.json` |
| 2 | DINOv2 ViT-B/14 fine-tuning, LR backbone 5e-06 | fine-tuning supervisado | 0,6840 [0,6468; 0,7155] | — | Kaggle, val n = 2014 | `results/finetune_lrprobe/results_dinov2_vitb14_ft_lrprobe.json` |
