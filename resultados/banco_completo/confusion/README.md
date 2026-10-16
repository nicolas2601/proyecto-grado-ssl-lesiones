# Matrices de confusión de los modelos auto-supervisados

Generado por `scripts/build_bench_tables.py` desde los JSON del repositorio. No editar a mano. Alcance (docs/33, E4): solo los auto-supervisados, DINOv2 ViT-S/14, DINOv2 ViT-B/14 y SimSiam ResNet-50, congelados y con fine-tuning, más control SSL sin consistencia de tono y Proposed sin segmentación (HAM10000) y, en ISIC, Proposed sin segmentación, pesos con exclusión de validación (canónico) y Proposed sin segmentación / control SSL sin consistencia de tono originales (sensibilidad: batch 16, transductivo).

Cada celda tiene un `.json` (clases, conteos, matriz normalizada por fila, MCC, fuente) y un `.png` con dos paneles: conteos y normalizada por fila. Filas = clase real, columnas = clase predicha.

## Origen de los números

- Baselines: `matriz_confusion` de los JSON de `results/` (probe `logreg` para congelados, `finetune` para fine-tuning).
- ISIC 2019 e ISIC 2020, Proposed sin segmentación, pesos con exclusión de validación y Proposed sin segmentación / control SSL sin consistencia de tono originales: `matriz_confusion` de la celda E1 (`resultados/banco_completo/isic_eval/...`); el JSON de cada matriz guarda n de la celda y n desde la matriz.
- control SSL sin consistencia de tono y Proposed sin segmentación (HAM10000): `resultados/clasificacion/v71/v71_final_eval.json` no trae matriz de confusión. Se derivó con el argmax de las probabilidades commiteadas (`v71_final_eval.probas.npz`); el JSON de cada celda guarda la accuracy del JSON y la recomputada desde la matriz, y un test exige que coincidan.

## Figura de diferencias frente a Proposed sin segmentación

No se genera figura de diferencias frente a Proposed sin segmentación: Proposed sin segmentación se evalúa sobre la partición v6/v7 (n = 2.004) y los baselines sobre la de Kaggle (n = 2.014); no comparten partición ni imágenes, así que una resta celda a celda no tiene sentido. En ISIC 2019 e ISIC 2020 Proposed sin segmentación, pesos con exclusión de validación sí comparte partición con los baselines, pero tampoco se genera la figura de diferencias (decisión de alcance).

## Matrices generadas

- `confusion/ham10000/dinov2_vits14__congelado.json` (+ `.png`): HAM10000, DINOv2 ViT-S/14, congelado; origen: matriz_confusion del JSON.
- `confusion/ham10000/dinov2_vits14__finetuning.json` (+ `.png`): HAM10000, DINOv2 ViT-S/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/ham10000/dinov2_vitb14__congelado.json` (+ `.png`): HAM10000, DINOv2 ViT-B/14, congelado; origen: matriz_confusion del JSON.
- `confusion/ham10000/dinov2_vitb14__finetuning.json` (+ `.png`): HAM10000, DINOv2 ViT-B/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/ham10000/simsiam_resnet50__congelado.json` (+ `.png`): HAM10000, SimSiam ResNet-50, congelado; origen: matriz_confusion del JSON.
- `confusion/ham10000/simsiam_resnet50__finetuning.json` (+ `.png`): HAM10000, SimSiam ResNet-50, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/ham10000/b71__congelado.json` (+ `.png`): HAM10000, Proposed sin segmentación (propuesto), congelado; origen: argmax de las probabilidades commiteadas (el JSON v7.1 no trae matriz_confusion).
- `confusion/ham10000/a71__congelado.json` (+ `.png`): HAM10000, control SSL sin consistencia de tono (ablación sin término de tono), congelado; origen: argmax de las probabilidades commiteadas (el JSON v7.1 no trae matriz_confusion).
- `confusion/ham10000/b71limpio__congelado.json` (+ `.png`): HAM10000, Proposed sin segmentación, pesos con exclusión de validación (canónico), congelado; origen: argmax de las probabilidades commiteadas de la sensibilidad de pesos limpios (cómputo canónico; análisis posterior, no es selección).
- `confusion/ham10000/a71limpio__congelado.json` (+ `.png`): HAM10000, control SSL sin consistencia de tono, pesos con exclusión de validación (canónico), congelado; origen: argmax de las probabilidades commiteadas de la sensibilidad de pesos limpios (cómputo canónico; análisis posterior, no es selección).
- `confusion/isic2019/dinov2_vits14__congelado.json` (+ `.png`): ISIC 2019, DINOv2 ViT-S/14, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2019/dinov2_vits14__finetuning.json` (+ `.png`): ISIC 2019, DINOv2 ViT-S/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/isic2019/dinov2_vitb14__congelado.json` (+ `.png`): ISIC 2019, DINOv2 ViT-B/14, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2019/dinov2_vitb14__finetuning.json` (+ `.png`): ISIC 2019, DINOv2 ViT-B/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/isic2019/simsiam_resnet50__congelado.json` (+ `.png`): ISIC 2019, SimSiam ResNet-50, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2019/simsiam_resnet50__finetuning.json` (+ `.png`): ISIC 2019, SimSiam ResNet-50, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/isic2019/b71limpio__congelado.json` (+ `.png`): ISIC 2019, Proposed sin segmentación, pesos con exclusión de validación (canónico), congelado; origen: matriz_confusion de la celda E1 (cómputo canónico: batch 64 + autocast).
- `confusion/isic2019/a71limpio__congelado.json` (+ `.png`): ISIC 2019, control SSL sin consistencia de tono, pesos con exclusión de validación (canónico), congelado; origen: matriz_confusion de la celda E1 (cómputo canónico: batch 64 + autocast).
- `confusion/isic2019/b71_original__congelado.json` (+ `.png`): ISIC 2019, Proposed sin segmentación original (canónico, transductivo), congelado; origen: matriz_confusion de la celda E1 (cómputo canónico: batch 64 + autocast).
- `confusion/isic2019/a71_original__congelado.json` (+ `.png`): ISIC 2019, control SSL sin consistencia de tono original (canónico, transductivo), congelado; origen: matriz_confusion de la celda E1 (cómputo canónico: batch 64 + autocast).
- `confusion/isic2019/b71_original_batch16__congelado.json` (+ `.png`): ISIC 2019, Proposed sin segmentación original (sensibilidad: batch 16, transductivo), congelado; origen: matriz_confusion de la celda E1 (sensibilidad: batch 16, transductivo).
- `confusion/isic2019/a71_original_batch16__congelado.json` (+ `.png`): ISIC 2019, control SSL sin consistencia de tono original (sensibilidad: batch 16, transductivo), congelado; origen: matriz_confusion de la celda E1 (sensibilidad: batch 16, transductivo).
- `confusion/isic2020/dinov2_vits14__congelado.json` (+ `.png`): ISIC 2020, DINOv2 ViT-S/14, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2020/dinov2_vits14__finetuning.json` (+ `.png`): ISIC 2020, DINOv2 ViT-S/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/isic2020/dinov2_vitb14__congelado.json` (+ `.png`): ISIC 2020, DINOv2 ViT-B/14, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2020/dinov2_vitb14__finetuning.json` (+ `.png`): ISIC 2020, DINOv2 ViT-B/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/isic2020/simsiam_resnet50__congelado.json` (+ `.png`): ISIC 2020, SimSiam ResNet-50, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2020/simsiam_resnet50__finetuning.json` (+ `.png`): ISIC 2020, SimSiam ResNet-50, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/isic2020/b71limpio__congelado.json` (+ `.png`): ISIC 2020, Proposed sin segmentación, pesos con exclusión de validación (canónico), congelado; origen: matriz_confusion de la celda E1 (cómputo canónico: batch 64 + autocast).
- `confusion/isic2020/a71limpio__congelado.json` (+ `.png`): ISIC 2020, control SSL sin consistencia de tono, pesos con exclusión de validación (canónico), congelado; origen: matriz_confusion de la celda E1 (cómputo canónico: batch 64 + autocast).
- `confusion/isic2020/b71_original__congelado.json` (+ `.png`): ISIC 2020, Proposed sin segmentación original (canónico, transductivo), congelado; origen: matriz_confusion de la celda E1 (cómputo canónico: batch 64 + autocast).
- `confusion/isic2020/a71_original__congelado.json` (+ `.png`): ISIC 2020, control SSL sin consistencia de tono original (canónico, transductivo), congelado; origen: matriz_confusion de la celda E1 (cómputo canónico: batch 64 + autocast).
- `confusion/isic2020/b71_original_batch16__congelado.json` (+ `.png`): ISIC 2020, Proposed sin segmentación original (sensibilidad: batch 16, transductivo), congelado; origen: matriz_confusion de la celda E1 (sensibilidad: batch 16, transductivo).
- `confusion/isic2020/a71_original_batch16__congelado.json` (+ `.png`): ISIC 2020, control SSL sin consistencia de tono original (sensibilidad: batch 16, transductivo), congelado; origen: matriz_confusion de la celda E1 (sensibilidad: batch 16, transductivo).

## Pendientes (sin matriz, no se inventa)


Modelos incluidos: dinov2_vits14, dinov2_vitb14, simsiam_r50; condiciones: congelado, fine-tuning.
