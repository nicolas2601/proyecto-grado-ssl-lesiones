# Matrices de confusión de los modelos auto-supervisados

Generado por `scripts/build_bench_tables.py` desde los JSON del repositorio. No editar a mano. Alcance (docs/33, E4): solo los auto-supervisados, DINOv2 ViT-S/14, DINOv2 ViT-B/14 y SimSiam ResNet-50, congelados y con fine-tuning, más A71 y B71.

Cada celda tiene un `.json` (clases, conteos, matriz normalizada por fila, MCC, fuente) y un `.png` con dos paneles: conteos y normalizada por fila. Filas = clase real, columnas = clase predicha.

## Origen de los números

- Baselines: `matriz_confusion` de los JSON de `results/` (probe `logreg` para congelados, `finetune` para fine-tuning).
- A71 y B71: `resultados/clasificacion/v71/v71_final_eval.json` no trae matriz de confusión. Se derivó con el argmax de las probabilidades commiteadas (`v71_final_eval.probas.npz`); el JSON de cada celda guarda la accuracy del JSON y la recomputada desde la matriz, y un test exige que coincidan.

## Figura de diferencias frente a B71

No se genera figura de diferencias frente a B71: B71 se evalúa sobre la partición v6/v7 (n = 2.004) y los baselines sobre la de Kaggle (n = 2.014); no comparten partición ni imágenes, así que una resta celda a celda no tiene sentido. Se hará cuando exista B71 sobre la misma partición que un baseline.

## Matrices generadas

- `confusion/ham10000/dinov2_vits14__congelado.json` (+ `.png`): HAM10000, DINOv2 ViT-S/14, congelado; origen: matriz_confusion del JSON.
- `confusion/ham10000/dinov2_vits14__finetuning.json` (+ `.png`): HAM10000, DINOv2 ViT-S/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/ham10000/dinov2_vitb14__congelado.json` (+ `.png`): HAM10000, DINOv2 ViT-B/14, congelado; origen: matriz_confusion del JSON.
- `confusion/ham10000/dinov2_vitb14__finetuning.json` (+ `.png`): HAM10000, DINOv2 ViT-B/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/ham10000/simsiam_resnet50__congelado.json` (+ `.png`): HAM10000, SimSiam ResNet-50, congelado; origen: matriz_confusion del JSON.
- `confusion/ham10000/simsiam_resnet50__finetuning.json` (+ `.png`): HAM10000, SimSiam ResNet-50, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/ham10000/b71__congelado.json` (+ `.png`): HAM10000, B71 (propuesto), congelado; origen: argmax de las probabilidades commiteadas (el JSON v7.1 no trae matriz_confusion).
- `confusion/ham10000/a71__congelado.json` (+ `.png`): HAM10000, A71 (ablación sin término de tono), congelado; origen: argmax de las probabilidades commiteadas (el JSON v7.1 no trae matriz_confusion).
- `confusion/isic2019/dinov2_vits14__congelado.json` (+ `.png`): ISIC 2019, DINOv2 ViT-S/14, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2019/dinov2_vits14__finetuning.json` (+ `.png`): ISIC 2019, DINOv2 ViT-S/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/isic2019/dinov2_vitb14__congelado.json` (+ `.png`): ISIC 2019, DINOv2 ViT-B/14, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2019/dinov2_vitb14__finetuning.json` (+ `.png`): ISIC 2019, DINOv2 ViT-B/14, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/isic2019/simsiam_resnet50__congelado.json` (+ `.png`): ISIC 2019, SimSiam ResNet-50, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2019/simsiam_resnet50__finetuning.json` (+ `.png`): ISIC 2019, SimSiam ResNet-50, fine-tuning; origen: matriz_confusion del JSON.
- `confusion/isic2020/dinov2_vits14__congelado.json` (+ `.png`): ISIC 2020, DINOv2 ViT-S/14, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2020/dinov2_vitb14__congelado.json` (+ `.png`): ISIC 2020, DINOv2 ViT-B/14, congelado; origen: matriz_confusion del JSON.
- `confusion/isic2020/simsiam_resnet50__congelado.json` (+ `.png`): ISIC 2020, SimSiam ResNet-50, congelado; origen: matriz_confusion del JSON.

## Pendientes (sin matriz, no se inventa)

- ISIC 2020, DINOv2 ViT-S/14 (fine-tuning): no ejecutado todavia; G1.3 de docs/33 (fine-tuning en ISIC 2020).
- ISIC 2020, DINOv2 ViT-B/14 (fine-tuning): no ejecutado todavia; G1.3 de docs/33 (fine-tuning en ISIC 2020).
- ISIC 2020, SimSiam ResNet-50 (fine-tuning): no ejecutado todavia; G1.3 de docs/33 (fine-tuning en ISIC 2020).
- HAM10000, B71-limpio (congelado): pendiente; E1 de docs/33.
- HAM10000, A71-limpio (congelado): pendiente; E1 de docs/33.
- ISIC 2019, B71-limpio (congelado): pendiente; E1 de docs/33.
- ISIC 2019, B71 original (transductivo) (congelado): pendiente; E1 de docs/33.
- ISIC 2019, A71-limpio (congelado): pendiente; E1 de docs/33.
- ISIC 2019, A71 original (transductivo) (congelado): pendiente; E1 de docs/33.
- ISIC 2020, B71-limpio (congelado): pendiente; E1 de docs/33.
- ISIC 2020, B71 original (transductivo) (congelado): pendiente; E1 de docs/33.
- ISIC 2020, A71-limpio (congelado): pendiente; E1 de docs/33.
- ISIC 2020, A71 original (transductivo) (congelado): pendiente; E1 de docs/33.

Modelos incluidos: dinov2_vits14, dinov2_vitb14, simsiam_r50; condiciones: congelado, fine-tuning.
