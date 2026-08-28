# Tests de baseline por modelo (Kaggle)

Una carpeta por modelo de la tabla del equipo. Cada notebook es autocontenido:
se sube UN archivo a Kaggle y corre el test completo de ese modelo, con el
mismo protocolo para los ocho (asi los numeros son comparables entre si).

Generados por `scripts/build_baseline_tests.py` — no editar a mano; para
cambiar el protocolo se edita el builder y se regeneran los ocho.

## Modelos

| Carpeta | Modelo | Supervision | Preentrenamiento |
|---|---|---|---|
| `alexnet/` | AlexNet (2012) | Supervisado | ImageNet-1k |
| `vgg19/` | VGG19 (2014) | Supervisado | ImageNet-1k |
| `efficientnet_b0/` | EfficientNet-B0 (2019) | Supervisado | ImageNet-1k |
| `resnet50/` | ResNet-50 (2015) | Supervisado | ImageNet-1k |
| `vit_b16/` | ViT-B/16 (2021) | Supervisado | ImageNet-1k (torchvision) |
| `simsiam_r50/` | SimSiam ResNet-50 (2021) | SSL | ImageNet-1k sin etiquetas |
| `dinov2_vits14/` | DINOv2 ViT-S/14 (2023) | SSL | LVD-142M sin etiquetas |
| `dinov2_vitb14/` | DINOv2 ViT-B/14 (2023) | SSL | LVD-142M sin etiquetas — par justo de tamano vs ViT-B/16 |

Los supervisados son comparadores congelados. El proposal del proyecto es
full SSL: las etiquetas solo se usan en la evaluacion (probes y metricas),
nunca en el preentrenamiento de nuestro modelo.

## Como correr un test en Kaggle

1. kaggle.com -> Create -> Notebook -> File -> Import Notebook -> subir el
   `.ipynb` de la carpeta del modelo.
2. Add Input: buscar `Skin Cancer MNIST: HAM10000` (kmader).
3. Settings: Accelerator = GPU T4 x2 (o P100), Internet = ON (descarga pesos).
4. Dejar `SMOKE = True` y Run All (~3-5 min). Valida el pipeline; los numeros
   NO son resultados.
5. Si el humo pasa: `SMOKE = False` -> Save Version -> Save & Run All.
6. Salida: `results_<modelo>.json` en Output + tabla impresa con todas las
   metricas (accuracy, balanced accuracy, F1 macro, precision macro, recall
   macro, AUC OvR macro, ECE) para los tres probes (kNN, logreg, prototipos).
7. Trazabilidad: la ultima celda registra el run en MLflow
   (`/kaggle/working/mlruns`, backend de archivos). Es no-fatal: si mlflow no
   esta disponible, el test corre igual y el JSON ya quedo guardado. Para ver
   la UI en local: `MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri
   file:mlruns` (MLflow 3.x exige ese opt-in para el file store).

## Hiperparametros y configuracion (identicos en los ocho tests)

Todos quedan registrados en el JSON de salida y en MLflow por cada corrida.

| Parametro | Valor | Donde | Nota |
|---|---|---|---|
| `SEED` | 42 | celda config | fija split, submuestreo y sklearn |
| `BATCH` | 64 | celda config | extraccion de features |
| `RES` | 224 | celda backbone | resolucion de entrada, todos los modelos |
| Interpolacion | BILINEAR (BICUBIC solo EfficientNet-B0) | celda backbone | distribucion de entrenamiento de cada modelo |
| Split | StratifiedGroupKFold, 5 folds -> test = 20 % | celda data | agrupado por `lesion_id`, assert 0 fuga |
| `KNN_K` | 20 | celda config | vecinos del probe kNN |
| kNN metrica | coseno sobre vectores L2-normalizados, `weights="distance"` | celda probes | protocolo DINO/DINOv2 |
| LogReg | `max_iter=2000`, `C=1.0`, `class_weight="balanced"` | celda probes | una sola capa (linear probe) |
| Prototipos | centroide por clase L2-norm, softmax con temperatura 10 | celda probes | cercania coseno |
| Normalizacion | mean/std de ImageNet | celda config | `[0.485,0.456,0.406]/[0.229,0.224,0.225]` |
| `SMOKE` | True por defecto | celda config | submuestreo estratificado 8 %; poner False para la corrida real |

Checkpoints por modelo (pesos exactos, verificables):

| Modelo | Checkpoint |
|---|---|
| AlexNet | torchvision `AlexNet_Weights.IMAGENET1K_V1` |
| VGG19 | torchvision `VGG19_Weights.IMAGENET1K_V1` (sin BN) |
| EfficientNet-B0 | torchvision `EfficientNet_B0_Weights.IMAGENET1K_V1` |
| ResNet-50 | torchvision `ResNet50_Weights.IMAGENET1K_V2` |
| ViT-B/16 | torchvision `ViT_B_16_Weights.IMAGENET1K_V1` |
| SimSiam | oficial Facebook `checkpoint_0099.pth.tar` (100 ep, batch 256) |
| DINOv2 ViT-S/14 | timm `vit_small_patch14_dinov2.lvd142m` (fallback torch.hub) |
| DINOv2 ViT-B/14 | timm `vit_base_patch14_dinov2.lvd142m` (fallback torch.hub) |

## Protocolo (identico en los ocho — no cambiar por modelo)

- Backbone congelado: `requires_grad=False` + `model.eval()` (BatchNorm no se
  congela solo con requires_grad).
- Extraccion: GAP del ultimo bloque convolucional (CNN) / token CLS (ViT),
  resolucion 224, interpolacion propia de cada modelo (EfficientNet: BICUBIC).
- Split por `lesion_id` con StratifiedGroupKFold; assert de 0 lesiones
  compartidas entre train y val.
- kNN con vectores L2-normalizados y similitud coseno (protocolo DINO).
- Metrica principal: balanced accuracy y F1 macro. Ancla de literatura:
  Gessert (ganador ISIC 2019) = 0.636 de balanced accuracy en test oficial.

## Pendiente

- Bancos sobre ISIC 2019 y 2020 RESIZED (pedido de la direccion, 2026-08-28):
  generados en `notebooks/baselines_isic2019/` y `notebooks/baselines_isic2020/`
  como tarea BINARIA (benigno/maligno) — los datasets RESIZED solo traen
  `target` binario. Split: ISIC 2020 agrupa por `patient_id` real; ISIC 2019
  RESIZED trae `patient_id` sintetico, asi que va estratificado simple y se
  reporta como limite. Ademas los 3 bancos ahora sacan precision/F1 por clase,
  matriz de confusion e IC bootstrap 95% (re-correr el de HAM10000 para
  obtenerlas). La variante multiclase de ISIC 2019 sigue pendiente: requiere
  `andrewmvd/isic-2019` (ground truth real de 8 clases).

## Resultados de la corrida real (2026-08-27, SMOKE=False, seed 42)

Mejor probe = regresion logistica en los 8. JSONs completos en `results/baselines/`.

| # | Modelo | Tipo | dims | acc | bal_acc | F1 macro | Prec macro | Recall macro | AUC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ViT-B/16 | Supervisado | 768 | 0.761 | **0.629** | **0.605** | 0.601 | 0.629 | **0.933** | 0.115 |
| 2 | DINOv2 ViT-B/14 | SSL | 768 | 0.747 | 0.618 | 0.595 | 0.598 | 0.618 | 0.907 | 0.162 |
| 3 | AlexNet | Supervisado | 256 | 0.666 | 0.589 | 0.493 | 0.450 | 0.589 | 0.896 | 0.079 |
| 4 | EfficientNet-B0 | Supervisado | 1280 | 0.734 | 0.586 | 0.546 | 0.519 | 0.586 | 0.917 | 0.125 |
| 5 | DINOv2 ViT-S/14 | SSL | 384 | 0.711 | 0.575 | 0.547 | 0.543 | 0.575 | 0.905 | 0.149 |
| 6 | ResNet-50 | Supervisado | 2048 | 0.731 | 0.572 | 0.528 | 0.498 | 0.572 | 0.908 | 0.097 |
| 7 | VGG19 | Supervisado | 512 | 0.618 | 0.515 | 0.415 | 0.377 | 0.515 | 0.870 | 0.071 |
| 8 | SimSiam R50 | SSL | 2048 | 0.490 | 0.494 | 0.317 | 0.335 | 0.494 | 0.829 | 0.247 |

Tablas de bal_acc por probe (kNN/logreg/prototipos) y recall por las 7 clases:
ver la seccion de resultados del README principal del repo. Fuente de todo:
los JSON de `results/baselines/`.

Ancla de literatura: Gessert (ganador ISIC 2019) = 0.636 balanced accuracy.

Lecturas clave:
- Ningun backbone generico congelado alcanza el ancla; ViT-B/16 queda a 0.007.
- A igual tamano (86M), DINOv2 ViT-B/14 (SSL) queda a 0.011 del ViT-B/16
  supervisado y supera a todas las CNN supervisadas. El ViT-S/14 (21M, 384d)
  iguala en F1 a los supervisados grandes siendo 4x mas chico.
- SimSiam-ImageNet es el peor del banco: el SSL generico de ImageNet no
  transfiere a dermatoscopia. Motiva el v6: pre-entrenar SSL EN piel.
- kNN colapsa hacia `nv` en todos los modelos (desbalance 58:1); `df` (115
  imgs) es la clase debil transversal.
