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
| `vit_b16/` | ViT-B/16 (2021) | Supervisado | ImageNet-1k (timm augreg_in1k) |
| `simsiam_r50/` | SimSiam ResNet-50 (2021) | SSL | ImageNet-1k sin etiquetas |
| `dinov2_vits14/` | DINOv2 ViT-S/14 (2023) | SSL | LVD-142M sin etiquetas |
| `dinov2_vitb14/` | DINOv2 ViT-B/14 (2023) | SSL | LVD-142M sin etiquetas — par justo de tamano vs ViT-B/16 |

Los supervisados son comparadores congelados. El proposal del proyecto es
full SSL: las etiquetas solo se usan en la evaluacion (probes y metricas),
nunca en el preentrenamiento de nuestro modelo.

## Fuente de pesos: Hugging Face Hub (decision del equipo, 2026-08-28)

Los modelos se cargan desde HF Hub via timm. Los tags `.tv_in1k` (vgg19) y
`.tv2_in1k` (resnet50) sirven exactamente los mismos tensores que torchvision.

**Los mismos tensores NO implican el mismo punto de extraccion** (hallazgo del
2026-09-03). Con `num_classes=0`, timm devuelve para VGG las pre-logits de 4096
dims (fc6/fc7 de la cabeza de clasificacion), no el GAP de 512 del ultimo
bloque conv que usa el resto del banco. La primera re-corrida de HAM10000 lo
delato como un salto silencioso de 512 a 4096 dims y -0.043 de balanced
accuracy. Corregido: vgg19 se lee con `gap(m.features)` y ambos runners
assertan `protocol.FEATURE_DIMS` antes de extraer.

Cambios de pesos reales:
`vit_b16` paso de torchvision a `augreg_in1k` (re-correr su test HAM) y
`efficientnet_b0` usa la receta `ra_in1k` de timm (verificar equivalencia o
re-correr). Excepciones sin pesos oficiales en HF: `alexnet` (torchvision) y
`simsiam_r50` (checkpoint oficial FAIR; pendiente mirror en repo HF del
equipo).

## Como correr un test en Kaggle

1. kaggle.com -> Create -> Notebook -> File -> Import Notebook -> subir el
   `.ipynb` de la carpeta del modelo.
2. Add Input: buscar `Skin Cancer MNIST: HAM10000` (kmader).
3. Settings: Accelerator = **GPU T4** (T4 x2 tambien sirve), Internet = ON.
   NO elegir P100: el PyTorch de Kaggle no trae kernels para su
   arquitectura (sm_60) y el test muere en el primer forward con
   `CUDA error: no kernel image is available`. `scripts/kaggle_push.py`
   ya fija `machine_shape: NvidiaTeslaT4` al subir por API.
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

Checkpoints por modelo (generado con
`python scripts/build_results_tables.py --weights`, fuente:
`src/dermssl/finetune/protocol.py`):

| Modelo | Checkpoint | Nota |
|---|---|---|
| AlexNet | torchvision `IMAGENET1K_V1` | sin pesos oficiales en HF Hub |
| VGG19 | Hugging Face via timm: `vgg19.tv_in1k` | mismos tensores que torchvision IMAGENET1K_V1 |
| EfficientNet-B0 | Hugging Face via timm: `efficientnet_b0.ra_in1k` | receta propia de timm (no es la de torchvision) |
| ResNet-50 | Hugging Face via timm: `resnet50.tv2_in1k` | mismos tensores que torchvision IMAGENET1K_V2 |
| ViT-B/16 | Hugging Face via timm: `vit_base_patch16_224.augreg_in1k` | supervisado SOLO con ImageNet-1k (comparador justo) |
| SimSiam R50 | `checkpoint_0099.pth.tar` | checkpoint oficial de FAIR, 100 epocas, batch 256 |
| DINOv2 ViT-S/14 | Hugging Face via timm: `vit_small_patch14_dinov2.lvd142m` | SSL sobre LVD-142M sin etiquetas |
| DINOv2 ViT-B/14 | Hugging Face via timm: `vit_base_patch14_dinov2.lvd142m` | SSL sobre LVD-142M sin etiquetas |

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

## Bancos relacionados

- Banco ISIC 2019: [`../baselines_isic2019/`](../baselines_isic2019/), tarea de
  ocho clases y la unica comparable directamente con el ancla de Gessert.
- Banco ISIC 2020: [`../baselines_isic2020/`](../baselines_isic2020/), tarea
  binaria agrupada por `patient_id`.
- Fine-tuning y probe de learning rate:
  [`../finetune/`](../finetune/), [`../finetune_isic2019/`](../finetune_isic2019/)
  y [`../finetune_lrprobe/`](../finetune_lrprobe/).

Las tablas finales se consultan en [`resultados/`](../../resultados/README.md)
y los JSON historicos en [`results/`](../../results/README.md). No mezclar una
corrida smoke con una corrida `SMOKE=False`.

## Resultados de la corrida real (2026-09-03, SMOKE=False, seed 42)

Tablas generadas con `python scripts/build_results_tables.py --bank baselines`.
JSONs completos en `results/baselines/`. El banco de ISIC 2020 esta en
`results/baselines_isic2020/`, con su tabla en el README principal del repo.

### Métricas completas (mejor probe de cada modelo)

| # | Modelo | Tipo | dims | acc | bal_acc | F1 macro | Prec macro | Recall macro | AUC | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | DINOv2 ViT-B/14 | SSL | 768 | **0.747** | **0.618** | **0.595** | **0.598** | **0.618** | 0.907 | 0.162 |
| 2 | AlexNet | Supervisado | 256 | 0.666 | 0.589 | 0.493 | 0.450 | 0.589 | 0.896 | 0.079 |
| 3 | EfficientNet-B0 | Supervisado | 1280 | 0.734 | 0.586 | 0.546 | 0.519 | 0.586 | 0.917 | 0.125 |
| 4 | DINOv2 ViT-S/14 | SSL | 384 | 0.711 | 0.575 | 0.547 | 0.543 | 0.575 | 0.905 | 0.149 |
| 5 | ResNet-50 | Supervisado | 2048 | 0.731 | 0.572 | 0.528 | 0.498 | 0.572 | 0.908 | 0.097 |
| 6 | ViT-B/16 | Supervisado | 768 | 0.723 | 0.567 | 0.540 | 0.530 | 0.567 | **0.920** | 0.168 |
| 7 | VGG19 | Supervisado | 512 | 0.618 | 0.515 | 0.415 | 0.377 | 0.515 | 0.870 | **0.071** |
| 8 | SimSiam R50 | SSL | 2048 | 0.490 | 0.494 | 0.317 | 0.335 | 0.494 | 0.829 | 0.247 |

### Intervalos de confianza (bootstrap 95%)

| Modelo | bal_acc | IC95 bal_acc | F1 macro | IC95 F1 macro | Separado del líder |
|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.618 | [0.576, 0.660] | 0.595 | [0.551, 0.635] | - |
| AlexNet | 0.589 | [0.548, 0.631] | 0.493 | [0.458, 0.521] | no |
| EfficientNet-B0 | 0.586 | [0.547, 0.623] | 0.546 | [0.509, 0.579] | no |
| DINOv2 ViT-S/14 | 0.575 | [0.531, 0.617] | 0.547 | [0.504, 0.584] | no |
| ResNet-50 | 0.572 | [0.533, 0.615] | 0.528 | [0.492, 0.563] | no |
| ViT-B/16 | 0.567 | [0.523, 0.612] | 0.540 | [0.497, 0.579] | no |
| VGG19 | 0.515 | [0.472, 0.555] | 0.415 | [0.386, 0.440] | sí |
| SimSiam R50 | 0.494 | [0.453, 0.539] | 0.317 | [0.294, 0.341] | sí |

### Balanced accuracy por probe

| Modelo | kNN | LogReg | Prototipos |
|---|---|---|---|
| DINOv2 ViT-B/14 | 0.311 | 0.618 | 0.458 |
| AlexNet | 0.402 | 0.589 | 0.482 |
| EfficientNet-B0 | 0.328 | 0.586 | 0.475 |
| DINOv2 ViT-S/14 | 0.318 | 0.575 | 0.499 |
| ResNet-50 | 0.328 | 0.572 | 0.437 |
| ViT-B/16 | 0.307 | 0.567 | 0.453 |
| VGG19 | 0.265 | 0.515 | 0.405 |
| SimSiam R50 | 0.338 | 0.494 | 0.491 |

### Recall por clase (mejor probe)

| Modelo | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.574 | 0.770 | 0.615 | 0.273 | 0.545 | 0.814 | 0.735 |
| AlexNet | 0.588 | 0.710 | 0.556 | 0.273 | 0.590 | 0.701 | 0.706 |
| EfficientNet-B0 | 0.603 | 0.760 | 0.580 | 0.136 | 0.505 | 0.810 | 0.706 |
| DINOv2 ViT-S/14 | 0.515 | 0.640 | 0.580 | 0.273 | 0.532 | 0.782 | 0.706 |
| ResNet-50 | 0.412 | 0.670 | 0.571 | 0.182 | 0.631 | 0.801 | 0.735 |
| ViT-B/16 | 0.529 | 0.590 | 0.600 | 0.318 | 0.581 | 0.795 | 0.559 |
| VGG19 | 0.471 | 0.520 | 0.478 | 0.182 | 0.514 | 0.674 | 0.765 |
| SimSiam R50 | 0.647 | 0.610 | 0.151 | 0.273 | 0.459 | 0.525 | 0.794 |

### Precisión por clase (mejor probe)

| Modelo | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.453 | 0.621 | 0.432 | 0.545 | 0.412 | 0.945 | 0.781 |
| AlexNet | 0.301 | 0.518 | 0.409 | 0.077 | 0.388 | 0.955 | 0.500 |
| EfficientNet-B0 | 0.451 | 0.571 | 0.476 | 0.143 | 0.380 | 0.929 | 0.686 |
| DINOv2 ViT-S/14 | 0.315 | 0.552 | 0.442 | 0.429 | 0.348 | 0.940 | 0.774 |
| ResNet-50 | 0.315 | 0.493 | 0.494 | 0.154 | 0.440 | 0.933 | 0.658 |
| ViT-B/16 | 0.400 | 0.440 | 0.504 | 0.318 | 0.389 | 0.929 | 0.731 |
| VGG19 | 0.227 | 0.311 | 0.408 | 0.053 | 0.339 | 0.934 | 0.366 |
| SimSiam R50 | 0.205 | 0.299 | 0.463 | 0.053 | 0.295 | 0.948 | 0.086 |

### F1 por clase (mejor probe)

| Modelo | akiec | bcc | bkl | df | mel | nv | vasc |
|---|---|---|---|---|---|---|---|
| DINOv2 ViT-B/14 | 0.506 | 0.688 | 0.507 | 0.364 | 0.469 | 0.875 | 0.758 |
| AlexNet | 0.398 | 0.599 | 0.471 | 0.120 | 0.468 | 0.809 | 0.585 |
| EfficientNet-B0 | 0.516 | 0.652 | 0.523 | 0.140 | 0.433 | 0.865 | 0.696 |
| DINOv2 ViT-S/14 | 0.391 | 0.593 | 0.502 | 0.333 | 0.421 | 0.854 | 0.738 |
| ResNet-50 | 0.357 | 0.568 | 0.529 | 0.167 | 0.519 | 0.862 | 0.694 |
| ViT-B/16 | 0.456 | 0.504 | 0.548 | 0.318 | 0.466 | 0.856 | 0.633 |
| VGG19 | 0.306 | 0.390 | 0.440 | 0.082 | 0.409 | 0.783 | 0.495 |
| SimSiam R50 | 0.311 | 0.401 | 0.228 | 0.088 | 0.359 | 0.675 | 0.155 |

Ancla de literatura: Gessert (ganador ISIC 2019) = 0.636 balanced accuracy.

Lecturas clave:
- Ningun backbone generico congelado alcanza el ancla.
- A igual tamano (86M), DINOv2 ViT-B/14 (SSL) queda 0.051 por encima del ViT-B/16
  supervisado con pesos `augreg_in1k`. Los intervalos se solapan, asi que la
  lectura honesta es que la ventaja del supervisado desaparece, no que el SSL gane.
- SimSiam-ImageNet es ultimo y su intervalo NO toca al del lider: esa diferencia
  si es real. El SSL generico de ImageNet no transfiere a dermatoscopia, que es
  lo que motiva el v6 (pre-entrenar SSL en piel).
- kNN colapsa hacia `nv` en todos los modelos (desbalance 58:1); `df` (115 imgs)
  es la clase debil transversal.
- El punto de extraccion de VGG19 se corrigio el 2026-09-03: timm con
  `num_classes=0` devolvia las pre-logits de 4096 dims, no el GAP convolucional.
  Ambos runners ahora assertan `protocol.FEATURE_DIMS` antes de extraer.
