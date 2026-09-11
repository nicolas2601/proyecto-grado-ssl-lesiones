# Banco de fine-tuning en el cluster (guia para la Dra. Karen)

Objetivo: tomar los 8 modelos baseline desde Hugging Face Hub, hacerles
fine-tuning end-to-end sobre los 2 datasets medicos (HAM10000 e ISIC 2019) con un protocolo identico, y dejar cada corrida registrada
(Weights & Biases + MLflow + JSON versionado en GitHub).

Son 16 corridas (8 modelos x 2 datasets). Cada una usa UNA GPU; el lanzador
las encola de a 4 en la cola de prioridad.

## Limitaciones que viajan con estos numeros

Se decidio (2026-09-04) correr el banco sobre **ISIC 2019** en lugar de ISIC 2020,
porque es el unico cuya tarea coincide con el ancla de Gessert (8 clases del
challenge oficial) y por lo tanto el unico comparable de frente contra ella.
Ese cambio trae dos limites que hay que reportar junto con los resultados:

1. **Split sin agrupacion por paciente.** HAM10000 se parte por `lesion_id` real
   e ISIC 2020 se partia por `patient_id` real. El release de ISIC 2019
   (`andrewmvd/isic-2019`) **no trae ninguna columna de paciente**, y el
   `patient_id` de la variante RESIZED es sintetico. El split de ISIC 2019 es
   entonces **estratificado simple**: dos imagenes del mismo paciente pueden
   caer en train y en val. Las metricas quedan con **sesgo optimista**, y el
   fine-tuning lo amplifica porque el modelo puede memorizar al paciente. Los
   numeros de ISIC 2019 **no son comparables 1:1** con los de HAM10000.
2. **Los dos datasets se solapan.** El conjunto de entrenamiento oficial de
   ISIC 2019 (25.331 imagenes) incluye a HAM10000. No son bancos independientes:
   una parte de ISIC 2019 son literalmente las mismas fotos de HAM10000, y al no
   haber `lesion_id` en ISIC 2019 esas fotos repetidas de una misma lesion
   tampoco se pueden agrupar.

Ambos limites quedan escritos en el campo `split` de cada JSON de resultados,
asi que viajan con los datos y no solo con este documento.

## 0. Lo que hay que tener a mano

| Item | De donde sale | Donde va |
|---|---|---|
| Repositorio | `git clone https://github.com/nicolas2601/proyecto-grado-ssl-lesiones` (rama `feature/v6-gpu-ssl-pretraining`) | `$HOME/proyecto-grado-ssl-lesiones` |
| Pesos preentrenados (sin internet) | `dermssl-weights-cache.tar.gz`, asset del Release `cluster-assets-v2` del repo | `$SCRATCH/dermssl/` |
| Datasets | HAM10000 e ISIC 2019 de Kaggle. Solo hay que dejarlos en la carpeta indicada en la seccion 0b | `$SCRATCH/dermssl/data/` |
| Python 3.11 | `module load python/3.11` o el que indiquen los administradores | - |

`$SCRATCH` es el filesystem de scratch del cluster. Si no existe esa variable,
el lanzador usa `$HOME/scratch`.

## 0b. Donde poner los datasets

El lanzador busca los archivos en `DATA_ROOT` (por defecto `$SCRATCH/dermssl/data`):

```
$SCRATCH/dermssl/data/
    ham10000/   HAM10000_metadata.csv              y las imagenes *.jpg (en cualquier subcarpeta)
    isic2019/   ISIC_2019_Training_GroundTruth.csv y las imagenes *.jpg (en cualquier subcarpeta)
```

Es el contenido tal cual de los datasets de Kaggle `kmader/skin-cancer-mnist-ham10000`
y `andrewmvd/isic-2019`; un enlace simbolico a la carpeta que
ya exista en el laboratorio es suficiente. Si `DATA_ROOT` apunta a otro sitio:
`DATA_ROOT=/ruta bash cluster/finetune/run_all.sh`.

**Importante para ISIC 2019:** tiene que ser `andrewmvd/isic-2019`, que es el
release con el ground truth multiclase (one-hot de 8 columnas: MEL, NV, BCC, AK,
BKL, DF, VASC, SCC; la columna UNK sale en cero y se descarta). El release
`nischaydnk/isic-2019-jpg-224x224-resized` **no sirve**: solo trae un target
binario y un `patient_id` sintetico.

## 1. Un solo comando

```bash
cd $HOME/proyecto-grado-ssl-lesiones
bash cluster/finetune/run_all.sh
```

El script hace, en orden y saltando lo que ya exista:

1. Verifica python3.11, git y si hay SLURM.
2. Crea el entorno virtual `$HOME/venvs/dermssl-v6` (`cluster/requirements.txt`).
3. Descomprime los pesos preentrenados en `$SCRATCH/dermssl/{hf-cache,torch-cache}`.
4. Verifica que los datasets esten en `$SCRATCH/dermssl/data/{ham10000,isic2019}` (o descomprime `dermssl-data.tar.gz` si esta ahi).
5. Corre un smoke test de 1 GPU (resnet50 sobre HAM10000, 1 epoca, pocos
   minutos). Si falla, se detiene ahi y no encola nada.
6. Encola el arreglo SLURM de 16 jobs (`cluster/finetune/finetune_bank.sbatch`).

Opciones: `--smoke-only` (parar tras el smoke), `--no-smoke`, `--dry-run`
(muestra que haria sin ejecutar).

Variables opcionales: `PRECISION=fp16` si las GPU son V100 o T4 (por defecto
`bf16`, para A100/H100); `WANDB_ENTITY=<equipo>` para que los runs caigan en
el team de W&B del laboratorio.

## 2. Seguimiento

```bash
squeue -u $USER                       # estado de los 24 jobs
tail -f logs/dermssl-ft-*_0.out       # log del job 0
python scripts/finetune_baseline.py --list-jobs   # que indice es que modelo/dataset
```

Cada job imprime por epoca `loss`, `val_bal_acc` y la mejor epoca, y al final
la tabla de metricas con IC bootstrap 95%.

Tiempo estimado por job en A100: 20-40 min (HAM10000, 8k imagenes), 60-90 min
(ISIC 2019, 20k imagenes de train). Banco completo: unas 16 GPU-horas, 4-6 horas
de reloj con 4 GPU en paralelo.

## 3. Salidas

| Que | Donde |
|---|---|
| Metricas completas (JSON) | `results/finetune/<dataset>/results_<modelo>_ft*.json` (dentro del repo) |
| Checkpoint del mejor epoch | `$SCRATCH/dermssl/finetune/runs/job<N>-<jobid>/ft_<modelo>*.pth` |
| Weights & Biases (offline) | `$SCRATCH/dermssl/wandb/wandb/offline-run-*` |
| MLflow (file store) | `$SCRATCH/dermssl/mlruns` |

El JSON es la fuente de verdad: trae accuracy, balanced accuracy, F1/precision/
recall macro y por clase, AUC, ECE, matriz de confusion, IC bootstrap, mejor
epoca, tiempo, fuente exacta de los pesos, commit del repo, host y GPU.

## 4. Publicar los resultados

Cuando los 16 jobs terminen:

```bash
bash cluster/finetune/publish_results.sh
```

Hace `git commit` + `git push` de `results/finetune/` (GitHub es el unico
destino de red que el cluster permite) e imprime el comando para subir los
runs de W&B desde una maquina con internet:

```bash
wandb login
wandb sync $SCRATCH/dermssl/wandb/wandb/offline-run-*
```

Proyecto W&B: `dermssl-baselines-finetune`. Si `WANDB_ENTITY` apunta al team
del laboratorio, Nicolas y Maria Paula lo ven al ser agregados al team; si no,
alcanza con compartir el proyecto como publico o por enlace.

## 5. Correr un solo job a mano

```bash
source $HOME/venvs/dermssl-v6/bin/activate
export HF_HOME=$SCRATCH/dermssl/hf-cache TORCH_HOME=$SCRATCH/dermssl/torch-cache HF_HUB_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/dermssl/wandb
python scripts/finetune_baseline.py --model dinov2_vitb14 --dataset isic2019 \
    --data-root $SCRATCH/dermssl/data --output-dir $SCRATCH/dermssl/finetune/manual \
    --results-dir results/finetune --mlflow-tracking-uri file:$SCRATCH/dermssl/mlruns
```

`--job-index N` (0..15) reemplaza a `--model/--dataset`. `--smoke` corre 1
epoca sobre un subconjunto para validar el cableado.

## 6. Protocolo (identico a los notebooks de Kaggle)

Definido en `src/dermssl/finetune/protocol.py` y verificado por tests.

| Item | Valor |
|---|---|
| Modelos | AlexNet, VGG19, EfficientNet-B0, ResNet-50, ViT-B/16, SimSiam R50, DINOv2 ViT-S/14, DINOv2 ViT-B/14 |
| Pesos | Hugging Face Hub via timm (`vgg19.tv_in1k`, `resnet50.tv2_in1k`, `efficientnet_b0.ra_in1k`, `vit_base_patch16_224.augreg_in1k`, `vit_small_patch14_dinov2.lvd142m`, `vit_base_patch14_dinov2.lvd142m`); excepciones documentadas: AlexNet (torchvision), SimSiam (checkpoint oficial FAIR) |
| Split | 5 folds (val 20 %). HAM10000: StratifiedGroupKFold agrupado por `lesion_id` real. ISIC 2019: estratificado simple, **sin agrupacion** por paciente (el release no la trae) - ver "Limitaciones que viajan con estos numeros" |
| Optimizador | AdamW, LR backbone 5e-5, LR cabeza 5e-4, weight decay 1e-4 |
| Schedule | coseno, 8 epocas, batch 32 (16 en VGG19, ViT-B/16, DINOv2 ViT-B/14) |
| Perdida | cross-entropy con pesos por clase |
| Precision | bf16 (A100/H100) o fp16 |
| Seleccion | mejor epoca por balanced accuracy de validacion |
| Semilla | 42 |

## 6b. Equivalencias con CT-Encoders

El runner acepta la misma forma de invocacion que `ibot/main_ibot.py` del
repositorio CT-Encoders del laboratorio, para no obligar a aprender un segundo
CLI. `--data_path` y `--output_dir` funcionan igual que `--data-root` y
`--output-dir`.

```bash
# CT-Encoders
python ibot/main_ibot.py --arch swin3D --data_path /../dataset --output_dir ./output_dir

# este banco, un job
python scripts/finetune_baseline.py --model resnet50 --dataset ham10000 \
    --data_path $SCRATCH/dermssl/data --output_dir ./output_dir
```

TensorBoard se activa solo y escribe dentro del `output_dir`, igual que el
`SummaryWriter` de CT-Encoders (`args.output_dir/tf_logs`); aca es
`<output_dir>/tb`, un subdirectorio por corrida. Se apaga con
`--tensorboard-dir none`. Para verlo:

```bash
tensorboard --logdir <output_dir>/tb
```

Dos diferencias deliberadas, por si llaman la atencion:

| Tema | CT-Encoders | Este banco | Por que |
|---|---|---|---|
| Paralelismo | DDP: un modelo repartido en N GPU | un modelo por GPU, N jobs en paralelo | son 16 corridas independientes de 20-90 min; repartir una sola entre 4 GPU no las termina antes y agrega una superficie de fallo (NCCL, sincronizacion) que aqui no compra nada |
| Registro | tensorboardX | `torch.utils.tensorboard` | misma salida y mismo visor; evita una dependencia mas en el entorno del cluster |

## 7. Si algo falla

- `weight caches missing`: falta el tar de pesos en `$SCRATCH/dermssl/`.
- `datasets missing`: falta el tar de datos, o `DATA_ROOT` apunta mal.
- `config error: ...`: el mensaje dice exactamente que argumento esta mal; se
  valida antes de cargar torch, asi que no gasta cola.
- OOM en un job: relanzar ese indice con `--batch 16` (ya es el valor por
  defecto en los tres modelos grandes) o `--batch 8`.
- Un job del arreglo caido: `sbatch --array=<N> cluster/finetune/finetune_bank.sbatch`
  relanza solo ese indice.
- `AVISO: sin wandb ni mlflow`: la corrida siguio y el JSON se escribio;
  revisar que el venv se creo con `cluster/requirements.txt`.
