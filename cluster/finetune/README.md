# Banco de fine-tuning en el cluster (guia para la Dra. Karen)

Objetivo: tomar los 8 modelos baseline desde Hugging Face Hub, hacerles
fine-tuning end-to-end sobre los 3 datasets medicos (HAM10000, ISIC 2019,
ISIC 2020) con un protocolo identico, y dejar cada corrida registrada
(Weights & Biases + MLflow + JSON versionado en GitHub).

Son 24 corridas (8 modelos x 3 datasets). Cada una usa UNA GPU; el lanzador
las encola de a 4 en la cola de prioridad.

## 0. Lo que hay que tener a mano

| Item | De donde sale | Donde va |
|---|---|---|
| Repositorio | `git clone https://github.com/nicolas2601/proyecto-grado-ssl-lesiones` (rama `feature/v6-gpu-ssl-pretraining`) | `$HOME/proyecto-grado-ssl-lesiones` |
| Pesos preentrenados (sin internet) | `dermssl-weights-cache.tar.gz`, asset del Release `cluster-assets-v2` del repo | `$SCRATCH/dermssl/` |
| Datasets | `dermssl-data.tar.gz` (lo prepara Nicolas con `cluster/prepare_datasets.sh`, ~4 GB) | `$SCRATCH/dermssl/` |
| Python 3.11 | `module load python/3.11` o el que indiquen los administradores | - |

`$SCRATCH` es el filesystem de scratch del cluster. Si no existe esa variable,
el lanzador usa `$HOME/scratch`.

## 1. Un solo comando

```bash
cd $HOME/proyecto-grado-ssl-lesiones
bash cluster/finetune/run_all.sh
```

El script hace, en orden y saltando lo que ya exista:

1. Verifica python3.11, git y si hay SLURM.
2. Crea el entorno virtual `$HOME/venvs/dermssl-v6` (`cluster/requirements.txt`).
3. Descomprime los pesos preentrenados en `$SCRATCH/dermssl/{hf-cache,torch-cache}`.
4. Descomprime los datasets en `$SCRATCH/dermssl/data/{ham10000,isic2019,isic2020}`.
5. Corre un smoke test de 1 GPU (resnet50 sobre HAM10000, 1 epoca, pocos
   minutos). Si falla, se detiene ahi y no encola nada.
6. Encola el arreglo SLURM de 24 jobs (`cluster/finetune/finetune_bank.sbatch`).

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
(ISIC 2020, 26k imagenes). Banco completo: unas 24 GPU-horas, 6-8 horas de
reloj con 4 GPU en paralelo.

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

Cuando los 24 jobs terminen:

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
python scripts/finetune_baseline.py --model dinov2_vitb14 --dataset isic2020 \
    --data-root $SCRATCH/dermssl/data --output-dir $SCRATCH/dermssl/finetune/manual \
    --results-dir results/finetune --mlflow-tracking-uri file:$SCRATCH/dermssl/mlruns
```

`--job-index N` (0..23) reemplaza a `--model/--dataset`. `--smoke` corre 1
epoca sobre un subconjunto para validar el cableado.

## 6. Protocolo (identico a los notebooks de Kaggle)

Definido en `src/dermssl/finetune/protocol.py` y verificado por tests.

| Item | Valor |
|---|---|
| Modelos | AlexNet, VGG19, EfficientNet-B0, ResNet-50, ViT-B/16, SimSiam R50, DINOv2 ViT-S/14, DINOv2 ViT-B/14 |
| Pesos | Hugging Face Hub via timm (`vgg19.tv_in1k`, `resnet50.tv2_in1k`, `efficientnet_b0.ra_in1k`, `vit_base_patch16_224.augreg_in1k`, `vit_small_patch14_dinov2.lvd142m`, `vit_base_patch14_dinov2.lvd142m`); excepciones documentadas: AlexNet (torchvision), SimSiam (checkpoint oficial FAIR) |
| Split | StratifiedGroupKFold 5 folds (test 20 %), agrupado por `lesion_id` (HAM) o `patient_id` (ISIC 2020); ISIC 2019 RESIZED estratificado simple (su `patient_id` es sintetico) |
| Optimizador | AdamW, LR backbone 5e-5, LR cabeza 5e-4, weight decay 1e-4 |
| Schedule | coseno, 8 epocas, batch 32 (16 en VGG19, ViT-B/16, DINOv2 ViT-B/14) |
| Perdida | cross-entropy con pesos por clase |
| Precision | bf16 (A100/H100) o fp16 |
| Seleccion | mejor epoca por balanced accuracy de validacion |
| Semilla | 42 |

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
