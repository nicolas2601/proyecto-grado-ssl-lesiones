# Configuracion para el cluster (v6 SSL pretraining)

Para el banco de fine-tuning de baselines (lo que corre la Dra. Karen), la guia
de un solo comando esta en [cluster/finetune/README.md](finetune/README.md):
`bash cluster/finetune/run_all.sh`.

Respuesta al equipo de administracion: entorno virtual, dependencias, lanzadores
para las dos colas (4 GPUs prioridad / 8 GPUs con espera) y los dos tamanos de
ViT (pequeno y grande).

## 1. Ambiente virtual (caracteristicas)

| Item | Valor |
|---|---|
| Python | 3.11.x |
| Gestor | `venv` + pip (sin conda; si el cluster prefiere conda, la lista de `requirements.txt` es la misma) |
| CUDA | wheels cu121 (compatible con driver >= 530; funciona sobre CUDA 12.x del sistema) |
| Framework | PyTorch 2.4.1 + torchvision 0.19.1 |
| Precision | bf16 (Ampere o superior); fallback `--precision fp16` si son V100 |
| Tracking | MLflow file-store (sin servidor, sin red) + Weights & Biases en modo offline |
| Disco | ~5 GB venv + ~1 GB cache de pesos + datos (~30 GB pool de imagenes) en scratch |

Creacion (una vez):

```bash
bash cluster/setup_env.sh          # crea ~/venvs/dermssl-v6
```

Si el cluster NO tiene salida a PyPI: pedimos a los administradores un entorno
base con `torch>=2.2+cu12` (module/conda) y sobre el instalamos el resto de
`cluster/requirements.txt`, o que ellos construyan la lista exacta.

## 2. Codigo y datos (sin internet, solo GitHub)

- Codigo: `git clone https://github.com/nicolas2601/proyecto-grado-ssl-lesiones` (permitido).
- Pesos DINOv2: `torch.hub` los baja de `dl.fbaipublicfiles.com` (bloqueado).
  Solucion: cache pre-armado y publicado como asset de un Release de GitHub del
  propio repo. Ver `cluster/prepare_offline_assets.sh` (se corre en una maquina
  con internet; en el cluster solo se descarga el tar del Release y se apunta
  `TORCH_HOME` al cache).
- Datos: el laboratorio ya cuenta con HAM10000 e ISIC 2020; ver la seccion 0b de
  `cluster/finetune/README.md` para la carpeta esperada.

## 3. Corridas propuestas (grande y pequena)

Metodo propuesto de la tesis: `simsiam_dinov2` (objetivo SimSiam sobre backbone
DINOv2 preentrenado, adaptacion de dominio 100% libre de etiquetas).

Decision 2026-08-28: todo en la cola de prioridad de 4 GPUs (la espera de la
cola de 8 no compensa).

| Corrida | Backbone | GPUs | Batch/GPU | Batch efectivo | Presupuesto |
|---|---|---|---|---|---|
| Pequena | DINOv2 ViT-S/14 (21M) | 4 | 32 | 128 | 40 GPU-h |
| Grande | DINOv2 ViT-B/14 (86M) | 4 | 32 | 128 | 80 GPU-h |

```bash
sbatch cluster/pretrain_vits_4gpu.sbatch   # pequena
sbatch cluster/pretrain_vitb_4gpu.sbatch   # grande
```

Mismo batch efectivo (128) en ambas: la comparacion S-vs-B queda limpia (una
sola variable: el tamano del backbone). El learning rate escala solo con la
regla lineal ya implementada en `src/dermssl/pretrain/config.py`.

Validacion barata antes de encolar la corrida real (3 minutos, 1 GPU):

```bash
python scripts/pretrain.py --method simsiam_dinov2 --smoke \
    --image-root $SCRATCH/dermssl/pool --output-dir /tmp/smoke
```

## 4. Tracking: Weights & Biases + MLflow

Decision 2026-09-03: W&B es el dashboard que ve todo el equipo (el laboratorio
ya lo usa offline + sync); MLflow file-store queda como rastro local en el
cluster; los JSON de resultados se versionan en `results/` y se suben a GitHub
con `cluster/finetune/publish_results.sh`.

## 4b. Detalle previo (pretraining v6)

- MLflow file-store (`file:$SCRATCH/dermssl/mlruns`): ya integrado en el
  codigo, no necesita red ni servidor. Es la fuente de verdad de la corrida.
- W&B (lo que usa el laboratorio): el cluster no tiene internet, asi que los
  jobs corren con `WANDB_MODE=offline` y los runs se suben despues desde una
  maquina con internet: `wandb sync $SCRATCH/dermssl/wandb/wandb/offline-run-*`.

## 5. Preguntas pendientes para los administradores

1. Gestor de colas: asumimos SLURM (`sbatch`). Confirmar nombres de
   `--partition` y `--account`.
2. Modelo de GPU (A100/V100/otro) para fijar bf16 vs fp16 y ajustar batch.
3. Hay acceso a PyPI o mirror interno, o solo GitHub? (define como se
   construye el venv).
4. Limite de walltime por job en cada cola (para partir el presupuesto en
   varios jobs con resume si hace falta; el loop ya soporta checkpoint/resume).
5. Canal para subir el dataset (~30 GB) si los Release assets de GitHub no
   estan permitidos (redirigen a `objects.githubusercontent.com`).
