# Camino B: solo CNN, preentrenamiento contrastivo y saliencia CAM

Estado (2026-09-19): implementado y probado en CPU con torch 2.14; **sin ejecutar aún en la GPU del clúster**. Primer paso obligatorio: el smoke de la sección 1. Las decisiones y desviaciones respecto a USL-Net y CCAM están en [DISENO.md](DISENO.md).

Idea: al estilo de USL-Net (Li et al., 2024). Aprende solo de nuestras ~68k imágenes sin etiquetar (HAM10000, ISIC 2019, ISIC 2020), sin pesos externos (salvo `--init-weights`, opcional y documentado como desviación) y sin máscaras ni etiquetas. Las máscaras de ISIC 2018, PH2 o Dermofit solo las abre la evaluación, después del entrenamiento.

Flujo: `cache` → MoCo v2 (`train_moco.py`) → CCAM y saliencia (`train_ccam.py`) → pseudo-máscaras con módulo de incertidumbre (`make_pseudo_masks.py`) → U-Net estudiante (`train_student.py`) → re-etiquetado con las predicciones y nuevas rondas → evaluación (`evaluate_segmentation.py`).

| Carpeta | Contenido |
|---|---|
| `preentrenamiento-contrastivo/` | `train_moco.py` y `contrastive_backbone.py` (ResNet50 desde cero, MoCo v2, interfaz de proveedor de features) |
| `pseudo-mascaras/` | `train_ccam.py`, `ccam_engine.py`, `make_pseudo_masks.py`, `uncertainty_module.py` (umbrales 0,35/0,65 y componente central, en numpy/scipy puro) |
| `entrenamiento/` | `run_pipeline.py`: encadena todas las etapas y las rondas |
| `../comun/` | `datos/prepare_cache.py`, `estudiante/`, `evaluacion/`, `nucleo/` (utilidades compartidas), `configs/` |
| `resultados/` | Métricas livianas (JSON/CSV) y figuras pequeñas |
| `salidas/` | Artefactos pesados (`.npy`, `.pt`). Ignorada por git |

Todos los comandos se ejecutan desde la raíz del repositorio. Sin `--config`, el driver usa `comun/configs/piloto.json`.

## 1. Verificar la instalación (smoke, menos de 2 minutos)

```bash
python plan-segmentacion/comun/nucleo/check_env.py          # versiones, GPU, bf16, /dev/shm
bash plan-segmentacion/camino-b-cnn/smoke.sh                # 64 imágenes sintéticas, 2 rondas, evaluación sintética
bash plan-segmentacion/camino-b-cnn/smoke.sh --nproc-per-node 2   # lo mismo con 2 GPU (DDP)
```

Debe terminar con `--- evaluate: ... (codigo 0)` y dejar `salidas/smoke/evaluacion_synthetic/summary.json`. Los números del smoke no significan nada (el modelo casi no entrena). Dependencias mínimas: `torch`, `torchvision`, `numpy`, `scipy`, `Pillow`.

## 2. Piloto completo (10 épocas)

Con el driver (recomendado):

```bash
python plan-segmentacion/camino-b-cnn/entrenamiento/run_pipeline.py \
  --workdir plan-segmentacion/camino-b-cnn/salidas/piloto \
  --root ham10000=/RUTA/ham10000 --root isic2019=/RUTA/isic2019 --root isic2020=/RUTA/isic2020 \
  --eval-image-dir /RUTA/isic2018/images --eval-mask-dir /RUTA/isic2018/masks --eval-name isic2018
# 2 GPU: agregar --nproc-per-node 2
# Rondas: --rounds N --epochs-per-round M sobrescriben el preset
# Solo algunas etapas: --stages contrastive ccam pseudo   (orden fijo del pipeline)
# Ver los comandos sin ejecutar: --dry-run
```

Etapa por etapa (equivalente; `$W` es la carpeta de salidas):

```bash
W=plan-segmentacion/camino-b-cnn/salidas/piloto
python plan-segmentacion/comun/datos/prepare_cache.py --output-dir $W/cache --resolution 224 \
  --root ham10000=/RUTA/ham10000 --root isic2019=/RUTA/isic2019 --root isic2020=/RUTA/isic2020

# 1 GPU
python plan-segmentacion/camino-b-cnn/preentrenamiento-contrastivo/train_moco.py \
  --cache-dir $W/cache --output-dir $W/moco --epochs 10
# 2 GPU
torchrun --nproc_per_node=2 plan-segmentacion/camino-b-cnn/preentrenamiento-contrastivo/train_moco.py \
  --cache-dir $W/cache --output-dir $W/moco --epochs 10

torchrun --nproc_per_node=2 plan-segmentacion/camino-b-cnn/pseudo-mascaras/train_ccam.py \
  --cache-dir $W/cache --backbone $W/moco/backbone.pt --output-dir $W/ccam --epochs 3
python plan-segmentacion/camino-b-cnn/pseudo-mascaras/make_pseudo_masks.py \
  --saliency $W/ccam/saliency.npy --output-dir $W/pseudo_r0 --source ccam

torchrun --nproc_per_node=2 plan-segmentacion/comun/estudiante/train_student.py \
  --cache-dir $W/cache --pseudo-masks $W/pseudo_r0/pseudo_masks.npy \
  --output-dir $W/student_r1 --epochs 10 --no-predict
# Con más rondas: quitar --no-predict y re-etiquetar
#   make_pseudo_masks.py --saliency $W/student_r1/predictions.npy --output-dir $W/pseudo_r1 --source student

python plan-segmentacion/comun/evaluacion/evaluate_segmentation.py \
  --checkpoint $W/student_r1/best.pt --image-dir /RUTA/isic2018/images \
  --mask-dir /RUTA/isic2018/masks --output-dir $W/evaluacion_isic2018 --hausdorff
```

`--selection last` (en `train_student.py`) elige la última época en vez de la mejor pérdida de pseudo-validación.

## 3. Salidas esperadas

| Ruta | Contenido |
|---|---|
| `cache/` | `images.npy` (~10,3 GB a 224), `manifest.csv`, `cache_info.json` (conteos, duplicados descartados, partición `val` de ~2 %) |
| `moco/` | `backbone.pt`, `last.pt`, `metrics.jsonl`, `timing.json`, `run_config.json` |
| `ccam/` | `saliency.npy` (float16, ~6,8 GB), `last.pt`, `metrics.jsonl`, `timing.json` |
| `pseudo_rN/` | `pseudo_masks.npy` (uint8: 0 fondo, 1 lesión, 255 ignorar), `pseudo_masks_stats.json` (`excluded_fraction`, fracción de primer plano) |
| `student_rN/` | `best.pt`, `last.pt`, `metrics.jsonl` (pérdida de entrenamiento y de pseudo-validación), `timing.json`, `predictions.npy` (solo si hay ronda siguiente) |
| `evaluacion_<nombre>/` | `per_image.csv` y `summary.json` (Dice, IoU, exactitud, sensibilidad, especificidad, opcional HD95) |

Cada época registra `seconds`, `images` e `images_per_second`. Para dimensionar el final, con el rendimiento medido en `timing.json`: horas = imágenes procesadas / (imágenes por segundo x 3600), y épocas = imágenes del presupuesto / imágenes del pool. Completar `comun/configs/final.json` (marcadores entre `<>`); el driver se niega a correr mientras queden marcadores.

## 4. Reanudar y qué enviar

Si un proceso se interrumpe, repetir exactamente el mismo comando: cada etapa detecta `last.pt` en su carpeta y continúa desde la época siguiente (`--no-resume` empieza de cero). Las etapas ya terminadas se vuelven a ejecutar con el driver salvo que se use `--stages`.

Enviar solo archivos pequeños: `*/timing.json`, `*/metrics.jsonl`, `*/run_config.json`, `cache/cache_info.json`, `pseudo_r*/pseudo_masks_stats.json`, `evaluacion_*/summary.json` y `per_image.csv`, y la salida de `check_env.py`. Los `.npy` y `.pt` se quedan en `salidas/`.

## 5. Pendientes y riesgos

- USL-Net usa 5 rondas de 300 épocas. Con 10 épocas de piloto no lo reproduce: si rinde peor, puede ser el presupuesto y no la idea.
- Sin medir en GPU real: velocidad, memoria (si hay error de memoria, reducir `batch_size` en el preset), bf16 y NCCL con 2 GPU. El smoke lo verifica.
- La selección de checkpoint por pérdida de pseudo-validación mide acuerdo con las pseudo-etiquetas, no calidad real (ver DISENO.md).
- Confirmar en el paper con qué dataset de entrenamiento valida sus números.
- Fallos declarados por los autores: manchas de yodo, pelo y sensibilidad a los umbrales 0,35 y 0,65. No hay análisis por tono de piel.
