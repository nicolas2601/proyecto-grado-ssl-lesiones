# Banco de fine-tuning (etapa 2 del banco de baselines)

Los tests de `notebooks/baselines*` midieron cada backbone **congelado**
(probes kNN / regresion logistica / prototipos). Esta etapa entrena los
mismos 8 modelos **end-to-end** sobre los datos medicos y reporta las mismas
metricas, para responder: cuanto cierra (o no) el fine-tuning la brecha entre
modelos supervisados y SSL?

## Estructura

```
notebooks/finetune/<modelo>/finetune_<modelo>.ipynb               HAM10000 (7 clases)
notebooks/finetune_isic2020/<modelo>/finetune_<modelo>_isic2020.ipynb  ISIC 2020 (binaria)
```

Generados por `scripts/build_finetune_tests.py`, que **importa** las celdas
compartidas de `scripts/build_baseline_tests.py` (datos, split anti-fuga,
loaders de backbone, metricas, reporte, MLflow). Un solo protocolo, cero
divergencia entre bancos. No editar notebooks a mano: editar el builder y
regenerar.

## Fuente de pesos

Igual que el banco congelado: **Hugging Face Hub via timm** (decision del
equipo con la Dra. Karen, 2026-08-28), con las mismas dos excepciones
documentadas (alexnet: torchvision; simsiam_r50: checkpoint oficial FAIR).
Congelado y fine-tuned parten de los MISMOS pesos -> la comparacion por
modelo es limpia.

## Protocolo (identico para los 8 modelos)

| Item | Valor |
|---|---|
| Arquitectura | backbone completo descongelado + cabeza lineal nueva |
| Optimizador | AdamW — LR backbone 5e-5, LR cabeza 5e-4, wd 1e-4 |
| Schedule | coseno sobre `EPOCHS * len(dl_tr)` pasos |
| Perdida | cross-entropy con pesos por clase (desbalance HAM) |
| Epocas | 8 (SMOKE: 1) |
| Batch | 32 (OOM en vgg19 / vit_b16 / dinov2_vitb14 -> bajar a 16) |
| Aumentos (train) | RandomCrop, flips H/V, ColorJitter suave |
| Precision | AMP fp16 en GPU |
| Seleccion | mejor epoca por balanced accuracy de validacion; la evaluacion final usa ESE checkpoint |
| Split | el MISMO split anti-fuga del banco congelado (misma semilla 42) |
| Salidas | `results_<modelo>_ft*.json`, `ft_<modelo>*.pth`, run MLflow en experimento `finetune-<dataset>` |

## Como correr (manual, por notebook)

1. Subir el `.ipynb` a Kaggle (New Notebook -> File -> Import).
2. Add Input segun el dataset (mismo input que el banco congelado):
   - HAM10000: `Skin Cancer MNIST: HAM10000` (kmader).
   - ISIC 2020: `ISIC 2020 JPG 256x256 RESIZED` (nischaydnk/isic-2020-jpg-256x256-resized).
3. Settings: GPU T4, Internet ON.
4. `SMOKE = True` -> Run All (valida el wiring en minutos).
5. `SMOKE = False` -> Save & Run All (Commit). ~1-2 h por modelo en T4.
6. Descargar del Output: el JSON de resultados, el checkpoint `.pth` y `mlruns/`.

## Automatizacion con Kaggle CLI (opcional, recomendado)

La API de Kaggle permite subir y ejecutar sin tocar el navegador:

```bash
pip install kaggle          # y ~/.kaggle/kaggle.json con un token NUEVO
cd notebooks/finetune/resnet50
cat > kernel-metadata.json <<EOF
{
  "id": "<usuario-kaggle>/finetune-resnet50-ham",
  "title": "finetune-resnet50-ham",
  "code_file": "finetune_resnet50.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["kmader/skin-cancer-mnist-ham10000"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF
kaggle kernels push -p .          # sube Y ejecuta (Save & Run All)
kaggle kernels status <usuario-kaggle>/finetune-resnet50-ham
kaggle kernels output <usuario-kaggle>/finetune-resnet50-ham -p ./resultados
```

Notas:
- Recordar `SMOKE = False` antes del push definitivo.
- Cuota GPU Kaggle: ~30 h/semana por cuenta. 8 modelos x ~1.5 h = ~12 h por
  dataset: un banco completo cabe en la semana de una cuenta; repartir entre
  los del equipo si se corren dos bancos.

## Comparacion final

Con ambos bancos corridos, cada modelo tiene dos JSON comparables campo a
campo: `results_<modelo>.json` (congelado) vs `results_<modelo>_ft.json`
(fine-tuned), mismas metricas, mismo split. La tabla de la tesis sale de ahi.

## Version para el cluster (Dra. Karen)

Decision 2026-09-03: ISIC 2019 queda fuera del banco; solo HAM10000 e ISIC 2020.

El mismo protocolo corre sin notebook en `scripts/finetune_baseline.py`
(fuente unica: `src/dermssl/finetune/protocol.py`, verificada por
`tests/test_finetune_protocol.py`). Lanzador de un solo comando y guia:
`cluster/finetune/README.md`. Resultados en `results/finetune/<dataset>/`.

Para subir y ejecutar los notebooks de Kaggle sin navegador:
`python scripts/kaggle_push.py --bank finetune_isic2020 --user <kaggle-user> --dry-run`
(transporte `--via composio` usa la cuenta Kaggle enlazada en Composio, sin token local).
