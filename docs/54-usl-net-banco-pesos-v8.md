> Continúa [53](53-simclr-despliegue-parche-cluster-v8.md).

# Banco USL-Net y forward sintético verificados en UNABIA

2026-10-14, UTC. Continúa `eed86c892ef1c0516e21a024a66f0b8788aedd76`.

## Qué pedía el cierre

`src/dermssl/segmentation_methods/usl_net/model.py` exige exactamente tres orígenes
contrastivos distintos: `{'simclr', 'mocov1', 'mocov2'}`
(`if set(encoders) != {'simclr','mocov1','mocov2'}: raise ValueError(...)`). MoCo v1 y
MoCo v2 ya estaban convertidos y verificados (`converted_strict`) desde sesiones
anteriores. SimCLR era el único origen faltante — el mismo bloqueo del doc 53.

## Qué se hizo

1. Se completó el árbol de paquete `dermssl` en el runtime del clúster
   (`runtime_simclr_same_eed86c8/src/dermssl/segmentation_methods/common/`):
   faltaban `checkpoints.py`, `dino.py` y `data.py` (el parche desplegado en doc 53
   solo traía `networks.py` y `bank.py`). Se transfirieron por el mismo método de
   base64 fragmentado, verificados hash por hash contra el repo local. Un intento de
   `git clone` del repo público para resolver esto más simple fue rechazado por un
   credential helper del clúster que pide autenticación incluso para HTTPS anónimo;
   se abandonó esa vía.
2. Se transfirió a UNABIA el checkpoint Hugging Face de SimCLR
   (`lightly-ai/simclrv1-imagenet1k-resnet50-1x`, revisión
   `476674d2c9804bc2cd4abd8bde6121b6d6b19990`) en
   `/home/jovyan/work/public_ssl_v8_preparation/staging-03/simclr_r50_1x/resnet50-1x.pth`.
   El archivo tiene 102.542.786 bytes y SHA-256
   `fdd96bbcc92008232f31e0f7de46d30810e7115a05005d7cff926e78e6fa37ea`.
3. Se corrió `scripts/prepare_public_ssl_weights_v8.py --origin simclr --format
   pytorch` de verdad en el clúster. El `checkpoint_sha256` resultante
   (`cff4a8bca6cf17d73c79be2aa727cb58dea3f64b5f1f1d727d11e05caf8aed4a`) coincide
   byte a byte con el `conversion_encoder_sha256` ya registrado en
   `readiness.review.json` de la sesión de validación local anterior — confirma que
   la conversión es determinística y reproducible entre entornos. 318 tensores
   mapeados, `fc.weight`/`fc.bias` descartados correctamente, `stem_pooling:
   tensorflow_same`.
4. Se construyó `source-bank.usl-net.reviewed-candidate.json` (convención idéntica a
   `source-bank.rpi-net.reviewed-candidate.json`) referenciando los 3 encoders.
5. **Verificación end-to-end real**: `source_bank()` de `common/bank.py` cargó los 3
   encoders con `strict=True`, sin fallback, cada uno con 23.508.032 parámetros
   (ResNet50 encoder consistente). Esto no es solo una revisión de archivos — es la
   misma función que usará `usl_net/seeds.py` en producción, ejecutada de verdad.

6. Se completó en el runtime `runtime_simclr_same_eed86c8/src/dermssl/` la
   instalación de cinco módulos requeridos para el banco/modelo: `w3_components.py`,
   `w3_usl_models.py`, `segmentation_methods/common/losses.py`,
   `segmentation_methods/usl_net/model.py` y
   `segmentation_methods/usl_net/__init__.py`. Los hashes coinciden con la revisión
   de repositorio `eed86c892ef1c0516e21a024a66f0b8788aedd76`.

7. **Forward sintético del banco en UNABIA:** se usó el banco revisado
   `/home/jovyan/work/public_ssl_v8_preparation/source-bank.usl-net.reviewed-candidate.json`,
   CPU, `eval`, `torch.inference_mode`, semilla 42 y una entrada sintética
   `torch.randn` float32 `[1,3,224,224]`. La salida fue finita `[1,9216,14,14]`;
   `real_images_read=0` y `training_started=false`. SHA-256 del banco:
   `306136b3abd29ae32ac8be2a1456cf4113cf4122f93b9f05d4663641cce6f6`.

## Qué NO se hizo

No se inició entrenamiento de USL-Net ni de CCAM. No se generaron pseudomáscaras. El
banco de pesos existe y carga correctamente, pero `execution_ready: false` se
mantiene explícito en el recibo — cargar los pesos no autoriza correr nada.

## Evidencia

- Clúster: `~/work/public_ssl_v8_preparation/source-bank.usl-net.reviewed-candidate.json`,
  `~/work/public_ssl_v8_preparation/usl-net.bank-verification.json`,
  `/home/jovyan/work/public_ssl_v8_preparation/staging-03/simclr_r50_1x/converted/encoder.pt(.json)`
  y recibo de forward `/home/jovyan/work/public_ssl_v8_preparation/usl-net-synthetic-forward-20261014.json`
  (SHA-256 `b392e7d32eba94fe663777a4af754f4f554df50c715baaa0807ae535b7f6a83a`).
- Repo: este documento + actualización de
  `resultados/preparacion_v8/readiness_20261014/readiness.review.json`.
- W&B autenticado en el clúster: proyecto `dermssl-segmentacion`, entity
  `nmoreno534-universidad-autonoma-de-bucaramanga`; run nuevo
  [`usl_net_real_bank_synthetic_forward_20261014` (`g8us8pz9`)](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/g8us8pz9).
  El artefacto `usl-net-synthetic-forward-20261014` (tipo `validation-receipt`)
  contiene `receipt.json` con el mismo SHA-256 del recibo. No se incluye ningún
  secreto/API key ni se afirma que el archivo crudo del recibo esté versionado en Git.

## Qué sigue pendiente

La equivalencia numérica con el checkpoint TensorFlow oficial de Google sigue sin
verificar (mismo bloqueo que doc 53: acceso anónimo al bucket GCS bloqueado por IAM).
Este banco usa el candidato HF, no el original TF — eso queda explícito en
`upstream_numerical_equivalence_verified: false` dentro de la provenance de SimCLR.

## Estado de preparación y gates pendientes (2026-10-14)

Este cierre solo cubre transferencia de pesos, consistencia hash del código y
forward de forma con entrada sintética. No marca el banco como habilitación de
corrida: `execution_enabled=false`, `launch_ready=false` y `training_started=false`.
Permanecen pendientes:

- equivalencia numérica contra el TensorFlow original de Google;
- manifiestos congelados, IDs/grupos paciente-lesión y manifiesto independiente de
  evaluación anatómica;
- cuotas efectivas y presupuestos completos por etapa y acumulados de GPU, CPU,
  RAM, VRAM, disco y tiempo;
- configs y manifiestos seriales de ejecución con gates revisados;
- revisión de SLED y disponibilidad/revisión del launcher y configuración CUDA en el
  runtime exacto de entrenamiento.

El recibo vive en el clúster y en W&B; su hash y metadatos se versionan aquí, pero
el archivo crudo no se afirma como agregado al repositorio. El flujo V7 de Nicolás
se documenta como referencia de preparación y cierre en [55](55-flujo-v7-para-corridas-v8-en-unabia.md);
V7 no aporta disponibilidad actual de datos en UNABIA ni una receta de entrenamiento
para segmentación V8.
