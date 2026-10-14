> Continúa [53](53-simclr-despliegue-parche-cluster-v8.md).

# Banco de pesos USL-Net cerrado y verificado end-to-end

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
2. Se descargó el checkpoint HuggingFace de SimCLR
   (`lightly-ai/simclrv1-imagenet1k-resnet50-1x`, 102.542.786 bytes) directo al
   clúster — tamaño y SHA-256 coinciden exactos con la validación local previa
   (doc 52).
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

## Qué NO se hizo

No se inició entrenamiento de USL-Net ni de CCAM. No se generaron pseudomáscaras. El
banco de pesos existe y carga correctamente, pero `execution_ready: false` se
mantiene explícito en el recibo — cargar los pesos no autoriza correr nada.

## Evidencia

- Clúster: `~/work/public_ssl_v8_preparation/source-bank.usl-net.reviewed-candidate.json`,
  `~/work/public_ssl_v8_preparation/usl-net.bank-verification.json`,
  `~/work/public_ssl_v8_preparation/staging-03/simclr_r50_1x/converted/encoder.pt(.json)`.
- Repo: este documento + actualización de
  `resultados/preparacion_v8/readiness_20261014/readiness.review.json`.

## Qué sigue pendiente

La equivalencia numérica con el checkpoint TensorFlow oficial de Google sigue sin
verificar (mismo bloqueo que doc 53: acceso anónimo al bucket GCS bloqueado por IAM).
Este banco usa el candidato HF, no el original TF — eso queda explícito en
`upstream_numerical_equivalence_verified: false` dentro de la provenance de SimCLR.
