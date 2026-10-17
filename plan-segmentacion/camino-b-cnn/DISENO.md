# Camino B: decisiones de diseño y desviaciones

Fuentes: USL-Net (Li et al., Biomed. Signal Process. Control 2024, arXiv 2309.13289, texto completo leído) y CCAM/C2AM (Xie et al., CVPR 2022, arXiv 2203.13505, leído en ar5iv). El repositorio oficial de CCAM es MIT; solo se consultó su README (batch 128, alpha 0,25, MoCo v2). No se copió código: todo está reescrito.

## Reglas de la tesis que el código hace cumplir

| Regla | Cómo se cumple |
|---|---|
| El entrenamiento nunca lee máscaras ni etiquetas | La cache guarda solo imágenes (`cache_info.json` registra `reads_masks_or_labels: false`). `tests/test_seg_static.py` verifica por AST y texto que ningún módulo de entrenamiento nombra máscaras de prueba, y que solo `cache_io.py` y `evaluate_segmentation.py` abren imágenes |
| Solo la evaluación abre máscaras | `evaluacion/evaluate_segmentation.py`; el driver solo pasa `--mask-dir` a la etapa `evaluate` (el generador sintético del smoke también lo nombra, pero solo escribe) |
| Sin pesos preentrenados | El backbone parte de inicialización aleatoria. `--init-weights` en `train_moco.py` es opcional y **desvía el escenario puro**; queda registrado en `run_config.json` y en `backbone.pt` |
| Checkpoint y umbral sin máscaras | Umbral fijo 0,5 (constante, sin flag). El modelo de segmentación elige el checkpoint por pérdida de pseudo-validación sobre píxeles confiables, o la última época (`--selection last`) |
| No reusar `train_unet` ni `ISIC2018SegmentationDataset` | Solo se importa la clase `UNet` (base 32, ~7,77 M parámetros); una prueba estática lo exige |

## Decisiones de ingeniería (clúster)

- Cache uint8 en memmap `.npy` (224x224 por defecto, ~10,3 GB para 68k imágenes) más `manifest.csv`. Sin DataLoader ni procesos: `/dev/shm` es de 64 MB. Los aumentos corren en GPU por lote con operaciones de torch (recorte afín, volteo, brillo, contraste, saturación, matiz por rotación YIQ, gris, desenfoque gaussiano), con generadores sembrados por (semilla, época, rank).
- bf16 con autocast en CUDA; pérdidas y similitudes en float32.
- DDP con `torchrun` (`--nproc_per_node=2`) y respaldo a un solo proceso; `broadcast_buffers=False`, así que las estadísticas de BatchNorm guardadas son las del rank 0.
- Reanudación por época desde `last.pt` (escritura atómica). El sampler es determinista por (semilla, época), por lo que reanudar en límite de época reproduce el orden.
- Partición de pseudo-validación: ~2 % del pool por hash con semilla, independiente del orden; se deduplica por id entre datasets (ISIC 2019 contiene HAM10000; gana la primera raíz). `pool_manifest.py` y `data_resolver.py` no encajan: indexan por stem colapsando raíces, calculan SHA-256 por archivo (una pasada extra sobre 68k imágenes) y su split es 70/15/15 para la clasificación.
- Las etapas MoCo y CCAM usan solo la partición `train`. Las pseudo-máscaras se generan para todas las imágenes, incluida `val`.
- Presupuesto: cada época registra segundos e imágenes por segundo (`metrics.jsonl`, `timing.json`); el final se dimensiona en imágenes procesadas.

## Desviaciones respecto a USL-Net

| # | USL-Net | Aquí | Motivo |
|---|---|---|---|
| 1 | Fusión SimCLR + MoCo v1 + MoCo v2 | Un solo método, MoCo v2 (codificador de momento, cola, MLP, desenfoque) | Simplificación de piloto. `FeatureProvider` y `FusedFeatureProvider` permiten fusionar después sin tocar CCAM |
| 2 | Capas 4 y 5 concatenadas, stride 1 en la última | Capas `layer3` y `layer4` de torchvision concatenadas (3072 canales). Stride 1 solo al extraer para CCAM; el preentrenamiento usa el stride estándar 2 | Equivalencia de nombres (torchvision empieza en `layer1` tras el stem). El stride 1 en preentrenamiento encarece notablemente el cómputo; los pesos son compatibles |
| 3 | Mapa de 4096 posiciones | 14x14 a 224 px (1/16) | La descripción no permite reproducir las 4096 posiciones |
| 4 | Cabeza CCAM 1x1 (texto de USL-Net) | Conv 3x3 + BN + sigmoide (CCAM), `--head-kernel 1` disponible | Se sigue el paper original de CCAM |
| 5 | CCAM reentrena | Backbone congelado, solo se entrena la cabeza | Costo; el piloto mide antes de complicar |
| 6 | Peso de centralidad = suma de distancias de píxeles | Distancia del centroide al centro (normalizada por la semidiagonal) entre regiones con área mínima; desempate por área | La suma de distancias favorece regiones diminutas, contrario a la intención. Área mínima 500/224² como fracción (alpha = 500 px) |
| 7 | "Muy lejos del centro" sin valor | Se excluye la muestra si la distancia normalizada supera 0,5 (`--max-center-distance`) | Sin dato en el paper; valor simple y ajustable |
| 8 | Muestra sin primer plano se excluye | Todos sus píxeles pasan a "ignorar" y se descarta del entrenamiento | Igual intención |
| 9 | Signo del mapa no descrito | Se voltea el mapa si el borde es más brillante que el centro (`orient_saliency`); normalización mín-máx por imagen | Un mapa agnóstico de clase tiene signo arbitrario; regla sin etiquetas |
| 10 | modelo de segmentación ResNet50 + FCN | U-Net de `torch_unet.py` (~7,77 M) | El modelo de segmentación debe ser idéntico en los caminos A y B |
| 11 | Pérdida BCE con ponderación (l-0,5)² | BCE + Dice suave, solo sobre píxeles confiables (255 se ignora) | Mismo efecto de ignorar la banda incierta; Dice según el plan |
| 12 | 5 rondas de 300 épocas | `--rounds` y `--epochs-per-round` separados; piloto 1 ronda de 10 épocas | Presupuesto; con 10 épocas no se reproduce el paper |
| 13 | Rondas siguientes desde la red | Predicciones del modelo de segmentación entran al mismo módulo de incertidumbre, sin normalizar ni orientar; cada ronda reentrena la U-Net desde cero | No especificado; la opción más simple |
| 14 | Evaluación a 224 | Igual: la máscara se redimensiona con vecino más cercano a la resolución de entrenamiento | Coincide con el protocolo |

## Desviaciones respecto a CCAM

- Pérdida `L_POS + L_NEG` como en el paper: vectores de primer plano y fondo por imagen, negativos entre todas las parejas cruzadas del lote, positivos ponderados por `exp(-alpha * rango)` con alpha 0,25. No hay muestreo de píxeles por ranking ni términos extra. Los mapas de características son no negativos (post-ReLU), por lo que las similitudes coseno están en [0, 1]; se recorta con epsilon para evitar `log(0)` en casos degenerados.
- Lote de 64 por GPU en el piloto (el README oficial usa 128 en otro dominio).

## Desviaciones respecto a MoCo v2

- Shuffle BN emulado con `SplitBatchNorm2d` (sub-lotes de estadísticas) y permutación del lote de claves, porque un solo proceso o DDP sin SyncBN filtraría estadísticas entre consulta y clave.
- AdamW con calentamiento y coseno (regla de la tesis: Adam/AdamW), no SGD. Momento 0,99 (el paper usa 0,999; con pocos pasos 0,999 casi no actualiza), cola de 16384 (paper 65536), temperatura 0,2. El preset final usa 0,999.
- Aumentos reimplementados en torch (matiz y desenfoque son aproximaciones de los de torchvision).

## Riesgos y límites conocidos

- **Circularidad de la selección.** La pérdida de pseudo-validación mide acuerdo con las pseudo-etiquetas, no calidad real. Puede favorecer el sobreajuste al ruido de la ronda. Alternativa: `--selection last`. Nunca se usa Dice contra máscaras reales.
- Sin torch en la máquina de desarrollo original, la verificación se hizo en un entorno CPU aparte (ver el informe). En GPU real quedan sin medir: velocidad, memoria (los lotes del preset pueden requerir reducirse), NCCL y bf16.
- Las etapas dependen de `dermssl.torch_unet`, que importa `torchvision` y `PIL` al nivel del módulo.
- Fallos declarados por los autores: manchas de yodo, pelo y sensibilidad a los umbrales 0,35 y 0,65. No hay análisis por tono de piel.
- Con pesos aleatorios y 10 épocas el mapa CCAM puede ser pobre; si el módulo de incertidumbre excluye demasiadas muestras, `pseudo_masks_stats.json` lo muestra (`excluded_fraction`) y el modelo de segmentación falla con un mensaje explícito si no queda ninguna.
