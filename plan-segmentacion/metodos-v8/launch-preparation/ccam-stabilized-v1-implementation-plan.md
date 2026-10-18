# CCAM stabilized v1: implementación

## Objetivo

Implementar una variante aislada de CCAM para USL-Net que controle la escala de los seis mapas de características, evite la saturación de logits y optimice con el objetivo contrastivo CCAM explícito. El borrador permanece deshabilitado hasta revisión estática y activación operativa.

## Diseño implementado

1. `ccam_stabilized_v1` normaliza cada mapa de etapa por RMS por imagen sobre canales y espacio, epsilon `1e-6`, antes de concatenar; el orden queda fijado como SimCLR, MoCo v1, MoCo v2, etapas 3 y 4.
2. La cabeza 1x1 existente normaliza sus logits por imagen sobre el plano espacial antes de `sigmoid`, sin parámetros afines, estadísticas móviles ni autocast. El perfil y el SHA del banco se ligan al checkpoint y sidecar. La generación exige el mismo perfil y banco.
3. `ccam_loss.py` implementa la semántica publicada de coseno por pares, diagonal incluida, ranking positivo con auto pares, clamp `[0.0005, 0.9995]` y media sobre `B²`. USL-Net conserva sus descriptores agrupados y cabeza 1x1; no es una reproducción de la arquitectura publicada con convolución 3x3 y BatchNorm.
4. Se usa batch efectivo 64, extracción congelada en micro-batches 2 y una sola actualización de cabeza por batch efectivo. El resto de 32 imágenes se conserva. En 34.784 imágenes son 544 actualizaciones por época. Semilla 42, 30 épocas, AdamW `lr=0.001`, `weight_decay=0.01`.
5. Ocho imágenes fijas del manifiesto de aprendizaje monitorizan componentes de pérdida, logits, norma y número de elementos de gradiente no nulos en float64, saturación exacta, componentes elegibles y Pearson global/individual. El candidato exige Pearson global >=0.98, las ocho máscaras elegibles con área mínima 500, fondo no nulo y tres épocas consecutivas elegibles. Es una heurística técnica, no validación anatómica.
6. Después de cinco épocas, tres monitores globalmente estables (Pearson >=0.98) con abstención en las ocho sondas detienen la corrida como fallo no exitoso; tres épocas con todos los mapas constantes o totalmente saturados también detienen. La regla y sus supuestos están en el config.
7. El gate admite una excepción explícita y limitada a `usl_net/ccam/ccam_stabilized_v1`: no se midieron recursos para esta variante ni se ejecutó smoke según la restricción del usuario. Ambos campos deben permanecer `false`; datos, procedencia, vínculo de configuración/manifiesto/protocolo y demás requisitos no se eximen. No se cambia `readiness.review.json` ni se reutiliza la recuperación de CCAM 003.

## Secuencia por archivos

- `src/dermssl/segmentation_methods/usl_net/model.py`, `src/dermssl/w3_usl_models.py`: perfiles de fusión y logits manteniendo legacy por defecto.
- `src/dermssl/segmentation_methods/usl_net/ccam_loss.py`: objetivo nuevo aislado.
- `src/dermssl/segmentation_methods/usl_net/seeds.py`: ejecución, monitoreo, selección y recuperación versionados por identidad del perfil.
- `src/dermssl/segmentation_methods/common/training.py`: admisión de waiver exacto, solo en CCAM estabilizado.
- `src/dermssl/segmentation_methods/common/checkpoints.py`, `common/inference.py`: comprobación del perfil al cargar la cabeza y reproducir la fusión para generación.
- `plan-segmentacion/metodos-v8/launch-preparation/configs/usl_ccam.stabilized.draft.json`: parámetros y pins del nuevo borrador, deshabilitado.
- `odd/tasks/v8-cierre-bloqueos-lanzamiento.md`: registro de qué quedó implementado y qué sigue pendiente.

## Alcance de verificación

No se ejecutan tests, lint, importación/compilación, GPU, navegador, clúster ni Git por instrucción explícita. La revisión estática previa al despliegue debe comprobar el contrato de gate, hashes del paquete/config/manifiesto/banco/protocolo, la exclusividad de perfil y la creación de recibos de una corrida nueva en `/runs_v8/usl_ccam-20261018-004` y `/validation_v8/usl-ccam-stabilized-launch-20261018-004/`. Los caps `24h`, `64 GiB RSS`, `18 GiB GPU` con `4 GiB` de reserva y `1024 MiB` de salida son límites de planificación heredados, no mediciones de esta variante. No hay promesa de convergencia, selección satisfactoria ni calidad anatómica.
