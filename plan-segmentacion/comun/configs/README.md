# Configuraciones

Aquí viven los argumentos de cada corrida, para no tocar código. El driver
`camino-b-cnn/entrenamiento/run_pipeline.py` lee uno de estos archivos con
`--config` y lo traduce a banderas de línea de comandos.

## Presets disponibles

| Preset | Para qué sirve | Rondas | Épocas contrastivas | Épocas de red de segmentación entrenada con pseudo-máscaras por ronda |
|---|---|---|---|---|
| `piloto.json` | Prueba local rápida del pipeline completo | 1 | 10 | 10 |
| `final.json` | Plantilla histórica con marcadores `<...>`; el driver se niega a correrla hasta completarla | — | — | — |
| `cluster-piloto.json` | Piloto en el clúster UNABIA: mide segundos por época e imágenes por segundo | 1 | 5 | 5 |
| `cluster-final.json` | Corrida final en el clúster UNABIA | 3 | 30 | 20 |

Los dos presets `cluster-*` están completos: no tienen marcadores y se corren
sin editar nada. El procedimiento exacto está en
[`../../RUNBOOK_CLUSTER.md`](../../RUNBOOK_CLUSTER.md).

## Secciones del archivo

- `pipeline`: `rounds`, `epochs_per_round` y `seed`. Las dos primeras se pueden
  sobrescribir desde la línea de comandos con `--rounds` y `--epochs-per-round`.
- `cache`: resolución, fracción de pseudo-validación, hilos de decodificación y
  los parámetros del campo del dermatoscopio (`field_black_threshold`,
  `field_margin_fraction`). Con `"no_field": true` se desactiva el campo y el
  driver deja de pasar `--field` a las etapas de pseudo-máscaras.
- `contrastive`, `ccam`, `student`, `evaluate`: hiperparámetros de cada script.
  En `evaluate`, `"hausdorff": true` agrega la distancia de Hausdorff-95 al
  reporte. Solo la trae `cluster-final.json`, porque es la métrica más lenta y
  el piloto mide velocidad, no calidad.
- `pseudo`: umbrales del módulo de incertidumbre, incluido `max_area_fraction`,
  que descarta una región cuando ocupa más de esa fracción del campo válido.
  - `min_area_fraction`: área mínima que debe tener una región para ser
    candidata a lesión, como fracción del campo válido. El valor por defecto
    viene de USL-Net (500 píxeles sobre 224x224, cerca del 1 %) y deja fuera
    toda lesión pequeña, que en ISIC2018 es buena parte del conjunto.
    `cluster-final.json` lo baja a `0.003` para que esas lesiones sí lleguen a
    ser candidatas. Bajarlo más deja pasar ruido pequeño como si fuera lesión.
  - `artifact_suppression`: con `true` (el valor por defecto) el driver le pasa
    a la etapa de pseudo-máscaras el `images.npy` de la caché con `--images`, y
    los píxeles de tinta quirúrgica violeta y de brillo especular dejan de ser
    candidatos y quedan marcados como ignorar. Es una decisión por color, sin
    máscaras ni etiquetas. Con `false` el driver omite `--images` y la etapa
    trabaja solo con la saliencia. No se traduce nunca a una bandera propia:
    es un interruptor que elige si `--images` viaja o no.

Cada clave se convierte en una bandera cambiando los guiones bajos por guiones:
`max_area_fraction` se vuelve `--max-area-fraction`. Un valor `true` se vuelve
una bandera sin argumento y un valor `false` o `null` se omite.

## Hardware de referencia

Clúster UNABIA: 2x RTX PRO 4000 de 24 GB, Xeon w7-2495X de 48 hilos, 123 GB de
RAM, CUDA 13.0 con driver 580 y torch 2.11 sobre Python 3.13. `/dev/shm` es de
67 MB, por lo que el pipeline no usa procesos de carga de datos.
