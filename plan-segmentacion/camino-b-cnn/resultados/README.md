# Resultados del camino B (segmentacion sin mascaras ni etiquetas)

Cada corrida vive en una carpeta con nombre `<fecha>-<nombre>` y contiene solo
artefactos livianos: `resultados.md`, los `summary.json` y `per_image.csv` de
cada conjunto de evaluacion, los overlays PNG, la hoja de contacto del campo y
los `*_stats.json` de la cache y de las pseudo-mascaras. Nunca la cache, los
checkpoints ni `field.npy`.

## Protocolo comun a todas las filas

- Entrenamiento sin mascaras ni etiquetas sobre el pool unico de 58 457
  imagenes (HAM10000 10 015, ISIC 2019 sin las de HAM 15 316, ISIC 2020 33 126;
  deduplicado por id, 2 % reservado como pseudo-validacion).
- Las mascaras de referencia entran solo en la etapa `evaluate`: ISIC 2018
  Task 1 (2 594) y HAM10000 (10 015, mascaras de Tschandl). Umbral fijo 0,5,
  metricas por imagen a 224 x 224, Dice medio y mediana, IoU, sensibilidad,
  especificidad, Hausdorff-95 cuando esta activo, cobertura predicha y de
  referencia, percentiles y peores ids.
- ISIC 2018 se reporta en dos filas: 706 ids que estan dentro del pool de
  entrenamiento por la deduplicacion (transductivo) y 1 888 fuera del pool.
- Estratificacion por fototipo (ITA sobre la pseudo-mascara, cortes de Groh)
  solo sobre HAM10000, con `min_n` 30 por bucket.
- Pisos no supervisados con el mismo evaluador: SegRNorm y SegXNorm
  (`scripts/run_segnorm_baseline.py`, ROI circular). No hay techo supervisado
  por decision del proyecto.
- Toda cifra cita el commit, el preset y el run de Weights & Biases del que
  sale. Proyecto: `nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion`.

## Corridas

| Corrida | Commit | Preset | Grupo W&B | Estado |
|---|---|---|---|---|
| piloto | `f19ab56` | `cluster-piloto.json` | `piloto` | terminada 2026-09-21, 37,4 min en 2x RTX PRO 4000 |
| final | `1c784bb` | `cluster-final.json` | `final` | pendiente |

## Piloto (detector de campo v1, sin supresion de artefactos)

| Conjunto | n | Dice medio | Dice mediana | Dice p10 | IoU | Sens. | Espec. |
|---|---|---|---|---|---|---|---|
| ISIC 2018 | 2 594 | 0,652 | 0,726 | 0,286 | 0,528 | 0,878 | 0,900 |
| HAM10000 | 10 015 | 0,759 | 0,810 | 0,509 | 0,638 | 0,896 | 0,911 |

HAM10000 por fototipo: brecha de Dice 0,071; peor bucket 6 (0,730, n = 3 256)
frente al bucket 1 (0,781, n = 5 392). HD95 sobre ISIC 2018: mediana 22,9 px,
p90 91,9 px.

Lectura del piloto: el Dice crece con el tamano de la lesion (correlacion +0,52
en ISIC 2018); las lesiones bajo el 10 % del cuadro rinden 0,46 y las grandes
0,80-0,83. Los peores casos (Dice 0) son imagenes con tinta quirurgica violeta,
burbujas de gel, reglas y brillos: la saliencia elige las marcas y no la lesion.
La hoja de contacto mostro ademas falsos campos circulares en imagenes de cuadro
completo con vineteo optico suave (disco inscrito, 72 % del area).

Cambios derivados para la corrida final, todos sin etiquetas: detector de campo
v2 (esquinas por brillo relativo o croma de piel, tres de cuatro), supresion de
tinta y brillos como pixeles ignorados en la pseudo-mascara, y area minima de
candidato 0,003 en lugar de 0,01.

## Final

Pendiente de ejecucion. La tabla se genera con
`plan-segmentacion/comun/evaluacion/build_results_table.py` y se copia aqui
junto con los pisos.
