# Resultados del camino B (segmentacion sin mascaras ni etiquetas)

Cada corrida vive en una carpeta `<fecha>-<nombre>` con artefactos livianos:
`resultados.md`, los `summary.json` y `per_image.csv` de cada conjunto de
evaluacion, los overlays PNG (peores, mejores y aleatorios), la hoja de
contacto del campo, los `*_stats.json` de la cache y de las pseudo-mascaras,
un resumen de conteos del manifiesto FST, el log de la corrida y la bitacora.
Nunca la cache, los checkpoints, `field.npy` ni el manifiesto FST completo
(115 MB), que quedan en el NFS del cluster (`~/work/proyectonp/salidas/`).

## Protocolo comun a todas las filas

- Entrenamiento sin mascaras ni etiquetas sobre el pool unico de 58 457
  imagenes (HAM10000 10 015, ISIC 2019 sin las de HAM 15 316, ISIC 2020 33 126;
  deduplicado por id, 2 % reservado como pseudo-validacion).
- Las mascaras de referencia entran solo en la etapa `evaluate`: ISIC 2018
  Task 1 (2 594) y HAM10000 (10 015, mascaras de Tschandl). Umbral fijo 0,5,
  metricas por imagen a 224 x 224: Dice medio con intervalo bootstrap del 95 %
  (1 000 remuestreos, semilla fija), mediana, percentiles, IoU, sensibilidad,
  especificidad, Hausdorff-95 en la corrida final, cobertura predicha y de
  referencia, peores ids.
- ISIC 2018 se reporta ademas en dos filas: ids dentro del pool de
  entrenamiento por la deduplicacion (transductivo, 722) y fuera del pool
  (1 872).
- Estratificacion por fototipo (ITA sobre la pseudo-mascara y el campo, cortes
  de Groh) con `min_n` 30 por bucket. Solo HAM10000 tiene buckets poblados.
- Pisos no supervisados con el mismo evaluador: SegRNorm y SegXNorm
  (`scripts/run_segnorm_baseline.py`, ROI circular, `--max-side 1024`). No hay
  techo supervisado por decision del proyecto.
- Toda cifra cita el commit, el preset y el grupo de Weights & Biases del que
  sale. Proyecto: `nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion`.
- Hardware: dos NVIDIA RTX PRO 4000 de 24 GB en el pod UNABIA, DDP con
  `torchrun`, bf16.

## Corridas

| Corrida | Commit | Preset | Grupo W&B | Duracion | Estado |
|---|---|---|---|---|---|
| piloto | `f19ab56` | `cluster-piloto.json` | `piloto` | 37,4 min | terminada 2026-09-21 |
| final | `d81a492` | `cluster-final.json` | `final` | 2 h 35 min | terminada 2026-09-21 |

El piloto uso el detector de campo v1, sin supresion de artefactos y con area
minima de candidato 0,01. El final uso el detector v2, supresion de tinta y
brillos, area minima 0,003, MoCo 30 epocas y tres rondas de re-etiquetado de
20 epocas cada una. El resto de hiperparametros es identico.

## Tabla principal

| Corrida | Conjunto | n | Dice medio [IC 95 %] | Dice mediana | Dice p10 | IoU | Sens. | Espec. | HD95 medio | HD95 mediana |
|---|---|---|---|---|---|---|---|---|---|---|
| final | ISIC 2018 | 2 594 | 0,718 [0,709, 0,726] | 0,792 | 0,402 | 0,602 | 0,783 | 0,973 | 26,1 px | 13,6 px |
| final | HAM10000 | 10 015 | 0,785 [0,781, 0,788] | 0,852 | 0,507 | 0,678 | 0,777 | 0,982 | 22,4 px | 12,6 px |
| piloto | ISIC 2018 | 2 594 | 0,652 [0,643, 0,661] | 0,726 | 0,286 | 0,528 | 0,878 | 0,900 | 38,2 px | 22,9 px |
| piloto | HAM10000 | 10 015 | 0,759 [0,755, 0,762] | 0,810 | 0,509 | 0,638 | 0,896 | 0,911 | - | - |

Los intervalos del piloto se calcularon a posteriori desde `per_image.csv`
con el mismo bootstrap; el HD95 del piloto sobre ISIC 2018 viene de una
evaluacion aparte con `--hausdorff`. Los pisos SegRNorm y SegXNorm se agregan a
`2026-09-21-final/resultados.md` cuando termine su evaluacion.

## Dentro y fuera del pool de entrenamiento

| Corrida | Conjunto | En el pool (n) | Dice | Fuera del pool (n) | Dice |
|---|---|---|---|---|---|
| final | ISIC 2018 | 722 | 0,801 | 1 872 | 0,686 |
| final | HAM10000 | 9 822 | 0,786 | 193 | 0,703 |

La cifra honesta de generalizacion en ISIC 2018 es la de fuera del pool:
0,686. La diferencia con la fila transductiva es la ganancia de haber visto la
imagen, sin su mascara, durante el entrenamiento.

## Fototipo (HAM10000)

| Bucket | n | Dice piloto | Dice final |
|---|---|---|---|
| 1 | 5 729 | 0,781 | 0,791 |
| 2 | 473 | 0,794 | 0,791 |
| 3 | 121 | 0,758 | 0,765 |
| 4 | 42 | 0,770 | 0,780 |
| 6 | 3 431 | 0,730 | 0,780 |
| brecha maximo - minimo | | 0,071 | 0,027 |

El bucket 5 (n = 26) queda bajo `min_n` y no se reporta. Los buckets salen del
estimador ITA sobre la pseudo-mascara, que la validacion externa mostro por
debajo de la linea base trivial en Fitzpatrick17k; leer la brecha como un
indicio, no como una medida de equidad validada.

## Tamano de la lesion

Dice medio por banda de cobertura de referencia:

| Cobertura de la lesion | ISIC 2018 piloto | ISIC 2018 final | HAM piloto | HAM final |
|---|---|---|---|---|
| menos del 10 % | 0,464 | 0,646 | 0,556 | 0,750 |
| 10 % a 25 % | 0,749 | 0,793 | 0,786 | 0,850 |
| 25 % a 40 % | 0,811 | 0,782 | 0,834 | 0,794 |
| 40 % a 60 % | 0,806 | 0,733 | 0,825 | 0,724 |
| 60 % o mas | 0,801 | 0,684 | 0,802 | 0,642 |

El area minima de candidato 0,003 resolvio las lesiones pequenas (41 % de
ISIC 2018) y la correlacion Dice-tamano paso de +0,52 a +0,11 en ISIC 2018 y
de +0,40 a -0,20 en HAM10000. El cuello de botella se movio a las lesiones
grandes: el cap de area de candidato (0,6 del campo) excluye del entrenamiento
las lesiones que llenan el cuadro, y el estudiante las subsegmenta (cobertura
predicha 0,17 frente a 0,21 de referencia en ISIC 2018). Es el cambio con mas
retorno para una corrida siguiente: subir el cap a 0,75 manteniendo el resto.

## Diagnostico de la corrida final

- Campo del dermatoscopio (detector v2): 7 528 imagenes circulares y 50 929 de
  cuadro completo, cero fallbacks. Por dataset: ISIC 2020 paso de 1 975 a 129
  circulares (los falsos circulares del piloto eran viñeteo optico de camara),
  HAM10000 de 1 322 a 687, ISIC 2019 de 7 465 a 6 712 (viñetas reales de
  BCN20000). Evidencia visual en `field_contact_sheet.png` de cada corrida.
- Pseudo-mascaras por ronda: excluidas 8,4 % (r0), 4,4 % (r1), 4,3 % (r2);
  fraccion de primer plano 0,131, 0,148, 0,143. El re-etiquetado converge en
  dos rondas y no deriva hacia el campo entero. Tinta y brillos ocupan el
  0,68 % de los pixeles y quedan ignorados en la perdida.
- Modo de fallo restante (`overlays/worst.png`): marcas de tinta violeta,
  cabello denso, gel y reglas siguen atrayendo predicciones aisladas cuando la
  lesion es palida; el estudiante no aprende a rechazarlas porque esos pixeles
  se ignoran en vez de marcarse como fondo.
- Reproducibilidad: 16 etapas con codigo 0, 9 283 s de computo, sin
  reinicios del pod durante la corrida buena. Dos corridas anteriores del
  mismo commit murieron con el pod al convivir con doce procesos de pisos a
  resolucion nativa; la bitacora documenta ambas.
