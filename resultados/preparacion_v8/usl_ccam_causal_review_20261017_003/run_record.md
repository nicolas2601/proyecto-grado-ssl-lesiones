# CCAM 003: revisión causal acotada, sin actualización de pesos

## Identidad y alcance

Familia: auditoría modular v8; método USL-Net; etapa CCAM; corrida original
`usl_ccam-20261017-003`. El diagnóstico terminó a
`2026-10-18T03:34:51.589450+00:00` (17 de octubre en Colombia). Es un análisis
completado de ocho originales del monitor de aprendizaje, no una corrida nueva
de entrenamiento ni una evaluación anatómica. Se hicieron cero actualizaciones
de optimizador y no se consultaron las 2.594 máscaras expertas reservadas.

La rama de trabajo es `experiment/modular-audit-v8`. El operador verificó 57
archivos fijados del runtime antes del diagnóstico; el HEAD Git del checkout
remoto no sustituye esos pines del código desplegado. El commit documental de
este registro se obtiene del historial Git; el diagnóstico no registra un commit
nuevo del código científico ni modifica sus fuentes.

Se conservan configuración, banco y manifiesto de la corrida original. El
manifiesto es
`/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/learning.manifest.json`,
SHA-256 `7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff`.
El banco tiene SHA-256
`306136b3abd29ae32ac8be2a1456cf411c3f1422f93b39f05d46636141cce6f6`.
La identidad, configuración, entorno y tracking completos del entrenamiento se
conservan en el [registro del cierre](../usl_ccam_completion_20261017_003/run_record.md).
El diagnóstico no creó otra corrida W&B. El comando exacto, RSS y código de
salida del helper no están registrados en estos tres snapshots; no se inventan.

## Evidencia local y fuentes remotas

Los snapshots locales son copias observadas de los JSON emitidos en UNABIA;
sus hashes de fuente remota corresponden a los bytes remotos, no al formato
indentado local. Script de diagnóstico SHA-256:
`e202962f56ffc13d3de5e26fe135fdd65c37d2c53c3f402fd1e076b40baeb008`.

| Copia local | Archivo completo en UNABIA | SHA-256 remoto |
|---|---|---|
| [probe.snapshot.json](probe.snapshot.json) | `/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/causal-probe-run003-001.log` | `909afd7bd614bd4b1605e90af7bfb569942ef42147eaa78a147bde6bfa1754a3` |
| [selected-head.snapshot.json](selected-head.snapshot.json) | `/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/causal-probe-run003-001.head1.log` | `ed185f20a4c1106c23e34d903ab6eb87b351d7d497ff67a5dfde0b33121b7f44` |
| [final-head.snapshot.json](final-head.snapshot.json) | `/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/causal-probe-run003-001.head2.log` | `2718fc1a93373ea07de24e32d602fc2c79a6352e55e41892d188e0100f73b675` |

Los checkpoints conservan sus hashes después de la inspección:
`selected.ccam.pt`, época 6:
`83bcf068ef91e6b989a5b71b6ac4231368e91a842d6874397c4555f6619edfec`;
`ccam.pt`, época 30:
`443e4c170451f68811d5dab533f57b511754acb446045200d2e6666c38639802`.
El objetivo inspeccionado tiene SHA-256
`3250cf52d60e7c3ebaab437c764148c8a88843f497f2df624e9430b4792c06a7`.

## 1. Mecanismo próximo confirmado

Los RGB, features de los tres encoders y fusión son finitos y diferentes entre
las ocho imágenes (`rgb_pairwise`, `encoder_statistics`, `fusion_pairwise`).
La pérdida de diversidad espacial útil aparece a la salida de la cabeza:
en el candidato de época 6, `records[*].background_exact_one_fraction` es
0,9948979616 en los ocho casos. El complemento foreground queda exactamente
en cero en 195/196 posiciones; la única posición superviviente es `[0,13]`
en todos los casos. El checkpoint final tiene seis mapas nulos y otros dos con
esa misma esquina superviviente. Esto explica el componente pequeño y las
abstenciones del análisis anterior, sin atribuir identidad clínica a las imágenes.

En `loss_observation_eval_mode`, con las primeras dos imágenes, la norma del
gradiente de pesos del candidato es aproximadamente `6,1043e-17`. En la cabeza
final hay entradas no nulas de hasta `3,6854e-27`; la norma L2 registrada en
float32 es cero por subdesbordamiento. No significa que todas las entradas del
gradiente sean exactamente cero. La saturación y los gradientes diminutos son
hechos observados; no reconstruyen cuándo comenzó el colapso durante la corrida.

## 2. Observaciones que pueden contribuir

- Hay escalas diferentes entre bloques de features. La contribución media al
  logit de SimCLR etapa 3 representa aproximadamente 89,62% del total en el
  candidato y 95,69% en el final. Es dominancia observada, no prueba de un
  checkpoint corrupto ni de culpa causal del espejo SimCLR.
- La norma de pesos de cabeza es 21,8226 en el candidato y 31,9330 en el final.
  Una cabeza fresca con semilla 42 tiene norma 0,58216, pero no es una captura
  del estado inicial exacto de esta corrida: no se informa una tasa exacta de
  crecimiento respecto de ella.
- Con batch 2 solo hay un par positivo no diagonal por ancla; ordenar esos
  positivos no aporta selección relativa entre varios positivos. Las diferencias
  del objetivo local respecto de la referencia CCAM en epsilon, diagonal y
  reducción requieren alineación explícita antes de considerar una reproducción.
- Comparar pérdidas/gradientes en evaluación con batch 2 y 8 es observacional.
  En los dos casos comparables, la fusión cambia hasta 0,98233 y 0,66432; la
  saliencia cambia cero. No se afirma invariancia exacta de features por batch,
  ni que batch 8 haya reparado el problema, ni que batch 2 sea la causa aislada.
- La revisión del wrapper confirma normalización por origen declarada en procedencia;
  no hay evidencia de un error de normalización de RGB que justifique corregirla.

La cabeza 1×1 es consistente con el [diseño USL-Net](https://arxiv.org/html/2309.13289v3#S3.SS3); sustituirla por 3×3 con
BatchNorm no es una corrección evidente. La simetría del objetivo deja una
ambigüedad de orientación foreground/background: el chequeo de orientación del
CCAM oficial debe distinguirse del complemento fijo descrito para USL-Net.

Referencias de código contrastadas por el operador:
[loss.py](https://raw.githubusercontent.com/CVI-SZU/CCAM/master/CUSTOM/core/loss.py),
[train_CCAM.py](https://raw.githubusercontent.com/CVI-SZU/CCAM/master/CUSTOM/train_CCAM.py),
[utils.py](https://raw.githubusercontent.com/CVI-SZU/CCAM/master/CUSTOM/utils.py)
y [model.py](https://raw.githubusercontent.com/CVI-SZU/CCAM/master/CUSTOM/core/model.py).
Es la rama `master` consultada actualmente, sin commit fijado: no reconstruye
por sí sola una versión histórica del paper. La correspondencia exacta entre
esa referencia y la variante USL-Net debe
documentarse antes de cambiar el objetivo.

## 3. Causa inicial y siguiente trabajo

El disparador inicial no está aislado. Los ocho monitores no son una muestra
representativa de todo el pool; este diagnóstico no mide calidad anatómica,
identidad de duplicados ni independencia clínica. Duró 1,575702963 s y registró
640,9771 MiB CUDA asignados y 862 MiB reservados como picos propios del helper;
no son mediciones de la corrida de 30 épocas.

Sigue preparar una corrección revisable: resolver primero la correspondencia del
objetivo con la referencia elegida y su ranking con el batch efectivo. Después,
proponer una única variante de control de escala de features/logits justificada,
preservando la cabeza 1×1. Cambiar simultáneamente LR, batch y arquitectura
impediría atribuir una eventual mejora. Añadir un monitor de degeneración sin
máscaras expertas protege la promoción, pero no corrige el entrenamiento.
No se implementó ninguna de estas alternativas, no se bajó `min_area=500`,
no se eligieron otras épocas y no se relanzó entrenamiento.

La generación completa de pseudo-máscaras y el estudiante siguen bloqueados por
las saliencias. `readiness.review.json` y todos los gates permanecen intactos.
