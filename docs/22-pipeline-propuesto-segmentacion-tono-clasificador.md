# 22. Pipeline propuesto: segmentacion sin etiquetas, tono de piel y clasificador

Documento de referencia unico para el pipeline propuesto de la tesis: que
hace cada etapa, con que datos, que resultados tiene hoy contra los
baselines, como se reproduce en el pod UNABIA y que queda por mejorar. Los
detalles de cada corrida viven en los paquetes de resultados enlazados; aqui
solo se cita lo que esta versionado.

Rama de trabajo: `feature/proposed-pipeline-tone` (sobre `v6`). Seguimiento:
Weights & Biases, proyecto
[`dermssl-segmentacion`](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion)
(grupos `piloto`, `final`, `v2`, `etapa2`).

## 1. Idea en una pagina

```
pool sin etiquetas (58 457 imagenes: HAM10000, ISIC 2019 sin HAM, ISIC 2020)
   |
   v
[1] segmentacion sin mascaras ni etiquetas (camino B)
    MoCo v2 -> CCAM (saliencia) -> modulo de incertidumbre -> U-Net estudiante
    salida: pseudo-mascara por imagen (lesion / piel / ignorar)
   |
   v
[2] tono de piel sobre la piel (fuera de la mascara, dentro del campo)
    mediana de L* y b* CIELAB -> tercil de luminosidad (claro / medio / oscuro)
    ITA / Fitzpatrick solo donde b* >= 5 (secundario)
   |
   v
[3] pares de tono (CIRCLe): misma imagen con la piel oscurecida 16 L*,
    lesion protegida pixel a pixel por la pseudo-mascara
   |
   v
[4] preentrenamiento SSL (SimSiam sobre DINOv2 ViT-S/14), dos brazos
    plain:    dos vistas de la imagen original
    proposed: vista original + vista con tono desplazado (consistencia)
   |
   v
[5] clasificador congelado en HAM10000 (7 clases, probes lineales y kNN)
    resultados por tercil de tono, contra DINOv2 publico y los bancos v5
```

Restricciones que gobiernan el diseno:

- Ninguna mascara ni etiqueta de diagnostico entra al entrenamiento de [1]
  ni de [4]. El pool de preentrenamiento es HAM10000, ISIC 2019 e ISIC 2020;
  **ISIC 2018 no entra**, y por eso sirve como conjunto de evaluacion externa.
  Las mascaras de referencia (ISIC 2018 Task 1, HAM10000) y las
  etiquetas `dx` de HAM10000 solo se usan para evaluar.
- Todo se ejecuta en el pod UNABIA (dos RTX PRO 4000 de 24 GB) con el
  procedimiento de `plan-segmentacion/RUNBOOK_CLUSTER.md`.
- Toda cifra cita el commit, el preset y el artefacto versionado del que sale.

## 2. Etapa 1: segmentacion sin etiquetas (cerrada)

Codigo: `plan-segmentacion/` (`camino-b-cnn/`, `comun/`). Resultados y
evidencia visual: `plan-segmentacion/camino-b-cnn/resultados/README.md`.
Corrida principal: `final`, commit `d81a492`, preset `cluster-final`.

| Corrida | Conjunto | n | Dice medio [IC 95 %] | Dice mediana | IoU | HD95 mediana |
|---|---|---|---|---|---|---|
| final | ISIC 2018 | 2 594 | 0,718 [0,709, 0,726] | 0,792 | 0,602 | 13,6 px |
| final | HAM10000 | 10 015 | 0,785 [0,781, 0,788] | 0,852 | 0,678 | 12,6 px |
| piloto | ISIC 2018 | 2 594 | 0,652 [0,643, 0,661] | 0,726 | 0,528 | 22,9 px |
| piloto | HAM10000 | 10 015 | 0,759 [0,755, 0,762] | 0,810 | 0,638 | - |

Contra los pisos no supervisados (normalizacion de color y Otsu dentro del
campo, sin aprendizaje), pareado por imagen:

| Diferencia | Dice | HD95 |
|---|---|---|
| ISIC 2018, estudiante menos SegRNorm | +0,002 [-0,007, +0,011], empate | -7,4 px [-8,6, -6,1], gana |
| ISIC 2018, estudiante menos SegXNorm | +0,022 [+0,013, +0,032], gana | -7,0 px [-8,2, -5,7], gana |
| HAM10000, estudiante menos SegRNorm | +0,020 [+0,016, +0,023], gana | -0,2 px [-0,7, +0,3], empate |
| HAM10000, estudiante menos SegXNorm | +0,052 [+0,049, +0,056], gana | -0,5 px [-1,0, 0,0], gana por poco |

Lectura: el metodo aprendido mejora el borde y los fallos catastroficos en
ISIC 2018 y el solapamiento en HAM10000; "supera a los pisos" a secas seria
incorrecto. Una ablacion con tope de area 0,75 (`v2`) pierde de forma
significativa y queda documentada como negativa; los volteos en inferencia
no mueven el Dice.

## 3. Etapa 2: tono de piel y equidad de la segmentacion (cerrada)

Estimador: `src/dermssl/fitzpatrick.py`. Manifiesto:
`plan-segmentacion/comun/evaluacion/fst_pseudo_manifest.py`.

Un defecto encontrado y corregido el 2026-09-21 cambia la lectura de todo
lo anterior: el bucket Fitzpatrick calculado con `ITA = arctan((L* - 50) / b*)`
saturaba en +-90 grados cuando b* tendia a cero, y el signo de b* decidia
entre el bucket 1 y el 6 con la misma luminosidad. En HAM10000 el 40,7 % de
las imagenes tenia |b*| < 5 y el grupo b* <= 0 concentraba 2,26 veces mas
melanoma. La brecha por fototipo publicada antes (0,027) queda retirada.

Correccion: el estimador exige mediana de b* >= 5 (`ita_indeterminate` por
debajo, commit `c272bdb`) y el estrato principal pasa a terciles de la
mediana de L* de la piel sobre el pool (`tone_tercile`, commit `bbd9bab`),
cortes 67,25 y 78,59. "Oscuro" es el tercio inferior de este pool (L* medio
62 en HAM10000), no un fototipo oscuro clinico.

| Conjunto | Estrato | n | Dice medio [IC 95 %] | IoU | HD95 mediana |
|---|---|---|---|---|---|
| HAM10000 | claro | 864 | 0,862 [0,854, 0,870] | 0,774 | 9,2 px |
| HAM10000 | medio | 4 876 | 0,803 [0,798, 0,808] | 0,699 | 12,0 px |
| HAM10000 | oscuro | 4 082 | 0,750 [0,744, 0,756] | 0,636 | 14,3 px |
| HAM10000 | brecha claro - oscuro | | 0,111 | | |
| ISIC 2018 (722 con tono) | claro | 260 | 0,820 [0,802, 0,838] | 0,718 | 11,1 px |
| ISIC 2018 (722 con tono) | medio | 304 | 0,802 [0,783, 0,819] | 0,695 | 13,1 px |
| ISIC 2018 (722 con tono) | oscuro | 158 | 0,765 [0,737, 0,788] | 0,646 | 15,5 px |
| ISIC 2018 (722 con tono) | brecha claro - oscuro | | 0,055 | | |

- En HAM10000 los tres terciles estan completamente separados (ningun par
  de intervalos se solapa) y el borde degrada en la misma direccion: la
  HD95 mediana pasa de 9,2 px en el tercil claro a 14,3 px en el oscuro.
  En ISIC 2018 el tercil medio solapa con los dos extremos.
- La brecha de HAM10000 sobrevive al control por diagnostico: dentro de
  cada una de las siete clases el Dice cae de claro a oscuro (monotono en
  seis). Es una limitacion real del metodo sin etiquetas, atribuible al
  menor contraste lesion/fondo en piel menos luminosa, y es la motivacion
  directa de la etapa 3.
- La de ISIC 2018 no se confirma tras controlar por tamano de lesion y se
  reporta como no confirmada en el conjunto externo.
- Evidencia: `plan-segmentacion/camino-b-cnn/resultados/2026-09-21-final/`
  (`strata_v2/`, `fst_diagnostico/`, `fst_diagnostico_v2/`).

## 4. Etapas 3 a 5: adaptacion de tono, SSL y clasificador (en ejecucion)

Codigo nuevo en esta rama:

| Pieza | Archivo | Que hace |
|---|---|---|
| Entradas de tono | `plan-segmentacion/comun/datos/export_tone_inputs.py` | Materializa a 224 las imagenes con mascara elegible, sus mascaras binarias y `mask_manifest.json` |
| Pares de tono | `scripts/generate_tone_transforms.py` (existente) | Resta 16 L* a la piel con la lesion protegida; manifiesto con hashes y revision |
| Preentrenamiento | `scripts/pretrain.py` (existente) + `--wandb-project` | SimSiam sobre DINOv2 ViT-S/14; modo pareado original/transformada con perdida de consistencia |
| Banco congelado | `scripts/cluster_frozen_bank.py` | Protocolo del banco v5 en el pod, con estratos por columna del manifiesto de tono |

Comandos exactos y orden: `plan-segmentacion/RUNBOOK_CLUSTER.md`, seccion 12.

Diseno de la comparacion (mismos datos en los tres brazos del banco, misma
particion por `lesion_id`, semilla 42):

| Backbone | Origen | Papel |
|---|---|---|
| `dinov2_public` | DINOv2 ViT-S/14 LVD-142M, sin adaptar | referencia; debe reproducir el macro-F1 0,547 del banco v5 |
| `ssl_plain` | SimSiam continuo sobre el pool, dos vistas | aisla el efecto del preentrenamiento en dominio |
| `ssl_proposed` | SimSiam con vista de tono desplazado | el metodo propuesto: invarianza al tono con la lesion protegida |

Resultado esperado a reportar: macro-F1 y AUC por probe y regimen de
etiquetas (100 % y 10 %), y para cada backbone la tabla por tercil de tono
con la brecha claro - oscuro. La hipotesis es que `ssl_proposed` reduce la
brecha sin perder macro-F1 global; si no lo hace, se reporta igual.

Referencia de los bancos previos (Kaggle, backbone congelado, probe
`logreg`, `results/baselines/`; `docs/19`): macro-F1 en HAM10000 0,595
(DINOv2 ViT-B/14), 0,547 (DINOv2 ViT-S/14), 0,546 (EfficientNet-B0), 0,540
(ViT-B/16), 0,528 (ResNet-50), 0,317 (SimSiam ResNet-50 v4). No se puede
afirmar que el SSL le gane al supervisado: la ventaja del supervisado
desaparece a igual tamano, con intervalos solapados.

### Datos que entran al preentrenamiento (2026-09-21, pod UNABIA)

| Paso | Salida | Conteo |
|---|---|---|
| Exportador de entradas de tono | `salidas/tone/inputs/` | 55 034 elegibles de 58 457 (94,1 %); 3 423 en cuarentena, todas por area de mascara fuera del 1-90 % |
| Pares de tono (`--shift-buckets 2`) | `salidas/tone/shift2/` | 55 034 transformados, 0 rechazados |

Cuarentena por dataset: HAM10000 111 + 81, ISIC 2019 1 535, ISIC 2020 1 696.
El pareado queda 1 a 1, asi que los dos brazos ven exactamente el mismo
conjunto de imagenes y la unica diferencia entre ellos es la segunda vista.
Revision del manifiesto de tono (la que exige el modo pareado):
`a754961bfd97bee21b4e48edd35b71314ed82b50cc6355480252b4d075e60e51`.

La cuarentena de la exportacion (3 423) y las imagenes sin tercil de tono
del manifiesto FST (3 479) son dos filtros distintos sobre el mismo pool:
el primero pide una mascara de area valida, el segundo ademas pide piel
suficiente para medir la mediana. La diferencia de 56 imagenes es esa
condicion extra, no una inconsistencia.

### Resultados de la primera corrida completa (2026-09-22)

Banco congelado en HAM10000, 7 clases, particion `StratifiedGroupKFold` por
`lesion_id` con semilla 42, primer pliegue, identica en los tres backbones.
Artefactos: `salidas/tone/bank/` en el pod; W&B run `2t73zuvf`.

| Backbone | macro-F1 (logreg) | macro-F1 (logreg_balanced) | AUC (kNN) | Brecha entre terciles |
|---|---|---|---|---|
| `dinov2_public` | 0,583 | 0,558 | 0,875 | 0,091 |
| `ssl_proposed` | 0,562 | 0,522 | 0,796 | 0,102 |
| `ssl_plain` | 0,481 | 0,484 | 0,782 | 0,162 |

**Lo que el experimento demuestra.** El brazo propuesto supera al control en
**cinco de las seis** combinaciones de probe y regimen de etiquetas: +0,081 de
macro-F1 con `logreg` al 100 %, +0,083 al 10 %, +0,039 con `logreg_balanced`
al 100 %, +0,057 al 10 %, y +0,014 de AUC con kNN al 100 %. La excepcion es
kNN al 10 % de etiquetas, donde pierde por 0,010 (0,705 contra 0,715). Y reduce la
brecha entre terciles de tono de 0,162 a 0,102, un 37 % menos. Como los dos
brazos comparten datos, presupuesto, semilla, particion y protocolo, y la
unica diferencia es la segunda vista, esa mejora es atribuible a la
adaptacion de tono.

El resultado se obtuvo ademas con un defecto en contra: hasta el commit
`08727e8` el brazo propuesto sumaba su perdida dos veces, entrenando con el
doble de gradiente que el control. AdamW absorbe la mayor parte de un factor
constante, pero la comparacion no era limpia; la corrida intensiva con el
codigo corregido es la que da la cifra definitiva.

**Lo que el experimento NO demuestra.** Ninguno de los dos brazos alcanza al
backbone publico sin adaptar (0,583). Cuatro horas-GPU sobre 55 034 imagenes
no compiten con LVD-142M: el preentrenamiento en dominio a ese presupuesto
mejora respecto de su propio control, no respecto del estado del arte. Decir
lo contrario seria falso.

**Lo que no se puede afirmar todavia.** La direccion de la brecha de tono en
clasificacion. El estrato peor no es el oscuro sino el claro, en los tres
backbones, pero ese estrato tiene n = 183 contra 820 (oscuro) y 957 (medio),
repartidos en siete clases; el estrato `unavailable`, con n = 44, es el peor
de todos, que es la firma de inestabilidad por tamano de muestra y no de un
efecto real. Restringiendo a `dark` contra `medium`, que tienen n comparable,
el signo se invierte entre backbones (publico -0,052, plain +0,036, proposed
+0,022). Hasta tener intervalos de confianza por estrato, la brecha de
clasificacion se reporta como no concluyente.

Esto NO contradice el resultado de equidad de la seccion 3: aquella es una
medicion de segmentacion, con intervalos sin solapamiento y control por
diagnostico. Son dos mediciones distintas sobre el mismo eje y conviene no
confundirlas.

**Nota de protocolo.** La prueba de sanidad del banco es 0,583 para
`dinov2_public` con `logreg`, no el 0,547 del banco v5: `probes.py` normaliza
con `StandardScaler` antes de la regresion y el banco v5 entrenaba `logreg`
sobre caracteristicas sin normalizar (`docs/19` §4.1). Son protocolos
distintos, asi que este banco no es comparable cabeza a cabeza con v5; la
comparacion valida es la interna entre sus tres backbones.

### Mejoras futuras, en orden de retorno esperado

1. **Mas presupuesto de preentrenamiento.** Cuatro horas-GPU son unas cinco
   vueltas al pool. La curva de SSL no se aplana ahi; es la palanca mas
   directa y la que ataca el unico resultado adverso (no alcanzar al publico).
2. ~~Intervalos de confianza por estrato~~ **hecho** (commit de 2026-09-22):
   `metrics.stratified_by_label` reporta `macro_f1_ci95_low/high` por estrato
   y el banco agrega `gap_established`, que dice si los estratos extremos
   estan realmente separados o si sus intervalos se solapan. Un `null` ahi
   significa "no hay con que juzgarlo", que NO es lo mismo que "no hay
   brecha".
3. **Desplazamiento de tono variable por imagen.** Hoy es fijo en dos niveles
   hacia el oscuro; Corbin y Marques 2023 usan hasta dos niveles con signo
   aleatorio, que expone al modelo a las dos direcciones.
4. **Un objetivo con dos terminos de verdad.** Hoy el brazo propuesto usa una
   sola perdida: SimSiam entre la original y su copia con tono desplazado. Eso
   ensena invarianza al tono, pero mezcla en el mismo termino la invarianza a
   la augmentacion. Un objetivo con tres vistas — dos augmentaciones de la
   original mas la transformada — separaria las dos senales y recien ahi
   `--consistency-weight` seria una palanca real. Con el diseno actual ese
   flag apenas mueve nada: escala la perdida entera y AdamW es casi invariante
   a la escala, asi que barrerlo no es el experimento barato que parece.
5. **Pseudo-mascaras de la ronda 3.** Las usadas son las de la ronda 2; el
   estudiante final no exporta predicciones sobre el pool. Exportarlas es
   trabajo mecanico y mejora la entrada de todo lo demas.
6. **Resolucion.** Todo se materializa a 224 para no transformar ISIC 2020 en
   resolucion nativa. Subirla eleva el techo, a costa de espacio y computo.
7. **Validacion externa en fototipos altos.** El pool es mayoritariamente de
   piel clara; "oscuro" es relativo a el. Sin datos externos no se puede
   afirmar nada sobre fototipos altos reales.

### Como se reproduce, en una linea

La etapa completa corre con un solo comando idempotente, pensado para un pod
que se cae: cada etapa declara el artefacto que prueba que termino, las
terminadas se saltan y las interrumpidas se relanzan igual.

```bash
python scripts/run_stage2.py \
    --repo . --outputs $HOME/work/proyectonp/salidas \
    --ham-root /home/jovyan/work/datos/ham10000 \
    --budget-gpu-hours 12 --dry-run     # quitar --dry-run para ejecutar
```

Si el contenedor muere: levantarlo y repetir exactamente esa linea.

## 5. Reproduccion

1. Pod UNABIA, venv `~/work/proyectonp/.venv` (Python 3.13, torch 2.11
   cu128, wandb). `export TORCH_HOME=$HOME/work/.torch`.
2. Etapa 1 y 2: `plan-segmentacion/RUNBOOK_CLUSTER.md`, secciones 5 a 9
   (cache, MoCo, CCAM, pseudo-mascaras, estudiante, `fst`, `evaluate` con
   `--strata-column tone_tercile`).
3. Etapas 3 a 5: seccion 12 del mismo runbook.
4. Tests sin GPU: `python -m pytest tests/test_seg_*.py tests/test_fitzpatrick.py tests/test_cluster_frozen_bank.py tests/test_pretrain_tracking.py -q`.

## 6. Limitaciones y siguientes pasos

- La segmentacion rinde peor en piel menos luminosa (seccion 3). Es el
  problema que la etapa 3 ataca; hasta tener su resultado, cualquier uso
  clinico del pipeline debe declararlo.
- El pool es mayoritariamente piel clara; "oscuro" es relativo a este pool.
  Validar en fototipos altos requiere datos externos (por ejemplo
  Fitzpatrick17k, que hoy solo sirve para la validacion del estimador).
- Las imagenes del preentrenamiento se materializan a 224 desde la cache
  para no transformar ISIC 2020 a resolucion nativa; los dos brazos ven la
  misma fuente, asi que la comparacion es justa, pero el techo absoluto puede
  subir con mas resolucion.
- Las pseudo-mascaras usadas en la etapa 3 son las de la ronda 2 (las que
  alimentaron el manifiesto de tono); el estudiante de la ronda 3 no exporta
  predicciones sobre el pool. Exportarlas es una mejora directa.
- Palancas abiertas: presupuesto de GPU del preentrenamiento, peso de la
  consistencia, desplazamiento de tono variable por imagen (Corbin y
  Marques 2023 usan hasta dos niveles con signo aleatorio), y el brazo
  CIRCLe sobre caracteristicas congeladas (`scripts/kaggle_circle_stage2.py`)
  como tercera comparacion.
