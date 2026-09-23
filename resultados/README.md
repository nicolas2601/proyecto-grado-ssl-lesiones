# Resultados completos del pipeline propuesto

Todo lo que produjo el proyecto, en un solo lugar: segmentacion sin
etiquetas, tono de piel, los cuatro brazos de preentrenamiento y el
clasificador. Que gano, que perdio, y como mejorarlo.

Cada cifra cita el commit, el preset y el artefacto del que sale. Lo que no
esta medido se dice que no esta medido.

## Indice

1. [Que se entreno y con que](#1-que-se-entreno-y-con-que)
2. [Segmentacion sin mascaras ni etiquetas](#2-segmentacion-sin-mascaras-ni-etiquetas)
3. [Tono de piel y equidad de la segmentacion](#3-tono-de-piel-y-equidad-de-la-segmentacion)
4. [Clasificacion: los seis backbones](#4-clasificacion-los-seis-backbones)
5. [Que gano](#5-que-gano)
6. [Que perdio](#6-que-perdio)
7. [Que no se puede afirmar](#7-que-no-se-puede-afirmar)
8. [Como mejorarlo](#8-como-mejorarlo)
9. [El modelo: descargar y probar](#9-el-modelo-descargar-y-probar)
10. [Reproducir todo](#10-reproducir-todo)

---

## 1. Que se entreno y con que

Pool de **58 457 dermatoscopias** sin etiquetas (HAM10000, ISIC 2019 sin la
parte de HAM, ISIC 2020). De ellas **55 034** (94,1 %) tienen una
pseudo-mascara valida y entran al preentrenamiento; las 3 423 restantes
quedan en cuarentena por area de mascara fuera del rango 1-90 %.

Ninguna mascara de referencia ni etiqueta de diagnostico entra al
entrenamiento. Las mascaras de ISIC 2018 y HAM10000 y las etiquetas `dx`
solo se usan para evaluar.

**ISIC 2018 NO forma parte del pool de entrenamiento.** El preentrenamiento
usa exclusivamente HAM10000, ISIC 2019 e ISIC 2020. ISIC 2018 entra solo
como **conjunto de evaluacion externa** de la segmentacion, y esa es su
razon de ser: HAM10000 esta dentro del pool, asi que medir sobre el es
medir en distribucion. ISIC 2018 es el unico conjunto con mascaras que el
modelo nunca vio, y por eso es el que responde si el metodo generaliza.

| Etapa | Que produce | Estado |
|---|---|---|
| Segmentacion (camino B) | Pseudo-mascara por imagen | Cerrada |
| Tono de piel | Tercil de luminosidad y buckets ITA por imagen | Cerrada |
| Pares de tono | 55 034 pares original / piel oscurecida | Cerrada |
| Preentrenamiento ×4 | Backbones adaptados (ViT-S y ViT-B, con y sin tono) | PENDIENTE |
| Banco congelado | Tabla de clasificacion con 6 backbones | PENDIENTE |

<!-- RESULTADOS: las secciones 4 a 7 se completan con la corrida de
     2026-09-22. Hasta entonces las cifras que aparecen son de la corrida
     preliminar de 4 horas-GPU y estan marcadas como tales. -->

### Lo que consumio cada brazo

<!-- TABLA-ENTRENAMIENTO:INICIO -->

| Brazo | Modo | Presupuesto (h-GPU) | Pasos | Muestras | Muestras/paso |
|---|---|---|---|---|---|
| `ssl_plain_vitb14` | two_view_ssl | 12,0 | 5.894 | 754.432 | 128,0 |
| `ssl_plain_vits14` | two_view_ssl | 12,0 | 6.286 | 804.608 | 128,0 |
| `ssl_proposed_vitb14` | original_transformed_consistency | 12,0 | 5.702 | 729.856 | 128,0 |
| `ssl_proposed_vits14` | original_transformed_consistency | 12,0 | 6.110 | 782.080 | 128,0 |

El presupuesto es en **horas-GPU**, no en epocas ni en muestras. Los
brazos pareados leen la imagen original y su copia con el tono
desplazado, asi que **a igual presupuesto procesan menos muestras que su
propio control**. Esa es la comparacion que declara la tesis; su limite,
en las dos direcciones, esta en la seccion 7.

<!-- TABLA-ENTRENAMIENTO:FIN -->

---

## 2. Segmentacion sin mascaras ni etiquetas

Commit `d81a492`, preset `cluster-final`. MoCo v2 sobre el pool, saliencia
CCAM, modulo de incertidumbre y U-Net estudiante con re-etiquetado en tres
rondas.

| Conjunto | Papel | n | Dice medio [IC 95 %] | Dice mediana | IoU | HD95 mediana |
|---|---|---|---|---|---|---|
| ISIC 2018 | **externo**, fuera del pool | 2 594 | 0,718 [0,709, 0,726] | 0,792 | 0,602 | 13,6 px |
| HAM10000 | en distribucion, dentro del pool | 10 015 | 0,785 [0,781, 0,788] | 0,852 | 0,678 | 12,6 px |

El Dice mas alto en HAM10000 no significa que el metodo funcione mejor ahi
en sentido util: significa que HAM10000 esta dentro del pool sobre el que se
entreno sin etiquetas. La cifra que habla de generalizacion es la de ISIC
2018, que es mas baja y es la honesta.

Contra los pisos sin aprendizaje (normalizacion de color y Otsu dentro del
campo del dermatoscopio), pareado por imagen:

| Diferencia | Dice | HD95 |
|---|---|---|
| ISIC 2018 contra SegRNorm | +0,002 [-0,007, +0,011], empate | -7,4 px, gana |
| ISIC 2018 contra SegXNorm | +0,022 [+0,013, +0,032], gana | -7,0 px, gana |
| HAM10000 contra SegRNorm | +0,020 [+0,016, +0,023], gana | -0,2 px, empate |
| HAM10000 contra SegXNorm | +0,052 [+0,049, +0,056], gana | -0,5 px, gana por poco |

Honestidad: **"supera a los pisos" a secas seria incorrecto**. En ISIC 2018
empata en solapamiento con el piso mas fuerte y gana en borde; el unico
indicador donde un piso supera al estudiante es el percentil 10 de Dice en
HAM10000 (0,530 contra 0,507).

---

## 3. Tono de piel y equidad de la segmentacion

Un defecto encontrado el 2026-09-21 obligo a retirar la primera medicion de
equidad: el bucket Fitzpatrick no medía tono sino el **signo de b\***, porque
`ITA = arctan((L* - 50) / b*)` satura en ±90 grados cuando b* tiende a cero.
En HAM10000 el 40,7 % de las imagenes tenia |b\*| < 5, y el grupo con
b\* ≤ 0 concentraba 2,26 veces mas melanoma: cualquier "brecha por fototipo"
habria medido prevalencia de enfermedad, no tono.

Corregido (commits `c272bdb` y `bbd9bab`): el estimador exige b\* ≥ 5 y el
estrato pasa a terciles de la mediana de L\* de la piel.

| Conjunto | Estrato | n | Dice medio [IC 95 %] | HD95 mediana |
|---|---|---|---|---|
| HAM10000 | claro | 864 | 0,862 [0,854, 0,870] | 9,2 px |
| HAM10000 | medio | 4 876 | 0,803 [0,798, 0,808] | 12,0 px |
| HAM10000 | oscuro | 4 082 | 0,750 [0,744, 0,756] | 14,3 px |
| HAM10000 | brecha claro - oscuro | | **0,111** | |
| ISIC 2018 | claro | 260 | 0,820 [0,802, 0,838] | 11,1 px |
| ISIC 2018 | medio | 304 | 0,802 [0,783, 0,819] | 13,1 px |
| ISIC 2018 | oscuro | 158 | 0,765 [0,737, 0,788] | 15,5 px |
| ISIC 2018 | brecha claro - oscuro | | 0,055 | |

**La brecha de HAM10000 es solida**: los tres intervalos estan separados sin
solaparse, y sobrevive al control por diagnostico — dentro de cada una de las
siete clases el Dice cae de claro a oscuro, monotono en seis. El mecanismo
mas plausible es fisico: a menor luminosidad de la piel hay menos contraste
lesion/fondo, y un metodo sin etiquetas que parte de saliencia sufre mas.

**La de ISIC 2018 no se confirma** al controlar por tamano de lesion y se
reporta como no confirmada en el conjunto externo.

"Oscuro" es el tercio inferior de ESTE pool (L\* medio 62 en HAM10000), no un
fototipo oscuro clinico. El pool es mayoritariamente piel clara.

---

## 4. Clasificacion: los seis backbones

**Datos.** Los backbones se preentrenan sin etiquetas sobre el pool (HAM10000
+ ISIC 2019 sin HAM + ISIC 2020, ver seccion 1). La evaluacion de
clasificacion que sigue es **solo sobre HAM10000**, las 7 clases `dx`, con
`StratifiedGroupKFold` por `lesion_id` (semilla 42, primer pliegue): 8 011
imagenes de entrenamiento del probe y 2 004 de validacion, sin fuga de lesion
entre ambas. ISIC 2019 e ISIC 2020 no traen las 7 etiquetas `dx`, por eso no
entran a esta tabla; ISIC 2018 no se usa en clasificacion.

**Que probe manda para que.** El resultado principal se lee con `logreg`
(`HEADLINE_PROBE`, fijado en el commit `3381cfa`, antes de que existiera el
banco). Los estratos por tono usan `logreg_balanced`
(`STRATA_PREFERRED_PROBE`), porque un estrato chico como el claro necesita el
probe balanceado para no colapsar en la clase mayoritaria. Son dos constantes
distintas a proposito: leer la tabla principal con el probe de los estratos
invierte parte del veredicto, y esa mezcla es un error facil — se declara aca
para que nadie lo repita.

<!-- TABLA-CLASIFICACION:INICIO -->

### Rendimiento

| Backbone | macro-F1 (logreg, 100 %) | macro-F1 (logreg, 10 %) | macro-F1 (balanced, 100 %) |
|---|---|---|---|
| `dinov2_public_vitb14` | 0,616 | 0,449 | 0,613 |
| `ssl_plain_vitb14` | 0,586 | 0,446 | 0,554 |
| `dinov2_public_vits14` | 0,583 | 0,470 | 0,558 |
| `ssl_plain_vits14` | 0,531 | 0,335 | 0,509 |
| `ssl_proposed_vitb14` | 0,524 | 0,363 | 0,537 |
| `ssl_proposed_vits14` | 0,484 | 0,319 | 0,506 |

### Metricas completas

| Backbone | acc | bal_acc | F1 macro | Prec macro | Recall macro | AUC | ECE ↓ |
|---|---|---|---|---|---|---|---|
| `dinov2_public_vitb14` | 0,782 | 0,592 | 0,616 | 0,651 | 0,592 | 0,926 | 0,131 |
| `ssl_plain_vitb14` | 0,799 | 0,543 | 0,586 | 0,655 | 0,543 | 0,945 | 0,016 |
| `dinov2_public_vits14` | 0,791 | 0,560 | 0,583 | 0,627 | 0,560 | 0,932 | 0,081 |
| `ssl_plain_vits14` | 0,783 | 0,477 | 0,531 | 0,686 | 0,477 | 0,931 | 0,021 |
| `ssl_proposed_vitb14` | 0,791 | 0,477 | 0,524 | 0,634 | 0,477 | 0,943 | 0,030 |
| `ssl_proposed_vits14` | 0,775 | 0,436 | 0,484 | 0,599 | 0,436 | 0,921 | 0,038 |

Probe `logreg` al 100% de etiquetas, el mismo para todos.
Las 3 semillas dan valores IDENTICOS a todos los decimales: al 100% de etiquetas no queda nada que submuestrear, asi que son la misma corrida repetida. **El n efectivo es 1 y la columna no lleva dispersion**; la banda tiene que venir de un bootstrap sobre el test, no de las semillas.
**En ECE menor es mejor** (marcado con ↓ en la cabecera); en el resto, mayor es mejor.

### Equidad por tono de piel

| Backbone | Brecha macro-F1 | Estrato peor | Intervalos separados |
|---|---|---|---|
| `dinov2_public_vitb14` | 0,014 | dark | no |
| `dinov2_public_vits14` | 0,091 | light | no |
| `ssl_plain_vitb14` | 0,108 | light | no |
| `ssl_plain_vits14` | 0,048 | light | no |
| `ssl_proposed_vitb14` | 0,034 | light | no |
| `ssl_proposed_vits14` | 0,097 | light | no |

Estratos medidos con probe `logreg_balanced` al 100%.
Un `no` significa que los intervalos de los estratos extremos se solapan:
esa brecha NO se cita como efecto de tono por grande que sea el numero.
`sin datos` no es lo mismo que ausencia de brecha.

_Generado desde `bank.json` por `scripts/build_results_readme.py`; modo de la corrida: full._

<!-- TABLA-CLASIFICACION:FIN -->

Diseno de la comparacion. Cada arquitectura entra con **su propia** linea
base publica, porque comparar un ViT-S adaptado contra un ViT-B publico
confundiria la adaptacion con la capacidad del modelo.

**Los cuatro brazos SSL NO parten de cero.** Se inicializan con los pesos
publicos de DINOv2 (`initial_weights = dinov2-public-lvd142m` en los cuatro
`run.json`) y hacen SimSiam de dominio encima. Por eso la comparacion honesta
es **"DINOv2 + adaptacion de dominio" contra "DINOv2 solo"**, no "un SSL
entrenado desde cero contra DINOv2". La ganancia mide lo que aporta la
adaptacion sobre el backbone publico, no reemplaza a LVD-142M.

| Backbone | Origen | Papel |
|---|---|---|
| `dinov2_public_vits14` | DINOv2 ViT-S/14 LVD-142M, sin adaptar | Referencia del tamano chico |
| `dinov2_public_vitb14` | DINOv2 ViT-B/14 LVD-142M, sin adaptar | Referencia del tamano grande |
| `ssl_plain_vits14` | DINOv2 ViT-S + SimSiam de dominio (dos vistas) | Control chico |
| `ssl_proposed_vits14` | DINOv2 ViT-S + SimSiam con pares de tono | Propuesto chico |
| `ssl_plain_vitb14` | DINOv2 ViT-B + SimSiam de dominio (dos vistas) | Control grande |
| `ssl_proposed_vitb14` | DINOv2 ViT-B + SimSiam con pares de tono | Propuesto grande |

Protocolo: `StratifiedGroupKFold` por `lesion_id`, semilla 42, primer
pliegue, identico en los seis. Probes `logreg`, `logreg_balanced` y `knn`, a
100 % y 10 % de etiquetas, tres semillas. El presupuesto es en **horas-GPU**,
no en epocas ni en muestras: a igual presupuesto el backbone grande ve menos
muestras, y esa es la comparacion que declara la tesis.

La frase precisa es **"a igual presupuesto de computo"**, nunca "a igual
entrenamiento" ni "a igual numero de muestras". Lo que se iguala son las
12,00 horas-GPU que recibe cada brazo; lo que no se iguala es el trabajo
hecho dentro de ellas, porque un paso del ViT-B cuesta mas que uno del
ViT-S y un paso pareado lee dos imagenes en vez de una. Por eso la tabla de
arriba muestra horas-GPU y muestras **una al lado de la otra**: es lo que
impide leer "igual computo" como "igual entrenamiento".

### Corrida preliminar (4 horas-GPU, solo ViT-S, con un defecto conocido)

Se conserva porque muestra la direccion del efecto, pero **no es la cifra
final**: el brazo propuesto entrenaba con el doble de gradiente por un
termino de perdida duplicado, corregido en el commit `08727e8`.

| Backbone | macro-F1 (logreg) | Brecha entre terciles |
|---|---|---|
| `dinov2_public` | 0,583 | 0,091 |
| `ssl_proposed` | 0,562 | 0,102 |
| `ssl_plain` | 0,481 | 0,162 |

El propuesto supera al control en **cinco de las seis** combinaciones de
probe y regimen; la excepcion es kNN al 10 % de etiquetas (0,705 contra
0,715).

---

## 5. Que gano

Leido sobre la tabla de metricas completas, con el probe fijado **antes** de
tener datos (`HEADLINE_PROBE = "logreg"`, commit `3381cfa`, 2026-09-21, un dia
antes de que cerrara el banco).

- **La adaptacion de dominio sobre DINOv2 produce los backbones mejor
  calibrados, y por mucho.** Los cuatro brazos adaptados tienen un ECE varias
  veces menor que los dos DINOv2 publicos sin adaptar. Es la diferencia mas
  grande de todo el panel y la unica que no depende del probe: aparece igual
  con `logreg` y con `logreg_balanced`. Como los cuatro parten de los mismos
  pesos publicos, la mejora de calibracion es atribuible a las 12 horas-GPU de
  SimSiam sobre dermatoscopias, no a la arquitectura. Para un clasificador
  medico, una probabilidad que significa lo que dice es una propiedad util por
  si sola.
- **`ssl_plain_vitb14` (DINOv2 ViT-B + adaptacion de dominio) supera a DINOv2
  ViT-B sin adaptar en accuracy, AUC, precision macro y calibracion.** No es
  "un SSL desde cero le gana a DINOv2": es que agregarle 12 horas-GPU de
  SimSiam de dominio al backbone publico lo mejora. En AUC los dos ViT-B
  adaptados superan a su base publica; en ViT-S el control queda a la par y el
  propuesto por debajo.
- **La cadena de entrenamiento es reproducible y resistente a fallas**: un
  cuelgue de NCCL al 95 % se recupero solo, y el resultado cayo dentro de la
  prediccion registrada antes de conocerlo
  ([prediccion-brazo3-interrupcion.md](prediccion-brazo3-interrupcion.md)).
- **La segmentacion sin etiquetas es utilizable** como paso previo: mediana
  de Dice 0,79 y 0,85, y la cuarentena detecta el 6 % de imagenes sin mascara
  confiable en vez de inventar una.

---

## 6. Que perdio

- **El emparejamiento por tono no mejora al control; lo empeora.** Con el
  probe registrado, `ssl_proposed` queda por debajo de `ssl_plain` en macro-F1,
  balanced accuracy y AUC **en los dos tamanos y en los dos regimenes de
  etiquetas**: doce comparaciones, doce en contra. La ganancia de la corrida preliminar (+0,081) **no se
  replica**: aquella corrida tenia el termino de perdida duplicado
  (`08727e8`), y con el defecto corregido y el triple de presupuesto la
  direccion se invierte.
- **Ningun backbone adaptado alcanza a DINOv2 publico en macro-F1 ni en
  balanced accuracy.** Los adaptados aciertan mas en global y separan mejor
  (AUC), pero recuperan peor las clases minoritarias: la adaptacion de dominio
  mejora accuracy, AUC y calibracion pero no el recall de las clases raras. Es
  el patron de un clasificador que se apoya en la clase mayoritaria (nevo):
  accuracy alta, recall macro bajo.
- **Con `logreg_balanced` el propuesto ViT-B gana en balanced accuracy y
  AUC.** Se reporta porque existe, pero **no se usa como resultado**: ese
  probe no fue el registrado, y elegir el probe despues de ver cual favorece
  al metodo es exactamente la seleccion que invalida una comparacion. Que el
  veredicto cambie con el probe es, en si mismo, evidencia de que la
  diferencia es chica.
- **El estudiante de segmentacion no supera al piso mas fuerte** en
  solapamiento medio sobre ISIC 2018; gana en borde y en fallos
  catastroficos.

---

## 7. Que no se puede afirmar

- **Ninguna diferencia de clasificacion tiene todavia intervalo de
  confianza.** Al 100 % de etiquetas las tres semillas son la misma corrida
  (n efectivo = 1, ver la nota bajo la tabla). El bootstrap pareado sobre el
  test esta en curso; hasta que llegue, "supera" en esta pagina significa
  "estimacion puntual mayor", no "diferencia establecida".
- **Ningun efecto de tono en clasificacion, en ninguna direccion.** En los
  seis backbones los intervalos de los estratos extremos se solapan
  (`Intervalos separados = no`). El estrato claro tiene 183 muestras contra
  820 y 957, y es el que mueve cada brecha. Ni "la adaptacion cierra la
  brecha" ni "la adaptacion la amplia" son afirmables con estos datos: la
  brecha cambia de orden segun se mida en macro-F1 o en AUC, que es lo que
  hace el ruido.
- **Que el propuesto pierda por el objetivo y no por ver menos datos**, solo
  en parte. El brazo pareado procesa menos muestras que su control al mismo
  presupuesto, pero la diferencia es de un **3 %** (tabla de entrenamiento).
  Un 3 % menos de datos explica mal una caida de varios puntos de macro-F1,
  asi que la lectura mas probable es que el objetivo actual no ayuda. La
  prueba definitiva es una corrida a igual numero de muestras, que no se
  hizo.
- **La version de scikit-learn del banco no quedo registrada.** El pliegue de
  `StratifiedGroupKFold` depende de ella. La identidad del conjunto de test
  se verifica por reproduccion exacta de las metricas publicadas, no por un
  registro de ids que el banco no guardo.
- **Nada sobre fototipos altos reales.** El pool es mayoritariamente piel
  clara. Sin datos externos no hay base para afirmar comportamiento en piel
  oscura en sentido clinico.
- **Nada clinico.** Ninguna de estas cifras se obtuvo en un entorno de uso
  real ni con validacion independiente.

---

## 8. Como mejorarlo

En orden de retorno esperado:

1. **Bootstrap pareado sobre el test** (en curso). No mejora el modelo, pero
   decide cuales de las diferencias de arriba existen. Es lo primero porque
   cambia el verbo de cada frase de las secciones 5 y 6.
2. **Un objetivo con dos terminos de verdad.** Hoy el brazo propuesto usa una
   sola perdida que mezcla invarianza a la augmentacion con invarianza al
   tono. Tres vistas — dos augmentaciones de la original mas la transformada
   — separarian las dos senales, y recien ahi `--consistency-weight` seria
   una palanca real: con el diseno actual escala la perdida entera y AdamW es
   casi invariante a la escala.
3. **Desplazamiento de tono variable por imagen.** Hoy es fijo en dos niveles
   hacia el oscuro; Corbin y Marques 2023 usan hasta dos niveles con signo
   aleatorio, exponiendo al modelo a las dos direcciones.
4. **Pseudo-mascaras de la ronda 3.** Las usadas son las de la ronda 2; el
   estudiante final no exporta predicciones sobre el pool. Exportarlas es
   trabajo mecanico y mejora la entrada de todo lo demas.
5. **Resolucion.** Todo se materializa a 224 para no transformar ISIC 2020 en
   resolucion nativa. Subirla eleva el techo, a costa de espacio y computo.
6. **Validacion externa en fototipos altos**, sin la cual la seccion de
   equidad no puede extrapolarse fuera de este pool.
7. **Agrupacion por paciente en ISIC 2019**, que el release no trae y por eso
   sus cifras cargan sesgo optimista.

---

## 9. El modelo: descargar y probar

Los pesos se publican en las **releases** de este repositorio, cada uno con
su `run.json` — sin ese archivo un `.pt` no es citable, porque no se sabe con
que datos ni con que presupuesto salio.

```bash
# el checkpoint carga y produce caracteristicas
python scripts/try_model.py --backbone ssl_proposed_backbone.pt --image lesion.jpg

# el checkpoint sirve para clasificar
python scripts/try_model.py --backbone ssl_proposed_backbone.pt \
    --ham-root /datos/ham10000 --limit 400
```

Detalle completo, preprocesamiento exacto y limitaciones de uso:
[`docs/23-modelo-descargable-y-uso.md`](../docs/23-modelo-descargable-y-uso.md).

**No es un dispositivo de diagnostico.** Sin validacion clinica, entrenado
sobre un conjunto mayoritariamente de piel clara, y con peor desempeno en el
tercil de piel menos luminosa.

---

## 10. Reproducir todo

Una sola linea, idempotente, pensada para un cluster que se cae:

```bash
python scripts/run_stage2.py \
    --repo . --outputs $HOME/work/proyectonp/salidas \
    --ham-root /ruta/a/ham10000 \
    --archs dinov2_vits14 dinov2_vitb14 \
    --budget-gpu-hours 12
```

Cada etapa declara el artefacto que prueba que termino: las hechas se saltan
y las interrumpidas se relanzan igual. Si el contenedor muere, se repite la
misma linea. Checkpoints cada 5 % del presupuesto.

Seguimiento en vivo:
[Weights & Biases, proyecto `dermssl-segmentacion`](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion).
