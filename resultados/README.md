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

_Pendiente: se genera desde los `run.json` de los brazos._

<!-- TABLA-ENTRENAMIENTO:FIN -->

---

## 2. Segmentacion sin mascaras ni etiquetas

Commit `d81a492`, preset `cluster-final`. MoCo v2 sobre el pool, saliencia
CCAM, modulo de incertidumbre y U-Net estudiante con re-etiquetado en tres
rondas.

| Conjunto | n | Dice medio [IC 95 %] | Dice mediana | IoU | HD95 mediana |
|---|---|---|---|---|---|
| ISIC 2018 | 2 594 | 0,718 [0,709, 0,726] | 0,792 | 0,602 | 13,6 px |
| HAM10000 | 10 015 | 0,785 [0,781, 0,788] | 0,852 | 0,678 | 12,6 px |

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

<!-- TABLA-CLASIFICACION:INICIO -->

_Pendiente: las tablas se generan solas desde `bank.json` con
`scripts/build_results_readme.py` en cuanto cierre el banco. No se transcriben
a mano: un numero copiado sobrevive a la corrida que lo cambio._

<!-- TABLA-CLASIFICACION:FIN -->

Diseno de la comparacion. Cada arquitectura entra con **su propia** linea
base publica, porque comparar un ViT-S adaptado contra un ViT-B publico
confundiria la adaptacion con la capacidad del modelo.

| Backbone | Origen | Papel |
|---|---|---|
| `dinov2_public_vits14` | DINOv2 ViT-S/14 LVD-142M | Referencia del tamano chico |
| `dinov2_public_vitb14` | DINOv2 ViT-B/14 LVD-142M | Referencia del tamano grande |
| `ssl_plain_vits14` | SimSiam sobre el pool, dos vistas | Control chico |
| `ssl_proposed_vits14` | SimSiam con pares de tono | Propuesto chico |
| `ssl_plain_vitb14` | SimSiam sobre el pool, dos vistas | Control grande |
| `ssl_proposed_vitb14` | SimSiam con pares de tono | Propuesto grande |

Protocolo: `StratifiedGroupKFold` por `lesion_id`, semilla 42, primer
pliegue, identico en los seis. Probes `logreg`, `logreg_balanced` y `knn`, a
100 % y 10 % de etiquetas, tres semillas. El presupuesto es en **horas-GPU**,
no en epocas: a igual presupuesto el backbone grande ve menos muestras, y esa
es la comparacion a igual computo que declara la tesis.

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

<!-- PENDIENTE: completar con la corrida definitiva -->

De la corrida preliminar, y a confirmar con la definitiva:

- **La adaptacion de tono aporta sobre su propio control**: +0,081 de macro-F1
  con `logreg` al 100 % de etiquetas, +0,083 al 10 %, y la brecha entre
  terciles baja de 0,162 a 0,102 (un 37 % menos). Como los dos brazos
  comparten datos, presupuesto, semilla, particion y protocolo, y la unica
  diferencia es la segunda vista, la mejora es atribuible a la adaptacion.
- **La segmentacion sin etiquetas es utilizable** como paso previo: mediana
  de Dice 0,79 y 0,85, y la cuarentena detecta el 6 % de imagenes sin mascara
  confiable en vez de inventar una.

---

## 6. Que perdio

<!-- PENDIENTE: completar con la corrida definitiva -->

- **Ningun brazo adaptado alcanzo al backbone publico sin adaptar** en la
  corrida preliminar (0,562 contra 0,583 con 4 horas-GPU). Cuatro horas de
  GPU sobre 55 mil imagenes no compiten con LVD-142M. La corrida definitiva
  triplica el presupuesto; si aun asi no alcanza, se reporta igual.
- **kNN al 10 % de etiquetas**: el propuesto queda por debajo del control
  (0,705 contra 0,715). Es la unica de las seis comparaciones que pierde.
- **El estudiante de segmentacion no supera al piso mas fuerte** en
  solapamiento medio sobre ISIC 2018; gana en borde y en fallos
  catastroficos.

---

## 7. Que no se puede afirmar

- **La direccion de la brecha de tono en clasificacion.** En la corrida
  preliminar el estrato peor no era el oscuro sino el claro, en los tres
  backbones — pero ese estrato tenia 183 muestras contra 820 y 957, sobre
  siete clases. El estrato sin tono asignado, con 44 muestras, era el peor de
  todos, que es la firma de inestabilidad por tamano de muestra. La corrida
  definitiva incorpora **intervalos de confianza por estrato** y un campo
  `gap_established` que dice si los extremos estan realmente separados.
- **Que el metodo propuesto sea mejor *como objetivo*, separado de su costo.**
  El brazo pareado lee dos imagenes por paso, asi que **al mismo presupuesto
  de horas-GPU procesa menos muestras que su propio control**. Esa es la
  comparacion que la tesis declara y la que le importa a quien tiene una GPU
  y un plazo: a igual computo, cual conviene. Pero tiene un limite que hay
  que decir en voz alta y en las dos direcciones. Si el propuesto **gana**,
  gana habiendo visto menos datos, y el resultado es mas fuerte de lo que
  parece. Si **pierde**, no se puede concluir que el emparejado por tono sea
  peor: puede estar perdiendo por haber visto menos muestras, no por el
  objetivo. Distinguir las dos cosas exige una corrida adicional a igual
  numero de muestras, que no se hizo. Por eso las tablas reportan pasos y
  muestras de cada brazo, y no solo el presupuesto.
- **Nada sobre fototipos altos reales.** El pool es mayoritariamente piel
  clara. Sin datos externos no hay base para afirmar comportamiento en piel
  oscura en sentido clinico.
- **Nada clinico.** Ninguna de estas cifras se obtuvo en un entorno de uso
  real ni con validacion independiente.

---

## 8. Como mejorarlo

En orden de retorno esperado:

1. **Mas presupuesto de preentrenamiento.** Es la palanca que ataca el unico
   resultado adverso. La corrida definitiva ya triplica el presupuesto de la
   preliminar.
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
