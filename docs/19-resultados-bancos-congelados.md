# 19. Resultados de los bancos congelados y como leerlos

> **Documento histórico.** Describe una etapa anterior del proyecto y fue superado. Los resultados vigentes están en el [README](../README.md) y en los documentos 27 a 33. Se conserva para mostrar cómo se llegó al diseño actual.

Cierra la etapa de comparadores del Objetivo 2: ocho backbones congelados sobre
HAM10000, ISIC 2020 e ISIC 2019, con el mismo protocolo y la misma semilla. Las
tablas viven en el README principal y se generan con
`python scripts/build_results_tables.py --bank <banco>`; este documento explica
que se puede concluir de ellas y que no.

## 1. Que se corrio

Veinticuatro corridas reales en Kaggle (`SMOKE=False`, semilla 42, resolucion 224,
backbone congelado, tres probes: kNN, regresion logistica y prototipos). Los JSON
estan en `results/baselines/`, `results/baselines_isic2020/` y
`results/baselines_isic2019/`, y traen metricas por clase, matriz de confusion e
intervalos bootstrap del 95 por ciento.

Los tres bancos NO son intercambiables, y la diferencia decide que se puede leer
en cada uno:

| Banco | Tarea | Agrupacion anti-fuga | Comparable con Gessert |
|---|---|---|---|
| HAM10000 | 7 clases | `lesion_id` real | no (otra tarea) |
| ISIC 2020 | binaria | `patient_id` real | no (otra tarea) |
| ISIC 2019 | 8 clases | ninguna (el release no la trae) | **si, misma tarea** |

ISIC 2019 es el unico cuya tarea coincide con el ancla de la literatura, y por eso
es el unico que se puede leer de frente contra ella. El precio es que ese release
no trae columna de paciente, asi que su split es estratificado simple y sus
metricas cargan sesgo optimista.

Ademas, **HAM10000 esta contenido entero dentro de ISIC 2019**: las 10015 imagenes
de HAM aparecen en ISIC 2019, que son el 39.5 por ciento de sus 25331. Es
subconjunto estricto, medido cruzando los dos ground truth oficiales. Los bancos de
HAM10000 y de ISIC 2019 no son evaluaciones independientes y no deben leerse como
si lo fueran.

## 2. El error de medicion que aparecio, y como se detecto

Al migrar los pesos a Hugging Face se asumio que `timm.create_model(..., num_classes=0)`
devuelve el promedio global del ultimo bloque convolucional. Para VGG19 no es asi:
devuelve las pre-logits de 4096 dimensiones, o sea la salida de las capas totalmente
conectadas fc6 y fc7 de la cabeza. Los pesos eran identicos a los de torchvision,
pero el punto de medicion se habia movido dos capas hacia adentro.

Se detecto porque el pipeline es determinista. Al re-correr HAM10000, AlexNet,
EfficientNet-B0 y ResNet-50 reprodujeron sus numeros al cuarto decimal, mientras
VGG19 se movio 0.0435 y su `dim_features` salto de 512 a 4096. Un cambio aislado en
un pipeline reproducible no es ruido: es un cambio de definicion.

El arreglo lee VGG19 con `gap(m.features)` y, sobre todo, ambos ejecutores
(el generador de notebooks y el runner del cluster) ahora comprueban
`protocol.FEATURE_DIMS` antes de extraer. VGG19 volvio a dar 0.5147, exactamente el
valor previo a la migracion, que es la evidencia de que el arreglo es correcto.

**Leccion transferible**: cuando dos configuraciones deben ser equivalentes, hay que
verificar la equivalencia de la MEDICION, no solo la de los pesos. Una assertion
sobre la dimension del vector de features cuesta una linea y habria hecho fallar la
corrida en el primer segundo en lugar de producir un numero plausible pero falso.

## 3. Que se sostiene

- **En la tarea comparable con la literatura, el SSL generico lidera.** En ISIC
  2019 los dos DINOv2 ocupan el primer y el segundo puesto: ViT-B/14 con 0.620 y
  ViT-S/14 con 0.597, por encima del ViT-B/16 supervisado (0.579). El ViT-S/14 lo
  consigue con 384 dimensiones y 21 millones de parametros.
- **Ningun backbone generico congelado alcanza el ancla clinica** (Gessert, 0.636)
  en estimacion puntual. El mejor de ISIC 2019 queda a 0.016, aunque su intervalo
  [0.591, 0.649] la cubre: con estos datos no se distingue de la marca ganadora del
  challenge, que no es lo mismo que igualarla.
- **El SSL generico de ImageNet no transfiere a dermatoscopia.** SimSiam ResNet-50
  es ultimo en HAM10000 y en ISIC 2019, y en ambos su intervalo no toca al del
  lider: esa diferencia si es real. Es el argumento que motiva la fase v6,
  pre-entrenar sobre piel.
- **Ningun backbone congelado sirve para tamizaje en ISIC 2020.** La precision de
  la clase maligna va de 0.045 a 0.076 en los ocho, es decir unos trece falsos
  positivos por cada acierto. La balanced accuracy de 0.77 es cosmetica: viene de un
  desbalance cercano a 50 a 1.
- El probe kNN colapsa en HAM10000 (0.26 a 0.40). Con desbalance 58 a 1 el
  vecindario lo domina `nv`. La regresion logistica con pesos de clase es el probe
  informativo.

## 4. Que NO se puede afirmar

El ranking no resuelve su propio medio, y el grado de resolucion cambia por banco:
en ISIC 2020 se solapan los ocho intervalos con el del lider, en HAM10000 seis de
ocho, y en ISIC 2019 cinco de ocho. Las posiciones intermedias no son un orden de
merito: son empates que el conjunto de validacion no separa. ISIC 2019 es el banco
con mas resolucion de los tres, y aun asi no separa su medio.

En particular, **no se puede escribir que el SSL le gana al supervisado**. Lo que se
puede escribir es que la ventaja del supervisado desaparece a igual tamano: con los
pesos `augreg_in1k` (el comparador honesto, entrenado solo con ImageNet-1k), el
ViT-B/16 queda por debajo de DINOv2 ViT-B/14 en los dos bancos multiclase, por 0.051
en HAM10000 y por 0.041 en ISIC 2019, pero los intervalos se solapan en ambos.

Tampoco se puede afirmar paridad con Gessert. El intervalo del lider cubre el ancla,
pero ese intervalo se calculo sobre un split sin agrupacion por paciente, que sesga
los valores absolutos hacia arriba. El sesgo afecta el nivel, no el orden.

### 4.1 Una hipotesis propia que el dato descarto

AlexNet quedo primero en ISIC 2020 y segundo en HAM10000. Un backbone de 2012 con
256 dimensiones por encima de DINOv2 es lo bastante raro como para buscarle causa, y
se propuso una: el probe de regresion logistica se entrena sobre las features SIN
normalizar (`logreg.fit(X_tr, ...)`, mientras kNN y prototipos usan `Xn_tr`) y con
`C=1.0` fijo para dimensionalidades que van de 256 a 2048, lo que penalizaria mas a
los espacios de mayor dimension y favoreceria sistematicamente a los backbones
chicos.

La hipotesis predecia que AlexNet volveria a quedar primero en ISIC 2019. **Quedo
sexto** (0.533), detras de los dos DINOv2, del ViT-B/16, de EfficientNet-B0 y de
ResNet-50. La prediccion fallo, asi que la hipotesis no explica el fenomeno.

La lectura que si se sostiene es mas simple: el orden de ISIC 2020 nunca fue
interpretable. Es una tarea binaria con desbalance cercano a 50 a 1, unos 133
positivos en validacion, y los ocho intervalos solapados entre si. No habia ranking
que explicar.

Queda en pie, como limitacion documentada y no como causa de nada, que entrenar el
probe lineal sobre features sin normalizar es una decision sin justificacion
explicita en el protocolo. Revisarla es trabajo pendiente; atribuirle el orden
observado seria repetir el error de dar por buena una explicacion que ya fallo una
prediccion.

## 5. Un fallo silencioso de infraestructura, por si vuelve

`kaggle kernels output` salio con codigo distinto de cero mientras el kernel ya
figuraba COMPLETE: Kaggle reporta la finalizacion un instante antes de que el output
sea descargable. El driver ignoraba ese codigo, asi que dio por perdida la corrida de
`dinov2_vitb14` en ISIC 2019 con un "missing result file", cuando el JSON estaba
sano y se descargo sin problema segundos despues. Una corrida de GPU terminada
figuro como fallida.

Corregido con reintentos y reporte explicito del fallo real. Es el mismo patron que
el error de medicion de la seccion 2: la herramienta no se rompio a gritos, devolvio
un estado plausible y equivocado. En este proyecto conviene asumir que cualquier
paso que pueda fallar en silencio, va a hacerlo.

## 6. Consecuencia para el proyecto

La pregunta del Objetivo 2 queda mejor planteada que antes: no se trata de ganarle a
un numero, sino de conseguir una ventaja que sobreviva a su propio intervalo de
confianza. Eso exige, en la fase v6, o bien un efecto mayor (pre-entrenamiento sobre
piel), o bien mas potencia estadistica (varias semillas y validacion cruzada), y
preferiblemente ambas cosas.

El banco de ISIC 2019 aporta el mejor argumento disponible hasta ahora: en la unica
tarea comparable con la literatura, y sin haber visto nunca una imagen
dermatoscopica, un backbone auto-supervisado congelado queda a 0.016 del ganador
oficial del challenge y su intervalo lo cubre. Ese es el punto de partida que la
fase v6 tiene que superar de forma medible, no el 0.567 del comparador supervisado.

Queda pendiente una decision de alcance que este documento no resuelve: cuatro
documentos del repositorio describen ISIC 2019 como conjunto de validacion externa,
y el solape del 39.5 por ciento con HAM10000 lo desmiente. La validacion externa
honesta seria el complemento `ISIC 2019 \ HAM10000`, unas 15316 imagenes.

## 7. Lo que cambio con los bancos de fine-tuning (2026-09-15)

Este documento describe los bancos congelados. Los dos bancos con fine-tuning
(HAM10000 e ISIC 2019, 8 de 8 cada uno) y el probe de learning rate estan en el
README, seccion "Baselines CON fine-tuning", con sus tablas generadas y la
tabla de decisiones del protocolo. Tres cosas de ahi corrigen la lectura de
este documento:

1. **El numero a superar ya no es el del congelado.** En ISIC 2019, la unica
   tarea comparable con Gessert (0.636), ViT-B/16 supervisado con fine-tuning
   llega a 0.805 de balanced accuracy, con intervalo [0.780, 0.828]. La fase v6
   tiene que medirse contra eso.
2. **La ventaja del DINOv2 congelado en ISIC 2019 no sobrevive al fine-tuning
   con el protocolo actual, y la razon es el protocolo.** Los dos DINOv2 caen
   por debajo de su propio congelado en ambos bancos. El probe de learning rate
   (`LR_BACKBONE` 5e-6 en vez de 5e-5, todo lo demas identico) lleva a ViT-B/14
   de 0.532 a 0.762 sobre HAM10000, convergiendo en la epoca 3. Un learning
   rate unico para ocho backbones no es neutral entre familias.
3. **La hipotesis "los backbones SSL necesitan mas presupuesto de fine-tuning"
   queda descartada.** Necesitaban otro learning rate, no mas epocas. Lo que
   sigue pendiente es decision de alcance: learning rate por familia con un
   control supervisado, y el probe sobre ISIC 2019.
