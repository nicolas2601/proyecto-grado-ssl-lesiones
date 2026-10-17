# Resultados completos del pipeline propuesto

Todo lo que produjo el proyecto, en un solo lugar: segmentacion sin
etiquetas, tono de piel, los cuatro brazos de preentrenamiento y el
clasificador. Que gano, que perdio, y como mejorarlo.

> Navegación: [README principal](../README.md) · [análisis final](../docs/24-analisis-final-resultados-y-escenarios-ada.md) · [fuentes y ramas](../docs/00-fuentes-y-ramas.md) · [demo](../demo/README.md).

## Dónde está cada resultado

| Resultado | Tabla (generada) | Datos | Estado |
|---|---|---|---|
| v7: las tres semillas combinadas (lectura principal) | [clasificacion/v7/SEMILLAS.md](clasificacion/v7/SEMILLAS.md) | los tres `final_eval.json` | Vigente |
| v7: evaluación única Modelo de control, Proposed sin segmentación y DINOv2 público sobre val, semilla 42 | [clasificacion/v7/README.md](clasificacion/v7/README.md) | `clasificacion/v7/final_eval.json` | Vigente |
| v7: réplica con semilla 43 (registrada en docs/28 §4.1) | [clasificacion/v7/s43/README.md](clasificacion/v7/s43/README.md) | `clasificacion/v7/s43/final_eval.json` | Vigente |
| v7: réplica con semilla 44 (registrada en docs/29 §2.5), con su procedencia (control s42 y arranque abortado) | [clasificacion/v7/s44/README.md](clasificacion/v7/s44/README.md) | `clasificacion/v7/s44/final_eval.json`, `clasificacion/v7/s44/eval_provenance.json` | Vigente |
| Clasificador: v6, C1 y v7 lado a lado, con matriz de confusión, F1 y recall por clase y melanoma | [clasificacion/TABLA_CLASIFICADOR.md](clasificacion/TABLA_CLASIFICADOR.md) | `clasificacion/bank.json`, `clasificacion/c1/c1_cells.json`, los tres `final_eval.json` y `final_eval.probas.npz` | Vigente |
| Segmentación: Experimento de segmentación 3, el publicado (Dice, IoU, HD95) y pseudo-máscaras v7 contra los umbrales de docs/27 | [segmentacion/TABLA_SEGMENTACION.md](segmentacion/TABLA_SEGMENTACION.md) | `summary.json` y `*_stats.json` de `plan-segmentacion/camino-b-cnn/resultados/` | Vigente |
| Cierre: pedidos del profesor y plan de Pau → qué se hizo → resultado, con veredicto por regla | [TABLA_CIERRE.md](TABLA_CIERRE.md) | los JSON de `clasificacion/` y de `plan-segmentacion/camino-b-cnn/resultados/` | Vigente |
| C1: grilla de condiciones de máscara por backbone | [clasificacion/c1/README.md](clasificacion/c1/README.md) | `clasificacion/c1/*.json` | Vigente |
| v6: bancos congelados y bootstrap pareado | Secciones 4 a 7 de este documento | `clasificacion/bank.json`, `clasificacion/bootstrap.json` | Superado por v7 |
| Segmentación y tono | Secciones 2 y 3 de este documento | `segmentacion/`, `tono/` | Vigente |
| Baselines | [baselines/README.md](baselines/README.md) | `baselines/` | Histórico |

Las tablas generadas tienen un test que falla si no coinciden con su JSON (`tests/test_generated_readmes_fresh.py`, `tests/test_resultados_readme_fresh.py`).

## Regla de lectura

Esta carpeta es la fuente canónica de los resultados del pipeline propuesto.
Cada conclusión debe leerse junto con el artefacto que la produce: JSON,
CSV, manifiesto, log o commit. Los bancos anteriores permanecen en
[`results/`](../results/README.md) y no sustituyen los artefactos de esta
carpeta.

Cada cifra cita el commit, el preset y el artefacto del que sale. Lo que no
esta medido se dice que no esta medido.

## Indice

1. [Que se entreno y con que](#1-que-se-entreno-y-con-que)
2. [Segmentacion sin mascaras ni etiquetas](#2-segmentacion-sin-mascaras-ni-etiquetas)
3. [Terciles de luminancia y equidad de la segmentacion](#3-terciles-de-luminancia-y-equidad-de-la-segmentacion)
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
| Luminancia de la piel | Tercil de luminancia y buckets de la estimación ITA no validada (no es FST) por imagen | Cerrada |
| Pares de tono | 55 034 pares original / piel oscurecida | Cerrada |
| Preentrenamiento ×4 | Backbones adaptados (ViT-S y ViT-B, con y sin tono) | Cerrada |
| Banco congelado | Tabla de clasificacion con 6 backbones | Cerrada |

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
CCAM, modulo de incertidumbre y U-Net entrenada con pseudo-máscaras con re-etiquetado en tres
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
indicador donde un piso supera al experimento de segmentación es el percentil 10 de Dice en
HAM10000 (0,530 contra 0,507).

---

## 3. Terciles de luminancia y equidad de la segmentacion

> **Nota: estimación ITA no validada (no es FST).** Lo que aquí aparece como «FST», «fototipo» o «bucket Fitzpatrick» por imagen es el ITA con los umbrales fijos de Groh (`src/dermssl/fitzpatrick.py`), y no está validado. Contra FST asignado por expertos acierta 28,5–30,8 % frente a 32,9 % de la baseline trivial en Fitzpatrick17k-C (11.098 imágenes, docs/20-validacion-externa-fst-fitzpatrick17k.md); en PAD-UFES-20, el ITA con umbrales fijos acierta 22,9 % frente a 51,3 % de la baseline de mayoría (Alipour et al., *Skin Research and Technology* 2026, PMC13459198; FST asignado por dermatólogo según ese artículo). Los terciles `tone_tercile` son terciles de luminancia (mediana de L\* de la piel), no tono de piel.

Un defecto encontrado el 2026-09-21 obligo a retirar la primera medicion de
equidad: el bucket Fitzpatrick no medía tono sino el **signo de b\***, porque
`ITA = arctan((L* - 50) / b*)` satura en ±90 grados cuando b* tiende a cero.
En HAM10000 el 40,7 % de las imagenes tenia |b\*| < 5, y el grupo con
b\* ≤ 0 concentraba 2,26 veces mas melanoma: cualquier "brecha por fototipo"
habria medido prevalencia de enfermedad, no tono.

Corregido (commits `c272bdb` y `bbd9bab`): el estimador exige b\* ≥ 5 y el
estrato pasa a terciles de luminancia: terciles de la mediana de L\* de la piel.

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

> **Etapa v6 (superada).** Las secciones 4, 5 y 6 describen la evaluación de la versión 6. El resultado vigente es el de la versión 7, en el [README principal](../README.md).

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

### Equidad por terciles de luminancia

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
esa brecha NO se cita como efecto de luminancia por grande que sea el numero.
Los estratos son terciles de luminancia (mediana de L\* de la piel), no tono
de piel ni fototipo validado.
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

### Contrastes con intervalo de confianza (bootstrap pareado)

Cada diferencia de arriba es una estimacion puntual. La tabla siguiente le
pone banda: bootstrap pareado sobre el test (los dos brazos remuestrean los
MISMOS indices, para no inflar la varianza), a `logreg` y 100 % de etiquetas.
**Un intervalo que incluye el cero no se cita como efecto**, por grande que
sea el punto.

<!-- TABLA-CONTRASTES:INICIO -->

| Contraste | Metrica | Delta | IC 95 % | Veredicto |
|---|---|---|---|---|
| `ssl_proposed_vits14` - `ssl_plain_vits14` | macro_f1 | -0,0467 | [-0,0878, -0,0036] | gana `ssl_plain_vits14` |
| `ssl_proposed_vits14` - `ssl_plain_vits14` | balanced_accuracy | -0,0404 | [-0,0769, -0,0060] | gana `ssl_plain_vits14` |
| `ssl_proposed_vits14` - `ssl_plain_vits14` | auc | -0,0095 | [-0,0216, 0,0045] | no se distingue del ruido |
| `ssl_proposed_vits14` - `ssl_plain_vits14` | ece | 0,0175 | [-0,0036, 0,0293] | no se distingue del ruido |
| `ssl_proposed_vitb14` - `ssl_plain_vitb14` | macro_f1 | -0,0619 | [-0,1131, -0,0109] | gana `ssl_plain_vitb14` |
| `ssl_proposed_vitb14` - `ssl_plain_vitb14` | balanced_accuracy | -0,0655 | [-0,1119, -0,0193] | gana `ssl_plain_vitb14` |
| `ssl_proposed_vitb14` - `ssl_plain_vitb14` | auc | -0,0013 | [-0,0070, 0,0054] | no se distingue del ruido |
| `ssl_proposed_vitb14` - `ssl_plain_vitb14` | ece | 0,0136 | [-0,0068, 0,0254] | no se distingue del ruido |
| `ssl_proposed_vits14` - `dinov2_public_vits14` | macro_f1 | -0,0989 | [-0,1499, -0,0495] | gana `dinov2_public_vits14` |
| `ssl_proposed_vits14` - `dinov2_public_vits14` | balanced_accuracy | -0,1241 | [-0,1711, -0,0802] | gana `dinov2_public_vits14` |
| `ssl_proposed_vits14` - `dinov2_public_vits14` | auc | -0,0111 | [-0,0289, 0,0050] | no se distingue del ruido |
| `ssl_proposed_vits14` - `dinov2_public_vits14` | ece | -0,0429 | [-0,0596, -0,0216] | gana `ssl_proposed_vits14` |
| `ssl_proposed_vitb14` - `dinov2_public_vitb14` | macro_f1 | -0,0922 | [-0,1473, -0,0420] | gana `dinov2_public_vitb14` |
| `ssl_proposed_vitb14` - `dinov2_public_vitb14` | balanced_accuracy | -0,1150 | [-0,1678, -0,0657] | gana `dinov2_public_vitb14` |
| `ssl_proposed_vitb14` - `dinov2_public_vitb14` | auc | 0,0177 | [0,0065, 0,0306] | gana `ssl_proposed_vitb14` |
| `ssl_proposed_vitb14` - `dinov2_public_vitb14` | ece | -0,1010 | [-0,1177, -0,0758] | gana `ssl_proposed_vitb14` |
| `ssl_plain_vits14` - `dinov2_public_vits14` | macro_f1 | -0,0522 | [-0,1038, -0,0041] | gana `dinov2_public_vits14` |
| `ssl_plain_vits14` - `dinov2_public_vits14` | balanced_accuracy | -0,0837 | [-0,1305, -0,0408] | gana `dinov2_public_vits14` |
| `ssl_plain_vits14` - `dinov2_public_vits14` | auc | -0,0016 | [-0,0181, 0,0147] | no se distingue del ruido |
| `ssl_plain_vits14` - `dinov2_public_vits14` | ece | -0,0604 | [-0,0711, -0,0343] | gana `ssl_plain_vits14` |
| `ssl_plain_vitb14` - `dinov2_public_vitb14` | macro_f1 | -0,0302 | [-0,0753, 0,0161] | no se distingue del ruido |
| `ssl_plain_vitb14` - `dinov2_public_vitb14` | balanced_accuracy | -0,0495 | [-0,0938, -0,0034] | gana `dinov2_public_vitb14` |
| `ssl_plain_vitb14` - `dinov2_public_vitb14` | auc | 0,0190 | [0,0074, 0,0320] | gana `ssl_plain_vitb14` |
| `ssl_plain_vitb14` - `dinov2_public_vitb14` | ece | -0,1146 | [-0,1239, -0,0882] | gana `ssl_plain_vitb14` |

Contraste pareado (los dos brazos remuestrean los MISMOS indices), estratificado por clase. 1.000 resamples, semilla 20260923.
Un intervalo que **incluye el cero** significa que la diferencia no se distingue del ruido, por grande que sea el punto. Ningun 'gana' se cita sin que su intervalo excluya el cero.

_Generado desde `bootstrap.json` por `scripts/build_results_readme.py`._

<!-- TABLA-CONTRASTES:FIN -->

---

## 5. Que gano

Leido sobre la tabla de metricas completas, con el probe fijado **antes** de
tener datos (`HEADLINE_PROBE = "logreg"`, commit `3381cfa`, 2026-09-21, un dia
antes de que cerrara el banco).

- **La calibracion es el resultado mas fuerte del trabajo, y ahora con banda.**
  El ECE de los cuatro brazos adaptados es varias veces menor que el de DINOv2
  publico, y el contraste pareado **excluye el cero en los cuatro, con margen
  amplio** (por ejemplo `ssl_plain_vitb14` vs DINOv2 ViT-B: d=-0,1146, IC
  [-0,1239, -0,0882]; ver la tabla de contrastes). Como los cuatro parten de
  los mismos pesos publicos, la mejora es atribuible a las 12 horas-GPU de
  SimSiam sobre dermatoscopias, no a la arquitectura. Para un clasificador
  medico, una probabilidad que significa lo que dice es una propiedad util por
  si sola.
- **Contra DINOv2 publico, la adaptacion de dominio gana de forma afirmable en
  AUC y en calibracion — en los dos tamanos ViT-B.** `ssl_plain_vitb14` supera
  a DINOv2 ViT-B en AUC (d=+0,0190, IC [+0,0074, +0,0320]) y `ssl_proposed_vitb14`
  tambien (d=+0,0177, IC [+0,0065, +0,0306]). No es "un SSL desde cero le gana a
  DINOv2": es que agregarle 12 horas-GPU de SimSiam de dominio al backbone
  publico lo mejora en esos ejes. En ViT-S el AUC no se distingue del publico.
- **La cadena de entrenamiento es reproducible y resistente a fallas**: un
  cuelgue de NCCL al 95 % se recupero solo, y el resultado cayo dentro de la
  prediccion registrada antes de conocerlo
  ([prediccion-brazo3-interrupcion.md](prediccion-brazo3-interrupcion.md)).
- **La segmentacion sin etiquetas es utilizable** como paso previo: mediana
  de Dice 0,79 y 0,85, y la cuarentena detecta el 6 % de imagenes sin mascara
  confiable en vez de inventar una.

---

## 6. Que perdio

- **El emparejamiento por tono no aporta, y ahora esta afirmado con banda.**
  De los 8 contrastes del propuesto contra su propio control, **cero favorecen
  al propuesto**. Pierde de forma afirmable (IC que excluye el cero) en macro-F1
  y balanced accuracy en los dos tamanos; en AUC y ECE la diferencia **no se
  distingue del ruido** (ViT-B AUC: d=-0,0013, IC [-0,0070, +0,0054]). La
  ganancia de la corrida preliminar (+0,081) **no se replica**: aquella corrida
  tenia el termino de perdida duplicado (`08727e8`) y ademas corrio con el peso
  de consistencia en 1,0, que es la identidad — el brazo "propuesto" entreno con
  la perdida del control sin escalar. Con el defecto corregido y el triple de
  presupuesto, la direccion se invierte.
- **Ni DINOv2 publico ni los adaptados ganan en todas las metricas.** En balanced
  accuracy DINOv2 ViT-B supera a los adaptados de forma afirmable (vs
  `ssl_plain_vitb14`: d=-0,0495, IC [-0,0938, -0,0034]). En macro-F1 la
  diferencia contra `ssl_plain_vitb14` **no se distingue del ruido** (d=-0,0302,
  IC [-0,0753, +0,0161]): no es correcto decir "DINOv2 gana en F1". Los
  adaptados separan mejor (AUC) y calibran mucho mejor, pero recuperan peor las
  clases minoritarias. Es el patron de un clasificador que se apoya en la clase
  mayoritaria (nevo):
  accuracy alta, recall macro bajo.
- **Con `logreg_balanced` el propuesto ViT-B gana en balanced accuracy y
  AUC.** Se reporta porque existe, pero **no se usa como resultado**: ese
  probe no fue el registrado, y elegir el probe despues de ver cual favorece
  al metodo es exactamente la seleccion que invalida una comparacion. Que el
  veredicto cambie con el probe es, en si mismo, evidencia de que la
  diferencia es chica.
- **El modelo de segmentación no supera al piso mas fuerte** en
  solapamiento medio sobre ISIC 2018; gana en borde y en fallos
  catastroficos.

---

## 7. Que no se puede afirmar

- **Las diferencias afirmables son solo las que la tabla de contrastes marca
  con IC que excluye el cero.** El resto son estimaciones puntuales que no se
  distinguen del ruido, y no se citan como resultado por grande que sea el
  numero. La banda viene del bootstrap pareado (1000 resamples, seed 20260923),
  no de las tres semillas: al 100 % de etiquetas las semillas son la misma
  corrida (n efectivo = 1).
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
- **La version de scikit-learn del banco no quedo registrada, pero la
  reproduccion la vuelve inofensiva.** El pliegue de `StratifiedGroupKFold`
  depende de ella; el volcado del bootstrap corrio con scikit-learn 1.9.0 y
  reprodujo `bank_rows.csv` con **sha256 identico byte a byte** al publicado
  (`67d6063...`), asi que ni la version ni el fold se movieron. La identidad del
  conjunto de test queda verificada por esa reproduccion, no por un registro de
  ids que el banco no guardo.
- **Nada sobre fototipos altos reales.** El pool es mayoritariamente piel
  clara. Sin datos externos no hay base para afirmar comportamiento en piel
  oscura en sentido clinico.
- **Nada clinico.** Ninguna de estas cifras se obtuvo en un entorno de uso
  real ni con validacion independiente.

---

## 8. Como mejorarlo

En orden de retorno esperado:

1. **Bootstrap pareado sobre el test** (hecho: ver la seccion 4, "Contrastes con intervalo de confianza"). No mejora el modelo, pero
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
   experimento de segmentación final no exporta predicciones sobre el pool. Exportarlas es
   trabajo mecanico y mejora la entrada de todo lo demas.
5. **Resolucion.** Todo se materializa a 224 para no transformar ISIC 2020 en
   resolucion nativa. Subirla eleva el techo, a costa de espacio y computo.
6. **Validacion externa en pieles de tono oscuro (sin tono validado)**, sin la cual la seccion de
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
