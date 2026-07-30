# 15 — Auditoría del pipeline y plan v5: hacia una evaluación 100 % auto-supervisada

_Fecha: 2026-07-29. Autor: Nicolás Moreno. Destinatarios: Karen Sánchez (asesora, KAUST),
Andrés Jerez (director), equipo del proyecto._

Este documento se escribió para leerse **sin conocer el proyecto ni el área**. Cada término
técnico se explica la primera vez que aparece. Las afirmaciones sobre el código citan el
archivo y la celda concreta, de modo que cualquiera pueda verificarlas.

---

## 0. Resumen en cinco frases

1. El pipeline actual (v4) reporta que el aprendizaje auto-supervisado **supera** al
   pre-entrenamiento supervisado de ImageNet: macro-F1 0.6907 ± 0.006 contra 0.6614 ± 0.004.
2. Auditando el código encontramos que ese pipeline **no es auto-supervisado puro**: la red
   auto-supervisada arranca de pesos entrenados con 1.28 millones de etiquetas humanas, y
   después el ajuste fino vuelve a entrenar toda la red usando etiquetas.
3. Eso no invalida el resultado de v4, pero **cambia lo que ese resultado significa**: mide
   cuánto aporta adaptar al dominio, no auto-supervisado contra supervisado.
4. Se implementó **v5**, una evaluación en la que **ninguna etiqueta toca el extractor de
   características**: red congelada más clasificadores diminutos encima.
5. v5 está construido, verificado con 101 pruebas automáticas que corren sin GPU, y listo para
   ejecutarse en Kaggle con una GPU de estudiante. El pre-entrenamiento continuo sobre imágenes
   dermatoscópicas —la etapa que sí necesita el clúster— es el paso siguiente y consumirá
   exactamente estas mismas características y probes.

---

## 1. Glosario mínimo

| Término | Explicación sencilla |
|---|---|
| **Backbone** (red base) | La red neuronal que convierte una imagen en una lista de números. Esa lista se llama *features* o representación. |
| **Features** | La descripción numérica que la red hace de la imagen. Si son buenas, imágenes parecidas tienen números parecidos. |
| **Aprendizaje supervisado** | Se entrena mostrando pares (imagen, etiqueta correcta). Necesita que un humano haya etiquetado todo. |
| **Aprendizaje auto-supervisado (SSL)** | Se entrena **sin etiquetas**. La red inventa su propia tarea, por ejemplo reconocer que dos recortes distintos vienen de la misma foto. Es el interés del proyecto porque las etiquetas médicas son caras y escasas. |
| **ImageNet** | Base de datos de 1.28 millones de fotos etiquetadas a mano. Casi todo modelo de visión parte de ahí. Usar ImageNet **es** usar supervisión. |
| **DINOv2** | Modelo de Meta entrenado con SSL sobre 142 millones de imágenes **sin ninguna etiqueta**. Es la referencia actual del área. |
| **Congelar el backbone** | Dejarlo intacto, solo leerlo. No se modifica ni un peso. |
| **Ajuste fino** (*fine-tuning*) | Lo contrario: seguir entrenando la red con etiquetas del problema nuevo. |
| **Probe** | Clasificador diminuto que se pone encima de las features congeladas. Mide la calidad de las features, no la del entrenamiento. |
| **k-NN** | El probe más simple posible: para clasificar una imagen, mira a qué imágenes de entrenamiento se parece más y copia su etiqueta. **No entrena nada.** |
| **Régimen de etiquetas** | Cuántas etiquetas le damos al probe: 1 %, 5 %, 10 %... Si el SSL sirve, con el 1 % ya debería funcionar. |
| **Data leakage** (fuga) | Que información de la prueba se filtre al entrenamiento. Infla las métricas sin que el modelo sea mejor. |
| **macro-F1** | Promedio del desempeño por clase, sin dar más peso a las clases frecuentes. Se usa porque el dataset está desbalanceado 58 a 1. |
| **AUC** | Probabilidad de que el modelo puntúe más alto un caso positivo que uno negativo. Mide capacidad de ordenar casos. |
| **ECE** | Error de calibración: si el modelo dice "90 % de confianza", ¿acierta el 90 % de las veces? |

---

## 2. Auditoría: qué hace el pipeline v4 hoy

Archivo: `notebooks/proyecto_ssl_lesiones_v4.ipynb`, 28 celdas, ejecutado en Kaggle con dos
GPU T4. Referencia previa: [05 — Pipeline](05-pipeline-objetivo2.md).

| Etapa | Celdas | Qué hace |
|---|---|---|
| Configuración y carga robusta | 1 | Perillas de cómputo, modo `SMOKE`, tolerancia a imágenes corruptas |
| Roles anti-leakage | 3–5 | HAM10000 etiquetado; ISIC 2020 como pool sin etiquetas; ISIC 2019 para validación externa |
| Pre-entrenamiento SSL | 7–8 | SimSiam sobre ResNet-50, 30 épocas, 224 px, unas 48 000 imágenes sin etiqueta |
| Ajuste fino | 10–13 | LP-FT más *layer-wise LR decay*, ablación con 2 semillas |
| Reporte clínico | 15 | Calibración por temperatura, umbral de tamizaje de melanoma |
| Fiabilidad | 16–27 | Linear probe, intervalos bootstrap, abstención, Grad-CAM, validación externa |

**Lo que está bien hecho y se conserva.** La partición por lesión con verificación por
`assert`, el modo `SMOKE`, el *resume* del backbone, los bloques secundarios no fatales, la
guarda de tiempo que **avisa** cuando recorta semillas. Esa ingeniería es la razón de que las
corridas terminen; v5 la hereda entera.

---

## 3. El hallazgo: v4 no es auto-supervisado puro

En la celda 7, dentro de la clase `SimSiam`:

```python
bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)  # backbone ImageNet
```

La red "auto-supervisada" **arranca de pesos entrenados con 1.28 millones de etiquetas
humanas**. Hay cuatro puntos donde entra supervisión:

1. **Inicialización.** El SSL parte de ImageNet supervisado. Técnicamente es *domain-adaptive
   SSL* (adaptación de dominio), no SSL.
2. **Ambos brazos comparten ancestro.** El brazo "SSL" y el brazo "ImageNet" salen del mismo
   punto. Por eso la diferencia de +0.0294 mide *cuánto aporta seguir entrenando sin etiquetas
   sobre imágenes de piel*, no *auto-supervisado contra supervisado*.
3. **El ajuste fino descongela todo** (`FT_EPOCHS = 15`, celda 11) y lo hace con etiquetas.
   LP-FT y LLRD reducen el daño pero no lo eliminan: el modelo entregable es un modelo
   supervisado que empezó desde features SSL.
4. **La normalización** usa media y desviación de ImageNet. Cosmético, pero es un rastro más.

**Esto ya estaba documentado con honestidad** en la decisión D-02 de
[04 — Decisiones técnicas](04-decisiones-tecnicas.md), y estaba bien justificado: con dos GPU T4
entrenar un ViT desde cero era inviable, y la Práctica 1 mostró empíricamente que partir de cero
pierde. La decisión fue correcta para el cómputo disponible.

**Pero limita lo que la tesis puede afirmar.** Un evaluador que abra la celda 7 lo ve en diez
segundos. Y como ahora habrá acceso a un clúster, esa limitación deja de ser necesaria.

---

## 4. Qué significa "completamente auto-supervisado": tres lecturas

| Lectura | Definición | Veredicto |
|---|---|---|
| **A. Purista** | Ningún peso externo. Entrenar DINO desde inicialización aleatoria solo con nuestras imágenes de piel. | **Va a perder, y se sabe por qué.** DINOv2 se entrenó con 142 millones de imágenes; nuestro pool tiene entre 48 000 y 70 000. La Práctica 1 ya midió este efecto. |
| **B. Sin etiquetas, nunca** (elegida) | Se permite un pre-entrenamiento ajeno de gran escala **siempre que no haya usado etiquetas** (los pesos oficiales de DINOv2), más pre-entrenamiento continuo sobre imágenes de piel, más evaluación con el backbone **congelado**. | **Ninguna etiqueta tocó jamás el extractor.** Afirmación fuerte, verificable y ejecutable. |
| **C. Híbrida** | Como B pero permitiendo ajuste fino. | Deja de ser "backbone auto-supervisado". Es lo que ya tenemos en v4. |

**Decisión: B como línea principal, A como ablación explícita.**

A no es un fracaso: es la evidencia empírica de que la escala de datos manda en SSL. Reportar
que pierde es ciencia, no derrota, y da un capítulo con contenido propio.

La regla dura que hace verificable la lectura B: **si el backbone está congelado, las etiquetas
solo sirven para entrenar el clasificador diminuto y para evaluar.** El notebook v5 lo verifica
con un `assert` que cuenta parámetros entrenables y exige cero.

---

## 5. Los cambios implementados

Todo lo de esta sección está en el repositorio y probado. Nada es una propuesta.

### 5.1 Rutas de datos desacopladas de Kaggle

**Problema.** v4 tenía `INPUT = Path("/kaggle/input")` escrito a mano (celda 3). Ese directorio
solo existe dentro de una sesión de Kaggle, así que el mismo notebook no podía correr en un
clúster universitario.

**Solución.** `src/dermssl/data_resolver.py` con tres backends y una sola interfaz:

- `kaggle-input`: los datasets que Kaggle ya montó. Instantáneo, sin internet.
- `kagglehub`: los descarga con el paquete `kagglehub` usando credenciales. Funciona en
  cualquier máquina. Dentro de Kaggle adjunta el input en vez de descargar, así que el mismo
  código sirve en los dos lados.
- `local`: copias ya guardadas en disco (scratch, NFS).

`auto` elige en ese orden y **reporta cuál usó**. Los datasets no cambian: siguen siendo
HAM10000, ISIC 2019 e ISIC 2020.

**Por qué importa.** Es el cambio que desbloquea todos los demás: sin él, cualquier trabajo en
el clúster obliga a mantener dos notebooks distintos, que inevitablemente se desincronizan.

**Nota de seguridad.** Las credenciales van en variables de entorno (`KAGGLE_USERNAME`,
`KAGGLE_KEY`) o en `~/.kaggle/kaggle.json` con permisos 600. Nunca dentro del notebook ni del
repositorio. Hay una prueba automática que falla si aparece algo parecido a una credencial
dentro del notebook.

### 5.2 Guardarraíl de resolución contra tamaño de parche

**Problema.** Un ViT corta la imagen en cuadraditos llamados *parches*. DINOv2 usa parches de
14 × 14, así que el lado de la imagen debe ser múltiplo exacto de 14. `224 = 16 × 14` sirve,
**pero 256 no** (256 / 14 = 18.28...). Y las imágenes de ISIC 2020 se distribuyen a 256 × 256.

Alimentar 256 a DINOv2 produce una de dos cosas: un error de formas dentro de la atención, o
—peor— una interpolación silenciosa de los *positional embeddings* que degrada las features sin
avisar. Las dos desperdician tiempo de clúster.

**Solución.** `src/dermssl/patch_guard.py`. Detecta el tamaño de parche del backbone, valida la
resolución, y cuando no encaja la ajusta al múltiplo válido más cercano **imprimiendo el
cambio** para que nunca sea invisible. El notebook imprime, como control, la línea
`256 con parche 14 -> 256 -> 252 (snapped down...)`.

### 5.3 Regímenes de pocas etiquetas, anidados y sin clases vacías

**Problema doble.** v4 medía desde el 10 % de etiquetas, y el rango donde el SSL suele ganar por
más margen es el 1–5 %. Además, tomar el 1 % de forma ingenua puede dejar en cero a la clase
`df` (dermatofibroma, unas 80 imágenes de entrenamiento), y un clasificador que nunca vio una
clase no puede predecirla: el macro-F1 se hunde por una razón que **no tiene nada que ver con
el backbone**.

**Solución.** `src/dermssl/label_regimes.py` garantiza dos propiedades:

- **Ninguna clase se queda sin muestras** (`min_per_class`).
- **Los regímenes están anidados**: el subconjunto del 1 % está contenido en el del 5 %, que
  está en el del 10 %, y así. Se logra mezclando cada clase una vez por semilla y tomando
  prefijos. Consecuencia: entre dos regímenes lo único que cambia son etiquetas **añadidas**,
  así que la curva mide eficiencia de etiquetas y no ruido de muestreo.

El notebook imprime una auditoría de cada régimen con el conteo de la clase más pequeña, para
que se pueda confirmar de un vistazo que nadie quedó en cero.

### 5.4 Métricas propias, verificadas a mano

`src/dermssl/metrics.py` implementa macro-F1, AUC (una contra el resto), ECE, matriz de
confusión, recall por clase e intervalos bootstrap **solo con numpy**.

**Por qué no scikit-learn.** Dos razones concretas. Primero, permite probar la lógica en
cualquier máquina sin dependencias, que es lo que hace posible el suite de pruebas. Segundo, la
definición exacta de cada número reportado queda visible en el repositorio en vez de depender de
una versión de una librería.

Detalle que suele estar mal en implementaciones caseras y que aquí está cubierto por una prueba:
el **AUC con empates**. Si un modelo asigna la misma probabilidad a muchos casos, el ranking
ingenuo da un AUC optimista falso. La implementación usa rangos promediados, y hay una prueba que
verifica que con todos los puntajes empatados el AUC es exactamente 0.5.

### 5.5 Probes sobre backbone congelado

`src/dermssl/probes.py`, en orden creciente de parámetros entrenados:

| Probe | Parámetros entrenados | Qué mide |
|---|---|---|
| `knn` | **cero** | La geometría pura de las features. Nada se aprende, así que el número refleja solo la calidad de la representación. Es el protocolo con que se evalúa DINO/DINOv2. |
| `prototype` | **cero** | Un vector promedio por clase, se clasifica por el más cercano. Muy robusto con pocas etiquetas, justo el régimen del 1 %. |
| `logreg` | una capa lineal | El *linear probe* estándar de la literatura. |

El k-NN usa similitud coseno con voto ponderado por temperatura (0.07, el valor del paper de
DINO) y **recorta k al número de muestras disponibles**, para que el régimen del 1 % no reviente.

### 5.6 Caché de features: lo que vuelve la rejilla de experimentos gratis

**Problema.** Pasar 10 000 imágenes por un ViT cuesta minutos de GPU. Ajustar un probe sobre los
vectores resultantes cuesta milisegundos. v4 volvía a extraer en cada variación.

**Solución.** `src/dermssl/feature_cache.py`: se extrae **una vez**, se guarda a disco, y después
se prueban todos los probes, regímenes y semillas que se quieran, gratis. Eso convierte
"3 backbones × 3 probes × 6 regímenes × 3 semillas" (162 experimentos) en minutos en vez de días,
y permite subir de 2 a 5 semillas sin discusión de presupuesto.

**La propiedad de seguridad que importa.** Un caché nunca debe devolver features de otra
configuración. La clave incluye una huella criptográfica del **orden exacto** de los
identificadores de imagen, porque las features se guardan alineadas a ese orden. Si cambia la
partición, cambia la clave, y el caché viejo se ignora en vez de contaminar en silencio todos
los números de más abajo. Hay pruebas para cada forma de invalidación: partición distinta,
resolución distinta, backbone distinto, metadatos corruptos, versión de esquema vieja.

Efecto secundario útil: si la sesión de Kaggle muere después de extraer, al reiniciar reutiliza
lo extraído.

### 5.7 El control que hace justa la comparación

Los tres backbones de la corrida por defecto:

| Clave | Qué es | Etiquetas usadas para crearlo |
|---|---|---|
| `ssl_dinov2_vits14` | DINOv2 ViT-S/14 | **ninguna** (142 M imágenes, auto-supervisado) |
| `sup_vit_b16` | ViT-B/16 de torchvision | 1 281 167 (ImageNet) |
| `sup_resnet50` | ResNet-50 de torchvision | 1 281 167 (ImageNet), línea base de v4 |

**`sup_vit_b16` no es opcional.** Si comparamos DINOv2 (un ViT) contra ResNet-50 y DINOv2 gana,
no sabemos si ganó *el método auto-supervisado* o simplemente *la arquitectura ViT*. Cambiar dos
variables a la vez y atribuir el resultado a una es el error metodológico más común en este área.
`sup_vit_b16` es un ViT supervisado: aísla la arquitectura y deja la supervisión como única
diferencia. Sin este control la conclusión no se sostiene ante un jurado.

El registro admite además `ssl_simsiam_r50`, que carga el backbone de v4 si se adjunta, para
mantener continuidad con lo ya hecho.

### 5.8 Notebook generado, no editado a mano

**Problema.** Una sesión de Kaggle no puede importar `src/dermssl` de este repositorio, así que
el notebook tiene que ser autocontenido. La salida fácil es copiar la lógica dentro del
notebook, y a partir de ese momento la copia probada y la copia que corre se separan.

**Solución.** El notebook se **genera**: `scripts/build_notebook_v5.py` inserta el núcleo numpy
tal cual está en `src/dermssl/`, y toma las celdas que dependen de torch de
`scripts/notebook_cells_v5.py`. Resultado: **el código que corre en Kaggle es exactamente el
código cubierto por las pruebas**. El notebook es un artefacto de compilación; para cambiar algo
se edita el módulo, se corre `pytest`, y se regenera.

```bash
python scripts/build_notebook_v5.py            # generar
python scripts/build_notebook_v5.py --check     # falla si el notebook está desactualizado
```

El generador **compila cada celda de código** antes de escribir el archivo. Es la prueba más
barata posible de que Kaggle no va a morir con un error de sintaxis después de cuarenta minutos
en la cola.

---

## 6. Cómo se verificó

### 6.1 Pruebas automáticas

```
$ python -m pytest tests/ -q
101 passed, 1 skipped in 0.75s
```

Corren en menos de un segundo, **sin GPU, sin sesión de Kaggle y sin internet**. El único
`skipped` es el linear probe, que necesita scikit-learn y sí está presente en Kaggle.

| Archivo | Qué prueba |
|---|---|
| `test_patch_guard.py` | Que 224 se acepte, que **256 se rechace con un mensaje que propone 252 y 266**, y que el ajuste automático nunca aumente la memoria por accidente |
| `test_label_regimes.py` | Que la clase de 8 muestras sobreviva al régimen del 1 %, que los regímenes anidan, que la misma semilla es reproducible y otra semilla da algo distinto |
| `test_metrics.py` | Cada métrica contra un valor **calculado a mano**: macro-F1 = (2/3 + 0.8)/2, AUC con empates = 0.5, AUC parcial = 0.75, clases indefinidas omitidas y no contadas como 0.5 |
| `test_probes.py` | Que features separables se clasifiquen perfecto, que **ruido puro quede en azar** (si diera bien, habría fuga de etiquetas), que el troceado por bloques dé idéntico resultado que un solo pase |
| `test_feature_cache.py` | Cada forma de invalidación del caché, incluido metadato corrupto y partición distinta, y que `get_or_extract` calcule una sola vez |
| `test_pipeline_integration.py` | **La rejilla completa** replicada sobre features sintéticas cuya respuesta correcta se conoce: un backbone estructurado supera 0.90 de AUC, uno inútil queda bajo 0.65, y las compuertas de aceptación pasan para el primero y fallan para el segundo |
| `test_notebook_build.py` | Que todas las celdas compilen, que **el núcleo se ejecute de verdad** y exponga las diez funciones que las celdas de torch invocan por nombre, que el notebook comiteado no esté desactualizado, que venga sin salidas guardadas, que `SMOKE` esté en `True` por defecto, y que no haya credenciales dentro |

Dos detalles sobre la calidad de estas pruebas:

- **Hay controles negativos.** `test_validate_detects_a_broken_cell` verifica que el validador
  *pueda* fallar; `test_pure_noise_is_near_chance` verifica que los probes *puedan* dar mal. Una
  prueba que no puede fallar no prueba nada.
- **Las pruebas encontraron dos errores reales durante el desarrollo.** Primero, un desajuste de
  claves en el cargador de DINOv2 (usaba el nombre del parche para indexar un diccionario
  indexado por clave de backbone). Segundo, una interacción entre `from __future__ import
  annotations`, `@dataclass` y la ejecución en un namespace: las anotaciones pasan a ser cadenas
  y `dataclass` intenta resolverlas buscando el módulo en `sys.modules`, que no existe al
  ejecutar una celda. Se resolvió eliminando ese import del notebook generado, que además no
  hace falta: `Path | None` es sintaxis nativa desde Python 3.10.

### 6.2 Lo que las pruebas locales NO cubren

Honestidad explícita: **la extracción de imagen a features no está probada localmente**, porque
requiere torch, GPU y los datasets reales. Eso es exactamente lo que la corrida en Kaggle
verifica. Todo lo demás —resolución de rutas, guardarraíles, muestreo, métricas, probes, caché,
agregación y compuertas— ya está verificado antes de gastar un solo minuto de GPU.

### 6.3 Compuertas de aceptación de la corrida

El notebook termina imprimiendo un reporte explícito. No hay "listo" sin evidencia.

| Compuerta | Criterio |
|---|---|
| G0 resolución de datos | Backend resuelto y HAM10000 localizado |
| G0 anti-leakage | **Cero** lesiones compartidas entre entrenamiento y validación |
| G1 pipeline end-to-end | Toda la rejilla corre sin fallos |
| G1 backbone congelado | Cero parámetros entrenables en todo backbone usado (verificado por `assert`) |
| G2 calidad SSL | AUC del probe k-NN por encima de 0.80 **sin entrenar nada** |
| G2 control de arquitectura | Hay un ViT supervisado en la comparación |

---

## 7. Cómo correr v5 en Kaggle

1. Settings: Accelerator **GPU T4 x2**, Internet **ON** (la primera vez, para descargar los
   pesos de DINOv2).
2. Add Input: `Skin Cancer MNIST: HAM10000` (kmader). Opcionales: ISIC 2019 y 2020.
3. `SMOKE = True` (valor por defecto) y **Run All**. Debería terminar en unos cinco minutos.
   Si llega al final sin error y el reporte de compuertas sale en verde, el pipeline sirve.
4. `SMOKE = False` y **Save & Run All (Commit)** para la corrida real.

Salidas: `RESULTS_SUMMARY_v5.json` (todos los números crudos, para regenerar figuras sin volver a
correr nada) y `probe_grid_v5.csv`.

**v5 no necesita el clúster.** ViT-S/14 congelado entra en una T4 de sobra, porque no se entrena.
Es una propiedad deliberada: el resultado auto-supervisado más limpio se puede obtener con
recursos de estudiante, y el clúster se reserva para la etapa que de verdad lo necesita.

---

## 8. Traer los datasets desde fuera de Kaggle

Verificado contra la documentación de `kagglehub`.

```python
import os, kagglehub
os.environ["KAGGLE_USERNAME"] = "..."   # o ~/.kaggle/kaggle.json con permisos 600
os.environ["KAGGLE_KEY"]      = "..."

path = kagglehub.dataset_download(
    "kmader/skin-cancer-mnist-ham10000",
    output_dir="/scratch/derm/ham10000",   # sin este argumento cachea en ~/.cache/kagglehub
)
```

En el notebook basta cambiar `DATA_BACKEND = "kagglehub"` (o `"local"` con `LOCAL_ROOT`). Nada
más cambia.

**Patrón correcto en un clúster** (no descargar en cada trabajo):

1. Un trabajo `stage-data` descarga **una sola vez** a almacenamiento compartido y escribe un
   manifiesto.
2. Los trabajos de entrenamiento leen de la ruta local.
3. Si los nodos de cómputo no tienen salida a internet —habitual en clústeres universitarios—
   se descarga desde el nodo de login y se sincroniza con `rsync`.
4. Con TPU conviene empaquetar a *shards* tipo WebDataset (archivos `.tar` de aproximadamente
   1 GB) en ese mismo paso: 48 000 JPEG pequeños generan demasiadas operaciones de disco.

**Espacio a solicitar.** HAM10000 unos 2.9 GB, ISIC 2019 a 224 px unos 3 GB, ISIC 2020 a 256 px
unos 5 GB, más checkpoints de ViT-B de unos 350 MB cada uno. Pedir entre 30 y 50 GB de scratch.

**Pendiente de seguridad, previo a cualquier uso del clúster.** El token de Kaggle marcado como
expuesto en [08 — Estado actual](08-estado-actual-y-next-steps.md) debe **rotarse antes** de
colocarlo en una máquina de la universidad.

---

## 9. Comparación de hardware

Órdenes de magnitud; los valores exactos deben confirmarse contra la documentación del clúster.

| Hardware | Memoria | bf16 nativo | Qué habilita |
|---|---|---|---|
| T4 x2 (Kaggle) | 16 GB cada una | No (arquitectura Turing) | v5 completo (backbones congelados). Pre-entrenar un ViT-B con multi-crop no entra. Límite de 12 h por sesión |
| A100 40/80 GB | 40 u 80 GB | Sí | Pre-entrenamiento continuo de DINOv2 ViT-B/14 con multi-crop real |
| H100 80 GB | 80 GB | Sí | ViT-L viable |
| TPU v3-8 | unos 16 GB por núcleo, 8 núcleos | Sí | Mucha memoria agregada, con la salvedad de abajo |
| TPU v4-8 / v5e | unos 32 / 16 GB por chip | Sí | Ídem |

### El costo real de las TPU

Pasar a TPU **no es cambiar `device="cuda"` por `"tpu"`**. Concretamente:

- La implementación oficial de DINOv2 usa **xFormers**, que es exclusivo de CUDA. En TPU hay que
  arrancarla con atención estándar (más memoria, más lento) o reimplementarla.
- Con PyTorch se usa `torch_xla`, que exige **formas fijas**: cualquier forma dinámica recompila
  el grafo en cada paso y el rendimiento se cae. El depurado es bastante más hostil.
- Con JAX/Flax la experiencia en TPU es más limpia, pero implica reescribir el pipeline.

Estimación honesta: entre una y tres semanas de trabajo de adaptación, con riesgo de que la
convergencia no sea equivalente. **No verificamos esto contra la documentación de `torch_xla`**;
hay que confirmarlo en la máquina real antes de comprometerse.

**Información relevante que ya tenemos.** En el laboratorio de la asesora ya se entrena con
DINOv2, lo que implica que el entorno CUDA con xFormers existe y converge allí. Eso reduce
mucho el riesgo: la línea principal va en GPU, sobre un entorno probado.

**Reparto recomendado.**

- **GPU**: el pre-entrenamiento continuo de DINOv2 sobre el pool dermatoscópico. Es la etapa
  costosa y la que ya está probada allí.
- **TPU**: los barridos masivamente paralelos de probes sobre features **ya extraídas**. Eso es
  álgebra sobre archivos `.npy`, no ViT: se traslada sin riesgo.
- **Kaggle y máquinas locales**: desarrollo y compuertas G0 a G2.

### Protocolo de comparación a igual cómputo

La literatura compara "N épocas contra N épocas". Nosotros mediremos a **igual presupuesto de
GPU-hora**: macro-F1 y AUC como *función del cómputo gastado*. El notebook ya registra, por
backbone, el tiempo de reloj y las imágenes por segundo de la extracción, precisamente para
alimentar esa curva.

---

## 10. Configuraciones distintas a lo que hay en la literatura

Ordenadas por relación aporte/costo. La recomendación es elegir las tres primeras, que cuentan
una sola historia coherente: *auto-supervisión con invarianzas propias de la dermatoscopía,
medida a igual cómputo*.

1. **Curvas a igual cómputo.** Aporte metodológico. Responde "a partir de cuánto cómputo conviene
   auto-supervisar en vez de usar ImageNet", una pregunta práctica que casi nadie contesta en
   dermatología. Costo bajo; la instrumentación ya está.

2. **Constancia de color como invarianza, no como preproceso.** Aplicar *shades-of-gray* o
   *gray-world* de forma **estocástica dentro del par de vistas** del SSL, en vez de como
   normalización fija previa. Así se obliga al codificador a ser invariante a la iluminación del
   dermatoscopio. La literatura lo usa siempre como preproceso determinista. Costo casi nulo (una
   transformación), y la motivación clínica es directa: la variabilidad de dispositivo e
   iluminación entre centros **es** el problema de generalización del proyecto.

3. **Multi-crop guiado por la lesión.** DINO usa 2 recortes globales y 8 locales **aleatorios**.
   En dermatoscopía la lesión está centrada y el **borde** es la señal diagnóstica (regla ABCD).
   Política propuesta: globales centrados en la lesión, locales sesgados al **anillo del borde**,
   obtenido con umbralado de Otsu o saliencia, **sin usar etiquetas**. Costo bajo o medio, y muy
   defendible clínicamente.

4. **Complementariedad de objetivos.** DINO (global, semántico) junto a MAE (local, textura),
   midiendo si sus errores están **descorrelacionados** y concatenando features. La pregunta no
   es cuál gana, sino cuánto se complementan. Costo medio.

5. **Predicción selectiva sobre features auto-supervisadas congeladas.** Curva de riesgo contra
   cobertura y ECE, comparando auto-supervisado y supervisado. La literatura de SSL en
   dermatología reporta AUC y F1, casi nunca calibración ni abstención. El proyecto **ya tiene
   ese stack construido** en v4, así que aplicarlo como eje comparativo es aporte casi gratis.

6. **Equidad por fototipo con backbone congelado.** Pendiente por disponibilidad de datos
   regionales; queda como diseño.

---

## 11. Qué NO está hecho todavía

1. **El pre-entrenamiento continuo de DINOv2 sobre el pool dermatoscópico.** Es la etapa que
   necesita el clúster. v5 prepara todo lo que la rodea (datos, guardarraíles, features, probes,
   compuertas) para que esa etapa sea un cambio de backbone, no un proyecto nuevo.
2. **Validación externa cruzada.** Sigue siendo el pendiente número uno del proyecto, heredado de
   v4. Con backbone congelado es donde el SSL debería mostrar su mayor ventaja.
3. **La ablación purista (lectura A).** DINO desde inicialización aleatoria sobre el pool de piel.
4. **Grad-CAM para ViT.** La implementación de v4 es específica de ResNet. Para un ViT
   corresponde *attention rollout* o Grad-CAM sobre el último bloque con reordenamiento de
   parches.
5. **La extracción de features en hardware real**, que es lo que la corrida de Kaggle verifica.

**Riesgo abierto que conviene anticipar.** El pre-entrenamiento continuo **puede degradar** las
features de DINOv2 por olvido catastrófico si la tasa de aprendizaje es alta. Mitigación: tasa
baja, *teacher* con media móvil exponencial, y monitorear el probe k-NN cada cierto número de
épocas para quedarse con el mejor checkpoint en vez del último. Si el efecto aparece, es un
hallazgo que vale la pena documentar, no un fallo que esconder.

---

## 12. Qué necesitamos del laboratorio de la asesora

Cinco datos concretos, ordenados por impacto:

1. **El repositorio o *fork* de DINOv2 que usan, con el archivo de configuración de una corrida
   real.** Ahí están resueltos tamaño de lote, multi-crop, planificación de la tasa de
   aprendizaje y calentamiento del *teacher*: semanas de ajuste que no habría que repetir.
2. **Cómo se lanza un trabajo**: gestor de colas, partición, límite de tiempo de pared, GPU por
   nodo y memoria de GPU.
3. **El entorno exacto**: versiones de PyTorch, CUDA y xFormers, y si existe una imagen de
   contenedor. Reproducirlo, no aproximarlo.
4. **Si los nodos de cómputo tienen salida a internet**, y cuánto espacio de scratch se asigna.
   Determina si `kagglehub` corre dentro del trabajo o hace falta el patrón de descarga previa.
5. **Detalles de las TPU**: `torch_xla` o JAX, y qué generación.

---

## 13. Estructura de los archivos nuevos

```
src/dermssl/                      # núcleo probado (solo numpy + biblioteca estándar)
├── data_resolver.py              # rutas de datos: Kaggle / kagglehub / local
├── patch_guard.py                # resolución contra tamaño de parche (la trampa del 256)
├── label_regimes.py              # regímenes de pocas etiquetas, anidados
├── metrics.py                    # macro-F1, AUC, ECE, bootstrap
├── probes.py                     # k-NN, prototipos, linear probe
└── feature_cache.py              # caché en disco con invalidación segura

tests/                            # 101 pruebas, sin GPU, sin internet, menos de 1 s
scripts/
├── notebook_cells_v5.py          # celdas que dependen de torch
└── build_notebook_v5.py          # generador del notebook (compila cada celda)
notebooks/
└── proyecto_ssl_lesiones_v5.ipynb  # artefacto generado: NO editar a mano
```

---

## 14. Referencias

- Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision.*
- Caron, M. et al. (2021). *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV.
- Kumar, A. et al. (2022). *Fine-tuning can distort pretrained features and underperform
  out-of-distribution.* ICLR. arXiv:2202.10054.
- Chen, X. y He, K. (2021). *Exploring Simple Siamese Representation Learning (SimSiam).* CVPR.
- Documentación de `kagglehub`: descarga programática de datasets.
- Documentos internos: [04 — Decisiones técnicas](04-decisiones-tecnicas.md),
  [05 — Pipeline](05-pipeline-objetivo2.md),
  [11 — Resultados finales](11-resultados-finales.md),
  [14 — Mejoras del pipeline](14-mejoras-pipeline.md).
