"""Markdown cells for the v5 notebook: the prose a first-time reader needs.

Kept apart from the executable cells so each file stays reviewable, and so
editing an explanation can never break a code cell by accident.
"""

HEADER_MD = r'''# Objetivo 2 - v5: evaluacion 100 % auto-supervisada (backbone congelado)

**Proyecto de Grado UNAB - generado por `scripts/build_notebook_v5.py`. No editar a mano.**

## Que cambia respecto a v4, y por que

En v4 el "SSL" arrancaba de pesos de ImageNet (celda 7: `resnet50(weights=IMAGENET1K_V2)`)
y despues el fine-tune descongelaba todo el backbone usando etiquetas. Es un pipeline
valido, pero **no** es auto-supervisado puro: hubo 1.28 millones de etiquetas humanas en
el camino.

v5 cambia la regla del juego: **ninguna etiqueta toca el extractor de caracteristicas.**
El backbone se congela y solo se ajusta un clasificador diminuto encima. Asi la pregunta
"cuanto sabe el backbone" se responde sin contaminacion.

| | v4 | v5 |
|---|---|---|
| Init del SSL | ImageNet supervisado | DINOv2 (entrenado **sin etiquetas**) |
| Backbone en evaluacion | se re-entrena (15 epocas) | **congelado** |
| Probes | linear | **kNN + prototipos + linear** |
| Regimenes de etiquetas | 10/20/50/100 % | **1/5**/10/20/50/100 % |
| Rutas de datos | `/kaggle/input` fijo | 3 backends (Kaggle / kagglehub / local) |
| Costo de repetir | re-entrenar todo | features cacheadas -> segundos |

## Como leerlo si no sabes nada de esto

- **Backbone**: la red que convierte una imagen en una lista de numeros ("features").
- **Congelado**: no se modifica. Solo se lee.
- **Probe**: clasificador minusculo encima de esas features. Mide la calidad de las
  features, no la del entrenamiento.
- **kNN**: clasifica una imagen mirando a que imagenes de entrenamiento se parece mas.
  **No entrena nada**: es la medida mas limpia posible.
- **Regimen de etiquetas**: cuantas etiquetas le damos al probe. Si el SSL sirve, con el
  1 % ya deberia andar bien.

## Como correrlo

1. Settings -> Accelerator **GPU T4 x2**; Internet **ON** (la primera vez, para bajar
   los pesos de DINOv2).
2. Add Input: `Skin Cancer MNIST: HAM10000` (kmader). Opcional: ISIC 2019 / 2020.
3. `SMOKE = True` -> Run All (~5 min). Valida que el pipeline entero corre.
4. `SMOKE = False` -> **Save & Run All (Commit)** para la corrida real.

El notebook nunca entrena el backbone, asi que **no necesita las GPUs de la universidad**:
ViT-S/14 congelado entra en una T4 de sobra. El pre-entrenamiento continuo sobre el pool
dermatoscopico (la etapa que si necesita el cluster) es el paso siguiente, y consumira
exactamente estas mismas features y probes.'''

DATA_MD = r'''## 1. Localizar los datos (el cambio que permite correr fuera de Kaggle)

v4 tenia `/kaggle/input` escrito a mano, asi que solo funcionaba dentro de Kaggle.
`resolve_datasets()` prueba tres estrategias y avisa cual uso:

- **kaggle-input**: los datasets que Kaggle ya monto. Instantaneo, sin internet.
- **kagglehub**: los baja con el paquete `kagglehub` usando tus credenciales. Sirve en
  cualquier maquina, incluido el cluster de la universidad.
- **local**: copias ya guardadas en un disco (scratch, NFS).

El mismo notebook, sin editar una linea, corre en los tres sitios.

> **Seguridad**: las credenciales van en variables de entorno (`KAGGLE_USERNAME`,
> `KAGGLE_KEY`) o en `~/.kaggle/kaggle.json` con permisos 600. **Nunca** dentro del
> notebook ni en el repositorio.'''

SPLIT_MD = r'''## 2. Particion honesta por lesion (anti-leakage)

HAM10000 contiene **varias fotos de la misma lesion**. Si esas fotos se reparten entre
entrenamiento y validacion, el modelo ya vio la respuesta y la metrica sube sin que el
modelo haya mejorado: eso es *data leakage*. Fue el pecado de la v1 (0.873 falso).

La particion se hace por `lesion_id` con `StratifiedGroupKFold`, y un `assert` verifica
**cero** lesiones compartidas. Si ese assert falla, el notebook se detiene: preferimos
parar que publicar un numero inflado.'''

BACKBONE_MD = r'''## 3. Backbones congelados (y el control que hace justa la comparacion)

Tres modelos, ninguno entrenado aqui:

| Clave | Que es | Etiquetas usadas para crearlo |
|---|---|---|
| `ssl_dinov2_vits14` | DINOv2 ViT-S/14 | **ninguna** (LVD-142M, auto-supervisado) |
| `sup_vit_b16` | ViT-B/16 de torchvision | 1.28 M (ImageNet) |
| `sup_resnet50` | ResNet-50 de torchvision | 1.28 M (ImageNet) |

**Por que `sup_vit_b16` es obligatorio.** Si comparamos DINOv2 (un ViT) contra ResNet-50 y
DINOv2 gana, no sabemos si gano *el metodo auto-supervisado* o simplemente *la
arquitectura ViT*. `sup_vit_b16` es un ViT supervisado: aisla esa variable. Sin este
control la conclusion no se sostiene ante un jurado.

**La trampa del parche 14.** Un ViT corta la imagen en cuadraditos. DINOv2 usa 14x14, asi
que el lado de la imagen debe ser multiplo de 14. `224 = 16 x 14` sirve, pero **256 no**
(256/14 = 18.28...). Las imagenes de ISIC 2020 vienen a 256. `resolve_resolution()`
detecta el caso y ajusta, imprimiendo el cambio para que nunca sea invisible.'''

EXTRACT_MD = r'''## 4. Extraer features una sola vez y cachearlas

Pasar 10 000 imagenes por un ViT cuesta minutos de GPU. Ajustar un probe sobre los
vectores resultantes cuesta milisegundos. Entonces: se extrae **una vez**, se guarda a
disco, y despues se prueban todos los probes, regimenes y semillas que queramos, gratis.

Eso convierte "3 backbones x 3 probes x 6 regimenes x 3 semillas" (162 experimentos) en
minutos en vez de dias. Y hace el notebook resistente a caidas: si la sesion muere
despues de extraer, al reiniciar reusa lo cacheado.

La clave del cache incluye una huella del **orden exacto** de los ids. Si cambia la
particion, cambia la clave, y el cache viejo se ignora en vez de contaminar los
resultados en silencio.'''

PROBE_MD = r'''## 5. La rejilla de probes (aca sale el resultado)

Para cada backbone, cada probe, cada regimen de etiquetas y cada semilla:

1. Se toma el subconjunto de etiquetas del regimen (anidado: el 1 % esta contenido en el
   5 %, que esta contenido en el 10 %... asi la curva mide eficiencia de etiquetas y no
   ruido de muestreo).
2. Se ajusta el probe sobre las features de entrenamiento **ya extraidas**.
3. Se evalua en validacion: macro-F1, AUC, exactitud, ECE.

Tambien se cronometra todo. Esos tiempos alimentan el capitulo de hardware: permiten
comparar configuraciones **a igual presupuesto de computo** en vez de a igual numero de
epocas, que es la comparacion que la literatura casi nunca hace.'''

REPORT_MD = r'''## 6. Resultados, tablas y el reporte de compuertas

Tres salidas:

1. **Tabla por regimen** - macro-F1 y AUC media +/- desviacion entre semillas. Es la tabla
   que va al informe.
2. **`RESULTS_SUMMARY_v5.json`** - todos los numeros crudos, para regenerar figuras sin
   volver a correr nada.
3. **Reporte de compuertas G0-G2** - criterios de aceptacion explicitos. Si una compuerta
   falla, se dice cual y por que; nada de "listo" sin evidencia.'''
