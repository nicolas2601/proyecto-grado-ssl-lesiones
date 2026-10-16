# 23. Modelo descargable: que se publica, como se descarga y como se prueba

## Nomenclatura de lectura

**Proposed sin segmentación** clasifica sin usar una máscara de lesión. **Proposed con segmentación** combina imagen completa, recorte y descriptores de lesión; es una variante experimental, sin mejora robusta demostrada. El Modelo de control no es una tercera propuesta ni la versión segmentada.

Los experimentos de segmentación se identifican por arquitectura, origen de pseudo-máscaras y ronda de aprendizaje. Las rondas no son arquitecturas distintas; los identificadores antiguos se conservan solo donde permiten localizar archivos o reproducir comandos.

Ver [nombres, equivalencias y explicación para lectores técnicos y no técnicos](10-glosario-eli5.md). Los registros históricos conservan sus cifras, predicciones y limitaciones.

Este documento es el contrato de entrega del modelo. Describe que archivos se
publican, que contiene cada uno, como cargarlos y como reproducir las cifras
del informe. Todo lo que aparece aqui sale de artefactos versionados o de
corridas con su commit citado.

## 1. Que se publica

**La lista exacta de archivos, con su arquitectura, presupuesto y semilla,
esta en las notas del release**, y no se repite aqui a proposito: esas notas
se generan desde los `run.json` publicados (`scripts/build_release_notes.py`),
asi que no pueden describir un modelo que el release no contiene. Una tabla
copiada a mano en este documento si podria, y sobreviviria a la corrida que
la dejo falsa.

Lo que si es estable es la convencion de nombres:

| Patron | Que es |
|---|---|
| `ssl_proposed_<arquitectura>_backbone.pt` | Backbone adaptado al dominio **con** pares de tono. El metodo propuesto. |
| `ssl_plain_<arquitectura>_backbone.pt` | El mismo backbone adaptado **sin** pares de tono. El control con el que se compara, al mismo presupuesto. |
| `<arm>_run.json` | La evidencia de esa corrida: arquitectura, modo, presupuesto en horas-GPU, semilla, hash de particion. |
| `student_r3/best.pt` | U-Net de segmentacion sin mascaras ni etiquetas (camino B). |
| `fst_pseudo_manifest.csv` | Por imagen del pool: bucket de la estimación ITA no validada (no es FST), tercil de luminancia, medianas L*/b*, pixeles de piel. |
| `bank.json`, `strata.json`, `bank.md` | Resultados del banco congelado y su desglose por terciles de luminancia. |

> **Nota: estimación ITA no validada (no es FST).** Lo que aquí aparece como «FST», «fototipo» o «bucket Fitzpatrick» por imagen es el ITA con los umbrales fijos de Groh (`src/dermssl/fitzpatrick.py`), y no está validado. Contra FST asignado por expertos acierta 28,5–30,8 % frente a 32,9 % de la baseline trivial en Fitzpatrick17k-C (11.098 imágenes, docs/20-validacion-externa-fst-fitzpatrick17k.md); en PAD-UFES-20, el ITA con umbrales fijos acierta 22,9 % frente a 51,3 % de la baseline de mayoría (Alipour et al., *Skin Research and Technology* 2026, PMC13459198; FST asignado por dermatólogo según ese artículo). Los terciles `tone_tercile` son terciles de luminancia (mediana de L\* de la piel), no tono de piel.

Cada peso viaja con su `run.json` y ambos llevan el nombre del brazo. Es
deliberado: un peso emparejado con el manifiesto de otro brazo pasa todos los
checksums —el SHA-256 verifica que el archivo llego integro, no que sea el
archivo correcto— y publicaria una mentira reproducible.

Los `.pt` de los backbones contienen **solo el `state_dict` del backbone**,
sin cabezas de proyeccion ni optimizador: es lo que se necesita para extraer
caracteristicas y nada mas. Los checkpoints intermedios por fraccion de
presupuesto (`*_ckpt_c0*.pt`) quedan en el pod y no se publican: son unos
20 por brazo, mas el `resume.pt`, y ninguno entra al entregable.

## 2. Como cargarlo

El backbone se instancia desde `torch.hub` sin pesos y se le carga el
`state_dict`; es el mismo camino que usa `scripts/cluster_frozen_bank.py`.

**La arquitectura no se adivina: se lee del `run.json` que viene con el peso.**
Se publica mas de una, y cargar un checkpoint ViT-B sobre un ViT-S instanciado
a mano falla por forma, o peor, carga a medias si alguien le pone
`strict=False`.

```python
import json
import torch

with open("ssl_proposed_dinov2_vitb14_run.json", encoding="utf-8") as handle:
    arquitectura = json.load(handle)["backbone"]      # p. ej. "dinov2_vitb14"

backbone = torch.hub.load("facebookresearch/dinov2", arquitectura, pretrained=False)
state = torch.load("ssl_proposed_dinov2_vitb14_backbone.pt", map_location="cpu", weights_only=True)
backbone.load_state_dict(state)                        # strict: si no encaja, que falle
backbone.eval()

for parameter in backbone.parameters():   # congelado: es un extractor
    parameter.requires_grad_(False)
```

La resolucion de entrada debe ser multiplo del parche del modelo (14 en todas
las variantes publicadas, que son `/14`): 224 es la usada en todas las
corridas. El preprocesamiento es el mismo del banco:
`Resize(int(224 * 1.14))` bilineal, `CenterCrop(224)`, `ToTensor` y
normalizacion ImageNet.

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

with torch.no_grad():
    features = backbone(preprocess(image).unsqueeze(0))

print(features.shape[-1])   # la dimension depende de la arquitectura
```

El backbone NO clasifica: devuelve un vector de caracteristicas cuya
dimension depende de la arquitectura, y por eso se lee de la salida en vez de
escribirse fija. Para diagnosticar hace falta entrenar encima un probe
lineal, que es exactamente lo que mide el banco
(`scripts/cluster_frozen_bank.py` obtiene la dimension de la misma forma).

## 3. Como reproducir las cifras

Con el repositorio y los datos:

El banco evalua **cada brazo adaptado contra los pesos publicos sin adaptar
de su propio tamano**. Medir un ViT-S adaptado contra un ViT-B publico
confunde la adaptacion con la capacidad: una diferencia asi no dice si ganó
el metodo o si ganó el modelo mas grande.

```bash
python scripts/cluster_frozen_bank.py \
    --ham-root <ruta a HAM10000> \
    --backbone dinov2_public_vits14=dinov2_vits14:public \
    --backbone dinov2_public_vitb14=dinov2_vitb14:public \
    --backbone ssl_plain_vits14=dinov2_vits14:<ruta>/ssl_plain_vits14/simsiam_dinov2_backbone_final.pt \
    --backbone ssl_proposed_vits14=dinov2_vits14:<ruta>/ssl_proposed_vits14/simsiam_dinov2_backbone_final.pt \
    --backbone ssl_plain_vitb14=dinov2_vitb14:<ruta>/ssl_plain_vitb14/simsiam_dinov2_backbone_final.pt \
    --backbone ssl_proposed_vitb14=dinov2_vitb14:<ruta>/ssl_proposed_vitb14/simsiam_dinov2_backbone_final.pt \
    --tone-manifest <ruta>/fst_pseudo_manifest.csv --strata-column tone_tercile \
    --probes logreg logreg_balanced knn --fractions 1.0 0.1 --seeds 0 1 2 \
    --output-dir <salida> --cache-dir <salida>/cache
```

No hace falta escribirlo a mano: `scripts/run_stage2.py` arma esta lista sola
a partir de `--archs` (`_bank_backbones`), que es la version que conviene usar
porque no se desincroniza de los brazos que realmente se entrenaron.

Escribe `bank.json`, `bank_rows.csv`, `strata.json` y `bank.md`. La particion
es `StratifiedGroupKFold` por `lesion_id` con semilla 42 y primer pliegue, la
misma en todos los backbones, de modo que la comparacion entre ellos es
directa.

Prueba de sanidad: `dinov2_public_vits14` con `logreg` al 100 % de etiquetas
debe dar **0,583** de macro-F1. Si se aleja mas de 0,02, la medicion cambio y
las cifras no son comparables. Ese 0,583 corresponde al protocolo actual, que
normaliza las caracteristicas antes del probe (`StandardScaler` en
`src/dermssl/probes.py`); el 0,547 que aparece en material viejo es del
protocolo v5 sin normalizar y NO es el valor esperado aqui.

## 3.1 Comprobacion rapida, sin escribir codigo

`scripts/try_model.py` carga un checkpoint publicado, verifica que encaje con
la arquitectura y que quede congelado, y o bien imprime el vector de una
imagen o bien entrena un probe lineal sobre una muestra de HAM10000 y reporta
las metricas. Corre en CPU.

```bash
# 1. el checkpoint se carga y produce features
python scripts/try_model.py --backbone ssl_proposed_dinov2_vits14_backbone.pt --image lesion.jpg

# 2. el checkpoint sirve para clasificar
python scripts/try_model.py --backbone ssl_proposed_dinov2_vits14_backbone.pt \
    --ham-root /datos/ham10000 --limit 400

# 3. comparar contra el publico sin adaptar
python scripts/try_model.py --backbone public --ham-root /datos/ham10000 --limit 400
```

Si el checkpoint pertenece a otra arquitectura el script lo dice y termina,
en vez de cargar pesos a medias y devolver numeros sin sentido. Las cifras de
esta comprobacion NO son las del informe: usan una muestra chica y una
particion aleatoria simple, no la particion por `lesion_id`. Sirven para
confirmar que el archivo funciona, no para citar.

## 4. Para que sirve y para que no

Sirve como **extractor de caracteristicas congelado** para clasificacion de
lesiones cutaneas, y como punto de partida para un ajuste fino sobre un
conjunto propio.

NO es un dispositivo de diagnostico. No esta validado clinicamente, se
entreno sobre un pool mayoritariamente de piel clara, y su desempeno cae en
el tercil de piel menos luminosa (seccion 3 del documento 22). Cualquier uso
sobre pacientes reales requiere validacion independiente.

## 4.1 Como se publica

`scripts/publish_model_release.sh` arma el release de GitHub:

```bash
scripts/publish_model_release.sh v1.0-tone /ruta/a/los/pesos
scripts/publish_model_release.sh v1.0-tone /ruta/a/los/pesos esperado.sha256
```

Descubre los brazos que haya —no una cantidad fija—, copia cada peso con su
`run.json` renombrandolos por brazo, agrega los resultados del banco, calcula
los SHA-256, genera las notas desde los manifiestos y publica con `gh`.

Acepta el directorio en dos formas: la de entrenamiento
(`<brazo>/simsiam_dinov2_backbone_final.pt` + `<brazo>/run.json`) y la plana
(`<brazo>_backbone.pt` + `<brazo>_run.json`). La anidada es preferible cuando
los archivos viajan entre maquinas, porque cada manifiesto va pegado a su peso
y desparejarlos se vuelve imposible por construccion.

Tres cosas se niega a hacer, todas a proposito:

1. **Publicar un checkpoint cuyo `run.json` no exista.** Un `.pt` sin su
   evidencia no es citable: no se sabe con que datos, que presupuesto ni que
   particion salio, y en una tesis eso lo vuelve inservible aunque el archivo
   funcione. Corta con el nombre del brazo, en vez de publicar tres de cuatro.
2. **Pisar un nombre ya ocupado en el staging.** Los cuatro brazos escriben el
   mismo nombre de archivo en origen; copiados sin renombrar se sobreescriben
   y el release *igual parece correcto* por cantidad de archivos, con el mismo
   peso repetido.
3. **Subir algo que no coincida con los checksums de origen**, si se le pasa
   la lista como tercer argumento. Un checkpoint corrompido en la
   transferencia falla aqui y no en manos de quien lo descargue.

## 5. Trazabilidad

Cada archivo publicado se acompana de su `run.json`, que registra el metodo,
el backbone, los pesos iniciales, la semilla, el presupuesto consumido en
horas-GPU, el numero de imagenes y el hash de la particion. Sin ese archivo
un `.pt` no es citable: no se sabe con que datos ni con que presupuesto
salio.
