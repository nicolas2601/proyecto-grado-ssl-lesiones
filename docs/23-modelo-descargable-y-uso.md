# 23. Modelo descargable: que se publica, como se descarga y como se prueba

Este documento es el contrato de entrega del modelo. Describe que archivos se
publican, que contiene cada uno, como cargarlos y como reproducir las cifras
del informe. Todo lo que aparece aqui sale de artefactos versionados o de
corridas con su commit citado.

## 1. Que se publica

| Archivo | Que es | Tamano |
|---|---|---|
| `simsiam_dinov2_backbone_final.pt` (ssl_proposed) | Backbone DINOv2 ViT-S/14 adaptado al dominio con pares de tono. El metodo propuesto. | 88 MB |
| `simsiam_dinov2_backbone_final.pt` (ssl_plain) | El mismo backbone adaptado SIN pares de tono. El control con el que se compara. | 88 MB |
| `student_r3/best.pt` | U-Net de segmentacion sin mascaras ni etiquetas (camino B). | 31 MB |
| `fst_pseudo_manifest.csv` | Tono por imagen del pool: bucket ITA, tercil de luminosidad, medianas L*/b*, pixeles de piel. | 3,6 MB |
| `bank.json`, `strata.json`, `bank.md` | Resultados del banco congelado y su desglose por tono. | < 40 KB |

Los `.pt` de los backbones contienen **solo el `state_dict` del backbone**,
sin cabezas de proyeccion ni optimizador: es lo que se necesita para extraer
caracteristicas y nada mas. Los checkpoints intermedios por fraccion de
presupuesto (`*_ckpt_c0*.pt`) quedan en el pod y no se publican.

## 2. Como cargarlo

El backbone se instancia desde `torch.hub` sin pesos y se le carga el
`state_dict`; es el mismo camino que usa `scripts/cluster_frozen_bank.py`.

```python
import torch

backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=False)
state = torch.load("simsiam_dinov2_backbone_final.pt", map_location="cpu", weights_only=True)
backbone.load_state_dict(state)
backbone.eval()

for parameter in backbone.parameters():   # congelado: es un extractor
    parameter.requires_grad_(False)
```

La resolucion de entrada debe ser multiplo del parche del modelo (14 para
ViT-S/14): 224 es la usada en todas las corridas. El preprocesamiento es el
mismo del banco: `Resize(int(224 * 1.14))` bilineal, `CenterCrop(224)`,
`ToTensor` y normalizacion ImageNet.

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

with torch.no_grad():
    features = backbone(preprocess(image).unsqueeze(0))   # (1, 384)
```

El backbone NO clasifica: devuelve un vector de 384 dimensiones. Para
diagnosticar hace falta entrenar encima un probe lineal, que es exactamente
lo que mide el banco.

## 3. Como reproducir las cifras

Con el repositorio y los datos:

```bash
python scripts/cluster_frozen_bank.py \
    --ham-root <ruta a HAM10000> \
    --backbone dinov2_public=dinov2_vits14:public \
    --backbone ssl_plain=dinov2_vits14:<ruta>/ssl_plain/simsiam_dinov2_backbone_final.pt \
    --backbone ssl_proposed=dinov2_vits14:<ruta>/ssl_proposed/simsiam_dinov2_backbone_final.pt \
    --tone-manifest <ruta>/fst_pseudo_manifest.csv --strata-column tone_tercile \
    --probes logreg logreg_balanced knn --fractions 1.0 0.1 --seeds 0 1 2 \
    --output-dir <salida> --cache-dir <salida>/cache
```

Escribe `bank.json`, `bank_rows.csv`, `strata.json` y `bank.md`. La particion
es `StratifiedGroupKFold` por `lesion_id` con semilla 42 y primer pliegue, la
misma en los tres backbones, de modo que la comparacion entre ellos es
directa.

Prueba de sanidad: `dinov2_public` con `logreg` al 100 % de etiquetas debe dar
**0,583** de macro-F1. Si se aleja mas de 0,02, la medicion cambio y las
cifras no son comparables.

## 3.1 Comprobacion rapida, sin escribir codigo

`scripts/try_model.py` carga un checkpoint publicado, verifica que encaje con
la arquitectura y que quede congelado, y o bien imprime el vector de una
imagen o bien entrena un probe lineal sobre una muestra de HAM10000 y reporta
las metricas. Corre en CPU.

```bash
# 1. el checkpoint se carga y produce features
python scripts/try_model.py --backbone simsiam_dinov2_backbone_final.pt --image lesion.jpg

# 2. el checkpoint sirve para clasificar
python scripts/try_model.py --backbone simsiam_dinov2_backbone_final.pt \
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
scripts/publish_model_release.sh v1.0-tone /ruta/a/salidas/tone
```

Copia los dos backbones con su `run.json`, agrega los resultados del banco,
calcula los SHA-256 de todo, redacta las notas con la tabla de archivos, la
comprobacion rapida y la advertencia de uso clinico, y publica con `gh`.

Se niega a publicar un checkpoint cuyo `run.json` no exista. Un `.pt` sin su
evidencia no es citable: no se sabe con que datos, que presupuesto ni que
particion salio, y en una tesis eso lo vuelve inservible aunque el archivo
funcione.

## 5. Trazabilidad

Cada archivo publicado se acompana de su `run.json`, que registra el metodo,
el backbone, los pesos iniciales, la semilla, el presupuesto consumido en
horas-GPU, el numero de imagenes y el hash de la particion. Sin ese archivo
un `.pt` no es citable: no se sabe con que datos ni con que presupuesto
salio.
