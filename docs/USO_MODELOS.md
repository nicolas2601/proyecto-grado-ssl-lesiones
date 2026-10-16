# Uso de los modelos v7 (release)

## Nomenclatura de lectura

**Proposed sin segmentación** clasifica sin usar una máscara de lesión. **Proposed con segmentación** combina imagen completa, recorte y descriptores de lesión; es una variante experimental, sin mejora robusta demostrada. El Modelo de control no es una tercera propuesta ni la versión segmentada.

Los experimentos de segmentación se identifican por arquitectura, origen de pseudo-máscaras y ronda de aprendizaje. Las rondas no son arquitecturas distintas; los identificadores antiguos se conservan solo donde permiten localizar archivos o reproducir comandos.

Ver [nombres, equivalencias y explicación para lectores técnicos y no técnicos](10-glosario-eli5.md). Los registros históricos conservan sus cifras, predicciones y limitaciones.

Guía paso a paso para usar los clasificadores v7 publicados, en tres formas: **(a)** PyTorch sin `transformers`, **(b)** Hugging Face `transformers` y **(c)** ONNX, en Python o en el navegador. Las tres leen los mismos pesos y dan las mismas probabilidades (ver `parity.json` de cada modelo).

**Uso de investigación, no diagnóstico.** No es un dispositivo médico. Ver `MODEL_CARD.md` (métricas evaluadas y de despliegue, limitaciones y licencias). Los pesos son CC BY-NC 4.0, uso no comercial (`LICENSE-WEIGHTS`); el código es Apache-2.0 (`LICENSE`).

Los bloques de código de esta guía se ejecutan tal cual en un entorno virtual limpio, sin el repositorio instalado, contra un modelo aleatorio pequeño con el mismo formato (`scripts/check_release_guides.py`). Si cambias un bloque, vuelve a correr ese chequeo.

## 0. Qué trae el release

```text
v7/
  README.md  MODEL_CARD.md  USO_MODELOS.md  LICENSE  LICENSE-WEIGHTS  SHA256SUMS  sizes.json
  preprocess.py        preprocesamiento de la evaluación (numpy + PIL)
  dinov2_plain.py      DINOv2 en PyTorch puro que lee los pesos de hf/
  web/                 página onnxruntime-web
  results/             final_eval.json de las dos semillas y las tablas
  A_s42/ A_s43/ B_s42/ B_s43/
    hf/config.json  hf/model.safetensors  hf/head.safetensors
    onnx/model.onnx  [onnx/model_fp16.onnx, solo si pasó la regla fp16]
    release.json  parity.json
```

`A_*` es el brazo Modelo de control (adaptación SSL anclada) y `B_*` el Proposed sin segmentación (más el término de tono); `_s42` y `_s43` son las semillas del preentrenamiento. Los ejemplos usan `B_s42`; cambia esa carpeta por la del modelo que quieras. Las clases son las 7 de HAM10000: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

Todos los comandos se corren desde la carpeta `v7/`, con la imagen a clasificar copiada como `lesion.jpg`. Antes de empezar, verifica la descarga:

```bash
sha256sum -c SHA256SUMS
```

## 1. Preprocesamiento (igual en las tres formas)

La imagen pasa por la misma transformación de la evaluación: RGB, lado corto a 255 px con bilineal de PIL (se conserva la proporción), recorte central de 224 x 224, escala a [0, 1] y normalización ImageNet (media `0.485, 0.456, 0.406`, desvío `0.229, 0.224, 0.225`). `preprocess.py` lo implementa sin torch y está comparado contra la transformación de torchvision con tolerancia 1e-6. Solo se admite 224 x 224: los pesos guardan la grilla de posiciones ya interpolada a ese tamaño.

## 2. Forma (a): PyTorch, sin transformers

Paso 1. Crear un entorno e instalar (torch para CPU; en GPU usa el índice de tu versión de CUDA):

```bash
python -m venv .venv-torch
source .venv-torch/bin/activate
```

```bash guide-a-install
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "safetensors>=0.4.5" "numpy>=1.26" "Pillow>=10"
```

Paso 2. Clasificar (`dinov2_plain.py` arma el ViT y carga `hf/model.safetensors` y la cabeza):

```python guide-a
import torch

from dinov2_plain import load_classifier
from preprocess import preprocess_batch

model, classes = load_classifier("B_s42/hf")
batch = torch.from_numpy(preprocess_batch(["lesion.jpg"]))
with torch.no_grad():
    probabilities = model(batch)[0].tolist()

for name, p in sorted(zip(classes, probabilities), key=lambda item: -item[1]):
    print(f"{name}\t{p:.6f}")
```

Para extraer solo las características (CLS de 768 dimensiones), usa `model.backbone(batch)`.

## 3. Forma (b): Hugging Face transformers

Paso 1. Entorno e instalación:

```bash
python -m venv .venv-hf
source .venv-hf/bin/activate
```

```bash guide-b-install
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "transformers>=4.45,<6" "safetensors>=0.4.5" "numpy>=1.26" "Pillow>=10"
```

Paso 2. `hf/` se carga con `AutoModel.from_pretrained` (un `Dinov2Model`); la cabeza lineal va aparte en `hf/head.safetensors`, con el orden de clases y el preprocesamiento en sus metadatos:

```python guide-b
import json

import torch
from safetensors import safe_open
from transformers import AutoModel

from preprocess import preprocess_batch

backbone = AutoModel.from_pretrained("B_s42/hf").eval()
with safe_open("B_s42/hf/head.safetensors", framework="pt") as head:
    weight, bias = head.get_tensor("weight"), head.get_tensor("bias")
    classes = json.loads(head.metadata()["classes"])

batch = torch.from_numpy(preprocess_batch(["lesion.jpg"]))
with torch.no_grad():
    cls = backbone(pixel_values=batch).pooler_output
    probabilities = torch.softmax(cls @ weight.T + bias, dim=-1)[0].tolist()

for name, p in sorted(zip(classes, probabilities), key=lambda item: -item[1]):
    print(f"{name}\t{p:.6f}")
```

`pooler_output` es el CLS después de la LayerNorm final, la misma característica con la que se entrenó el probe. No uses un `AutoImageProcessor` de DINOv2 de otro repositorio: no reproduce esta transformación; usa `preprocess.py`.

## 4. Forma (c): ONNX en Python (CPU)

Paso 1. Entorno e instalación (sin torch):

```bash
python -m venv .venv-onnx
source .venv-onnx/bin/activate
```

```bash guide-c-install
python -m pip install "onnxruntime>=1.20,<2" "numpy>=1.26" "Pillow>=10"
```

Paso 2. El grafo recibe el lote preprocesado (`pixel_values`, float32 `N x 3 x 224 x 224`) y devuelve `probabilities` (`N x 7`):

```python guide-c
import json

import onnxruntime as ort

from preprocess import preprocess_batch

session = ort.InferenceSession("B_s42/onnx/model.onnx", providers=["CPUExecutionProvider"])
with open("B_s42/release.json", encoding="utf-8") as handle:
    classes = json.load(handle)["classes"]

outputs = session.run(["probabilities"], {"pixel_values": preprocess_batch(["lesion.jpg"])})
probabilities = outputs[0][0].tolist()

for name, p in sorted(zip(classes, probabilities), key=lambda item: -item[1]):
    print(f"{name}\t{p:.6f}")
```

Si el modelo trae `onnx/model_fp16.onnx`, es la misma red en media precisión (mitad de tamaño), publicada solo si pasó la regla registrada (argmax igual al de fp32 en al menos el 99,5 % de val y |Δ macro-F1| <= 0,002; resultado en `parity.json`). Se usa igual, cambiando el nombre del archivo.

## 5. Forma (c): ONNX en el navegador

1. Desde la carpeta `v7/`, levanta un servidor local: `python -m http.server 8000`.
2. Abre `http://localhost:8000/web/`.
3. Deja la ruta del modelo (`../B_s42/onnx/model.onnx`) o cámbiala por la de otro modelo, y pulsa **Load model**. El primer uso baja onnxruntime-web 1.30.0 desde jsDelivr.
4. Elige una imagen. Todo corre en el navegador; la imagen no sale de tu equipo.

`web/preprocess.js` reproduce el remuestreo bilineal de PIL bit a bit (probado contra `preprocess.py` con node). La única diferencia posible es el decodificador JPEG del navegador, que puede cambiar algunos píxeles en uno o dos niveles respecto de libjpeg; la página desactiva la gestión de color y la rotación EXIF, igual que PIL.

## 6. Qué no hacer

- Usarlo para diagnosticar, priorizar pacientes o decidir tratamientos.
- Esperar el mismo rendimiento en fotos de celular o webcam: se evaluó solo en dermatoscopía (HAM10000).
- Cambiar el tamaño de entrada o el preprocesamiento: cambia las características y el probe deja de corresponder.
