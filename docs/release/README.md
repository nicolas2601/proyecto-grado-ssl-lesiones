# v7 skin-lesion classifiers (research release)

## Nomenclatura de lectura

**Proposed sin segmentación** clasifica sin usar una máscara de lesión. **Proposed con segmentación** combina imagen completa, recorte y descriptores de lesión; es una variante experimental, sin mejora robusta demostrada. El control SSL sin consistencia de tono no es una tercera propuesta ni la versión segmentada.

Los experimentos de segmentación se identifican por arquitectura, origen de pseudo-máscaras y ronda de aprendizaje. Las rondas no son arquitecturas distintas; los identificadores antiguos se conservan solo donde permiten localizar archivos o reproducir comandos.

Ver [nombres, equivalencias y explicación para lectores técnicos y no técnicos](../10-glosario-eli5.md). Los registros históricos conservan sus cifras, predicciones y limitaciones.

Four classifiers from the thesis "self-supervised adaptation of DINOv2 to dermoscopy": `A_s42`, `A_s43` (A\*, anchored SSL adaptation) and `B_s42`, `B_s43` (B\*, plus a tone-consistency term), each a DINOv2 ViT-B/14 backbone adapted without labels and a linear head for the 7 HAM10000 classes.

**Research use only. Not a medical device; do not use for diagnosis.** Weights: CC BY-NC 4.0, non-commercial (`LICENSE-WEIGHTS`); code: Apache-2.0 (`LICENSE`).

| File | What it is |
|---|---|
| `MODEL_CARD.md` | What the models are, what changed from v6, evaluated metrics with 95 % intervals, deployment metrics, limitations, licenses |
| `LICENSE`, `LICENSE-WEIGHTS` | Apache-2.0 for the code; CC BY-NC 4.0 for the weights, with the dataset and DINOv2 attributions |
| `USO_MODELOS.md` | Step-by-step usage guide (Spanish): torch, transformers, ONNX in Python and in the browser |
| `<model>/hf/` | `config.json` + `model.safetensors` (Hugging Face `Dinov2Model`) and `head.safetensors` (fused linear head, class order, preprocessing) |
| `<model>/onnx/model.onnx` | End-to-end classifier: preprocessed image in, 7 probabilities out (fp32; an fp16 copy only if it passed the registered rule) |
| `<model>/release.json` | Provenance: source weights sha256, code revision, split fingerprints, probe settings |
| `<model>/parity.json` | torch vs transformers vs ONNX agreement and reproduction of the evaluation |
| `preprocess.py`, `dinov2_plain.py`, `web/` | Preprocessing (numpy + PIL), plain PyTorch model, browser demo |
| `results/` | The committed evaluation files the metrics come from |
| `SHA256SUMS`, `sizes.json` | Checksums (`sha256sum -c SHA256SUMS`) and file sizes |

Quick start (ONNX, CPU, no torch), from this folder with an image saved as `lesion.jpg`:

```bash
python -m pip install "onnxruntime>=1.20,<2" "numpy>=1.26" "Pillow>=10"
python -c "import json, onnxruntime as ort; from preprocess import preprocess_batch; s = ort.InferenceSession('B_s42/onnx/model.onnx'); c = json.load(open('B_s42/release.json'))['classes']; p = s.run(None, {'pixel_values': preprocess_batch(['lesion.jpg'])})[0][0]; print(dict(zip(c, p.round(4))))"
```

Source code, training and evaluation protocol: the project repository (branch `proyecto-de-grado`), `docs/28-rediseno-brazo-propuesto-v7.md`.
