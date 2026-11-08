# MoCo V10 — complete classifier usage

Research artifact only. Not a medical device; probabilities are not calibrated clinical risks.

The proposed model is MoCo-v3-style continued SSL on DINOv2 ViT-B/14, quota12+4, normalized CLS from blocks8–11 concatenated to3072 values, per-image L2 and separate balanced logistic heads. It is not an ensemble. The release fixes encoder seed42 and final readers C=30.

## Files and class contracts

Download from the private release (GitHub repository access is required):
https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/v10.0.0

- `v10.0.0-runtime.zip`: offline Python architecture, preprocessing, manifest, licenses, readers, synthetic fixtures and validation metadata; extract into `v10/`.
- `v10.0.0-classifier.pt`: complete backbone plus both heads; save as `v10/classifier.pt`.
- `v10.0.0-model.onnx`: complete shared-backbone graph with both heads; save as `v10/model.onnx`.
- `v10.0.0-docker.tar.gz`: offline ONNX runtime image.
- `v10.0.0-source.zip`: curated source, tests, recipe and reproduction tooling.
- `v10.0.0-results.zip`: aggregate lesion tables, E01 recipe and Pau-ready draft.
- `SHA256SUMS`, `V10_MANIFEST.json`, `V10_VALIDATION.json`: integrity and verification records.

Only download the model form you need. The CLI permits the other model binary to be absent; all present files and the selected form must pass integrity checks. Verify the release-asset checksum before renaming a downloaded model file. Never load an unverified checkpoint or use `weights_only=False` as a workaround.

Input is an RGB image, not a vector of diagnosis labels. Separate outputs:
- HAM7: `akiec,bcc,bkl,df,mel,nv,vasc`; graph output `ham7_probabilities`.
- PAD6: `ACK,BCC,MEL,NEV,SCC,SEK`; graph output `pad6_probabilities`.

Do not merge these ontologies. Preprocessing is EXIF transpose→RGB→bicubic squash224→ImageNet normalization. Inference is one view, no D4/TTA, prior correction, StandardScaler or color constancy.

## PyTorch CPU

Tested in a clean Python3.13 Linux environment with the bundled CPU requirements; Windows commands/wheels have not been independently exercised.

```bash
python3 -m venv .venv-v10
. .venv-v10/bin/activate
python -m pip install -r v10/requirements-torch.txt
python v10/runner.py --bundle v10 --runtime torch --task HAM --image v10/fixtures/synthetic-0.png
python v10/runner.py --bundle v10 --runtime torch --task PAD --image v10/fixtures/synthetic-0.png
```

For direct tensor use, first verify the release checksums, then:

```python
import sys
import torch
from PIL import Image
sys.path.insert(0, "v10/code")
from dermssl.release.moco_bundle import load_checkpoint
from dermssl.release.moco_preprocess import preprocess_image
model, metadata = load_checkpoint("v10/classifier.pt")
with Image.open("example.png") as image:
    pixels = torch.from_numpy(preprocess_image(image))
with torch.inference_mode():
    ham7, pad6 = model.eval()(pixels)
```

The loader uses `torch.load(..., weights_only=True)`, strict state shapes and native classes. No torch.hub, transformers, sklearn estimator pickle, network model fetch or training dataset is needed for inference.

## ONNX CPU

Save the downloaded graph as `v10/model.onnx` and extract the runtime archive into the same folder.

```bash
python -m pip install -r v10/requirements-onnx.txt
python v10/runner.py --bundle v10 --runtime onnx --task HAM --image v10/fixtures/synthetic-0.png
python v10/runner.py --bundle v10 --runtime onnx --task PAD --image v10/fixtures/synthetic-0.png
```

The FP32 graph uses opset17 and dynamic batch. Tensor input is `pixel_values` with normalized float32 shape `(N,3,224,224)`. Outputs have shapes `(N,7)` and `(N,6)`. Do not feed raw uint8 pixels or resize differently.

## Docker offline

The image includes the ONNX model, both readers and exact runtime; no model download occurs at startup. It runs as UID65532, with ONNX on CPU/2 threads.

```bash
docker load -i v10.0.0-docker.tar.gz
docker run --rm --network none --read-only --tmpfs /tmp --cpus 2 --memory 1500m \
  -v "$PWD/example.png:/input/image.png:ro" \
  dermssl-moco:v10.0.0 --task HAM --image /input/image.png
```

Change `--task HAM` to `--task PAD` for the separate PAD ontology. Docker does not require installing PyTorch on the host.

## Versioned API

```bash
curl https://derm.nicolasmoreno.site/api/status
curl -F 'file=@example.png' 'https://derm.nicolasmoreno.site/api/predict?model=MOCO_HAM7&mode=base'
curl -F 'file=@example.png' 'https://derm.nicolasmoreno.site/api/predict?model=MOCO_PAD6&mode=base'
```

Default is `MOCO_HAM7`; `MOCO_PAD6` must be selected explicitly. A71/B71 preserve V9 identities and D4 semantics. A client explicitly requesting B71 still receives V9. MoCo TTA is rejected explicitly. JPEG/PNG/WebP, maximum10MiB/20MP, bounded upload/inference concurrency and request rate apply. `/health`, `/api/models`, `/api/report` and `/openapi.json` expose readiness/inventory/research evidence/schema.

## Evidence and limits

Native FP32→PyTorch/checkpoint reload→ONNX CPU, offline Docker and HTTP/HTTPS are checked at probability tolerance1e-4 with fixture argmax agreement. BF16 research versus FP32 serving drift is reported separately; these are not bit-identical precision modes. The OOF means0.674628PAD/0.620537HAM average three training seeds, not the final globally fitted heads. PAD is transductive and HAM development is selection-exposed. Sealed tests, tonal T0/M1 experiments, clinical validity, fairness and absence of overfitting are not established by this release.
