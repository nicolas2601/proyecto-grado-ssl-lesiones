# DermLIP ResNet34-UNet segmenter, release v10.4.1

Binary skin-lesion segmentation for dermoscopic images. Network input 224x224.

This release replaces the v10.2.0 segmenter with the teacher of run `abl026-resnet34-ima0-s42` (the checkpoint reported in the thesis).

## Packages

| Package | Files | Run with |
|---|---|---|
| ONNX (CPU) | `segmenter.onnx`, `run_segmenter_onnx.py`, `requirements-onnx.txt` | `python run_segmenter_onnx.py in.jpg out.png` |
| PyTorch | `segmenter.pt`, `segmenter_model.py`, `run_segmenter_torch.py`, `requirements-torch.txt` | `python run_segmenter_torch.py in.jpg out.png` |
| Docker | `Dockerfile.segmenter` (ONNX runner, non-root; written, not built) | `docker run -v $PWD:/data IMG /data/in.jpg /data/out.png` |
| Shared | `preprocess.py` (preprocessing and postprocessing, used by both runners) | - |
| Runtime zip | `v10.4.1-segmenter-runtime.zip` (code, requirements, manifest, README; no weights) | - |

Checksums and metrics: `segmenter_manifest.json`, `SHA256SUMS`. Evidence: `evidence/`.

## Model contract

- Input `pixel_values`: float32, NCHW, `[N,3,224,224]`, RGB in [0,1]. ImageNet normalisation is inside the model.
- Output `logits`: float32, `[N,1,224,224]`. Sigmoid, then threshold at 0.5.
- Preprocessing (`preprocess.py`): resize to 224x224 bilinear; hair removal (OpenCV black-hat, 9px ellipse kernel, threshold 10; Telea inpainting radius 3); divide by 255.
- Postprocessing: sigmoid, uint8, bilinear resize to the original size, threshold 0.5. No morphology, hole filling or largest component.

## Verification

- PyTorch (standalone `segmenter_model.py`) vs ONNX on 24 parity images, shipped runners: max abs logit difference 7.2e-05, minimum mask IoU 0.999997 (`evidence/verify_package_24.json`).
- Shipped preprocessing vs training-code reference preprocessing: max abs difference 0.0.
- Checkpoint loads with `strict=True` into the standalone architecture (290 tensors), `evidence/export_info.json`.
- ISIC2018 (2594 images), thesis evaluation of this checkpoint (torch, frozen evaluator, raw path): Dice 0.8320 (95% CI 0.8248 to 0.8394), IoU 0.7483 (95% CI 0.7401 to 0.7571), HD95 291.1 (95% CI 273.9 to 310.1; n = 2592). Source: thesis report `isic_report.json`, 1000-sample bootstrap.
- The full-set evaluation of the ONNX file was not run in this release; the Dice above is the torch checkpoint's number.
- CPU latency (ONNX Runtime 1.31.1, 2 intra-op threads, 2 cores): p50 53.9 ms inference, 63.4 ms end to end, 50 timed runs (`evidence/latency.json`; host load about 3.3 during the run).

## Requirements (exact pins, unchanged from v10.2.0)

- `requirements-onnx.txt`: numpy 2.5.3, pillow 12.3.0, opencv-python-headless 5.0.0.93, onnxruntime 1.31.1.
- `requirements-torch.txt`: torch 2.11.0 (CPU index), plus the same numpy, pillow and opencv pins.

## Licenses

- ResNet34 encoder initialisation: torchvision ImageNet weights, BSD-3-Clause.
- DermLIP: CC-BY-NC-ND-4.0, used only during training-time calibration, not at inference.
- Research / non-commercial use. Verify the ISIC2018 image license before redistributing any sample images.

## Not in this release

- Sample images (`samples/`), full-set ONNX evaluation, and the Docker image build.
