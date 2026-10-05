# Release plan: v7 classifiers

Status: code, tests and docs ready; **nothing built or uploaded yet.** Packaging, parity and upload run on the pod, where the weights are. Tag: `v7.0.0`.

## What gets published

Per model (`A_s42`, `A_s43`, `B_s42`, `B_s43`): `hf/` (config.json, model.safetensors ~330 MiB, head.safetensors), `onnx/model.onnx` (~330 MiB), `onnx/model_fp16.onnx` (~165 MiB, only if it passes the fp16 rule), `release.json`, `parity.json`. Shared: README, MODEL_CARD (with the deployment metrics), USO_MODELOS, LICENSE (Apache-2.0, code), LICENSE-WEIGHTS (CC BY-NC 4.0, weights), preprocess.py, dinov2_plain.py, web/, results/, sizes.json, SHA256SUMS. Sizes are estimates for ViT-B/14 in fp32 (86 M parameters x 4 bytes); `sizes.json` records the real ones.

- **GitHub Release:** every file is one asset, flattened (`B_s42/onnx/model.onnx` -> `B_s42__onnx__model.onnx`). Per-asset limit 2 GiB; `finalize_release.py` and `upload_release.py` both refuse larger files. Total ~3 GB for four models.
- **Hugging Face Hub (option):** one model repository with the folder layout as is (`upload_folder`), so `AutoModel.from_pretrained("<repo>", subfolder="B_s42/hf")` works directly.

## Before publishing (blocking)

Parity must pass for the four models (below). There is no override: a model whose packaged head does not give exactly the committed predictions and macro-F1, or whose forms disagree, is not released. Licenses are settled: code Apache-2.0 (`LICENSE`), weights CC BY-NC 4.0 (`LICENSE-WEIGHTS`; HAM10000, ISIC 2019 and ISIC 2020 are CC BY-NC 4.0, DINOv2 Apache-2.0).

## Pod commands

Paths as in `c1_final_eval.py`. The tree must be clean (the scripts record the code revision). Packaging needs a GPU: it only accepts `--device cuda`, which reproduces the evaluation's fp16-autocast features, so the refit head is the evaluated one. Parity runs on CPU; its `deployment_metrics` block is ONNX fp32 (and fp16) on CPU end to end, reported with deltas to the evaluated values.

```bash
cd ~/work/proyectonp/proyecto-grado-consolidado      # repo checkout on the pod
pip install "safetensors>=0.4.5,<1" "onnx>=1.17,<2" "onnxruntime>=1.20,<2" \
    "onnxconverter-common>=1.14,<2" "transformers>=4.45,<6"
V=/home/jovyan/work/proyectonp/salidas/v7
HAM=~/work/datos/ham10000
F42=resultados/clasificacion/v7/final_eval.json
F43=resultados/clasificacion/v7/s43/final_eval.json

# 1. package (refit probe, fuse, write hf/ + onnx/)
python scripts/build_release_package.py --ham-root $HAM --model A_s42 --weights $V/final_A/simsiam_dinov2_backbone_final.pt     --final-eval $F42 --device cuda
python scripts/build_release_package.py --ham-root $HAM --model B_s42 --weights $V/final_B/simsiam_dinov2_backbone_final.pt     --final-eval $F42 --device cuda
python scripts/build_release_package.py --ham-root $HAM --model A_s43 --weights $V/final_A_s43/simsiam_dinov2_backbone_final.pt --final-eval $F43 --device cuda
python scripts/build_release_package.py --ham-root $HAM --model B_s43 --weights $V/final_B_s43/simsiam_dinov2_backbone_final.pt --final-eval $F43 --device cuda

# 2. parity on CPU (writes <model>/parity.json with deployment_metrics; exit 1 on any failed gate)
for M in A_s42 B_s42; do python scripts/release_parity.py --ham-root $HAM --model-dir release/v7/$M --final-eval $F42; done
for M in A_s43 B_s43; do python scripts/release_parity.py --ham-root $HAM --model-dir release/v7/$M --final-eval $F43; done

# 3. assemble: model card with deployment metrics, licenses, code, results, fp16 rule, sizes.json, SHA256SUMS, 2 GiB check
python scripts/finalize_release.py --release release/v7

# 4. upload: dry run first; --execute only when Nico says so
chmod 600 ~/work/proyectonp/.secrets/*
python scripts/upload_release.py --target github --github-repo OWNER/REPO --token-file ~/work/proyectonp/.secrets/github_token
python scripts/upload_release.py --target hf --hf-repo OWNER/REPO --token-file ~/work/proyectonp/.secrets/hf_token
```

Optional, needs network: `python scripts/check_release_guides.py` runs `USO_MODELOS.md` literally in clean venvs against a tiny model.

## Rollback

A GitHub Release can be deleted (`gh release delete v7.0.0 --cleanup-tag`); a Hub repository can be made private or deleted. The release folder is rebuilt from the weights and the committed code, so nothing is lost by deleting it.
