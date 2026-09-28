# Release plan: v7 classifiers

Status: code, tests and docs ready; **nothing built or uploaded yet.** Packaging, parity and upload run on the pod, where the weights are. Tag: `v7.0.0`.

## What gets published

Per model (`A_s42`, `A_s43`, `B_s42`, `B_s43`): `hf/` (config.json, model.safetensors ~330 MiB, head.safetensors), `onnx/model.onnx` (~330 MiB), `onnx/model_fp16.onnx` (~165 MiB, only if it passes the fp16 rule), `release.json`, `parity.json`. Shared: README, MODEL_CARD, USO_MODELOS, LICENSE, preprocess.py, dinov2_plain.py, web/, results/, sizes.json, SHA256SUMS. Sizes are estimates for ViT-B/14 in fp32 (86 M parameters x 4 bytes); `sizes.json` records the real ones.

- **GitHub Release:** every file is one asset, flattened (`B_s42/onnx/model.onnx` -> `B_s42__onnx__model.onnx`). Per-asset limit 2 GiB; `finalize_release.py` and `upload_release.py` both refuse larger files. Total ~3 GB for four models.
- **Hugging Face Hub (option):** one model repository with the folder layout as is (`upload_folder`), so `AutoModel.from_pretrained("<repo>", subfolder="B_s42/hf")` works directly.

## Before publishing (blocking)

1. Licenses of ISIC 2019 and ISIC 2020 (pretraining pool) are not recorded in the repository: verify them and update `docs/release/model_card_template.md` and `docs/release/LICENSE.md`.
2. The repository has no license file: decide the code license stated in `LICENSE.md`.
3. Parity must pass for the four models (below). If `final_eval.json` is not reproduced exactly, read `parity.json` before using `--allow-json-drift`.

## Pod commands

Paths as in `c1_final_eval.py`. The tree must be clean (the scripts record the code revision). `--device cuda` for the packaging step reproduces the evaluation's fp16-autocast features, so the refit probe is the evaluated one; on CPU the probe is refit on fp32 features and `parity.json` shows the resulting difference.

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

# 2. parity (CPU is fine; writes <model>/parity.json, exit 1 on a failed gate)
for M in A_s42 B_s42; do python scripts/release_parity.py --ham-root $HAM --model-dir release/v7/$M --final-eval $F42; done
for M in A_s43 B_s43; do python scripts/release_parity.py --ham-root $HAM --model-dir release/v7/$M --final-eval $F43; done

# 3. assemble: docs, code, results, fp16 rule applied, sizes.json, SHA256SUMS, 2 GiB check
python scripts/finalize_release.py --release release/v7

# 4. upload: dry run first; --execute only when Nico says so
chmod 600 ~/work/proyectonp/.secrets/*
python scripts/upload_release.py --target github --github-repo OWNER/REPO --token-file ~/work/proyectonp/.secrets/github_token
python scripts/upload_release.py --target hf --hf-repo OWNER/REPO --token-file ~/work/proyectonp/.secrets/hf_token
```

Optional, needs network: `python scripts/check_release_guides.py` runs `USO_MODELOS.md` literally in clean venvs against a tiny model.

## Rollback

A GitHub Release can be deleted (`gh release delete v7.0.0 --cleanup-tag`); a Hub repository can be made private or deleted. The release folder is rebuilt from the weights and the committed code, so nothing is lost by deleting it.
