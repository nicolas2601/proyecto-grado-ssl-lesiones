# License note for the v7 model release

The model files in this release (`hf/model.safetensors`, `hf/head.safetensors`, `onnx/*.onnx`) are a **non-commercial research derivative**. They may be used, shared and adapted for non-commercial research with attribution; commercial use is not permitted.

Why: the linear head was fitted on HAM10000 and HAM10000 was part of the self-supervised pretraining pool. HAM10000 is licensed **CC BY-NC 4.0** (Harvard Dataverse, doi:10.7910/DVN/DBW86T), and the models inherit its non-commercial condition.

| Component | Used for | License |
|---|---|---|
| HAM10000 (Tschandl et al., 2018) | probe training and pretraining pool | CC BY-NC 4.0 |
| ISIC 2019 | pretraining pool, unlabelled | not recorded in the project repository; to be verified before publishing |
| ISIC 2020 | pretraining pool, unlabelled | not recorded in the project repository; to be verified before publishing |
| DINOv2 ViT-B/14 (Oquab et al., 2023) | initial weights | Apache License 2.0 |

Attribution: cite HAM10000 (Tschandl, Rosendahl and Kittler, "The HAM10000 dataset", Scientific Data 5, 180161, 2018), DINOv2 (Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023) and this project.

The code files (`preprocess.py`, `dinov2_plain.py`, `web/`) come from the project repository, which has no license file yet; until one is added, no license for the code is granted beyond reading and running it for research. **To decide before publishing.**

No warranty. The models are not a medical device and must not be used for diagnosis or any clinical decision.
