# MoCo proposed classifier — V10 model card

## Identity
- Method: local MoCo-v3-style continued self-supervised adaptation of public DINOv2 ViT-B/14; not an official MoCo v3 reproduction.
- Frozen release encoder: quota12source+4PAD student, seed42; backbone SHA `fe64610560880e2df1185bf17fbe49e2da46a31e0644ddad76bfa7e8651aab1e`.
- Features: final-LayerNorm CLS tokens from blocks8–11, concatenated3072D, then per-image L2.
- Readers: separate balanced multinomial logistic heads for HAM7 and PAD6; grouped-development-selected C=30, refit on each allowed development cohort.
- One shared backbone, two outputs; not an ensemble. SimSiam is a historical comparator, not the current proposed method.

## Inputs and outputs
Input is an RGB image: EXIF transpose, RGB conversion, bicubic squash224, ImageNet mean/std, FP32 NCHW. Labels are outputs, not seven backbone input nodes.

| Task | Native class order | API selection |
|---|---|---|
| HAM7 | akiec,bcc,bkl,df,mel,nv,vasc | MOCO_HAM7 (default) |
| PAD6 | ACK,BCC,MEL,NEV,SCC,SEK | MOCO_PAD6 (explicit) |

There is no cross-ontology mapping, D4/TTA, prior correction, StandardScaler or color constancy in new MoCo inference. Unsupported TTA requests fail explicitly. Existing A71/B71 API IDs remain V9, with their original identity/recipe.

## Training and deployment fit
Unlabeled continuation pool:33620source images plus1134PAD-adaptation images. Batch16 with12+4role sampling;6000steps; upper4blocks+finalnorm trainable; symmetric in-batch InfoNCE temperature0.2; EMAteacher0.99; AdamW backbone/head LR1e-5/1e-4, weight decay0.05, constant LR. No cosine anchor/L2-SP, hue jitter, local crops or tonal-treatment hooks. The delivered encoder is the student.

Heads use only approved development records: PAD1134images/682components; HAM1528/1152. C was selected by grouped3-fold CV repeated with seeds20261025/26/27, grid0.3/1/3/10/30/100, macro-F1 and smaller-C ties, followed by complete-development refit. Those fitting scores are not independent test scores.

## Development evaluation
The classifier design was evaluated using three encoder seeds42/43/44, each with three nested grouped5-outer/3-inner CV readouts. OOF means: PAD macro-F1 `.674628`, HAM `.620537`. These are not ensemble results or an independent score of the deployed seed42/global-head fit. Seed42 OOF CLS4 means were `.673685`/`.634424` under the BF16 research extraction protocol.

Per-lesion precision/recall/F1 and support are in `resultados/clasificacion/v10/`. Supplemental architecture/fine-tuning bank results are separate because their training procedure differs. The proposed method is not best for every lesion. SCC remains weak; rare MEL(PAD27images/18components) and df(HAM18/6) have substantial uncertainty.

## Verification versus performance
The release provides complete PyTorch weights-only loading, a FP32 dual-output ONNX graph with dynamic batch, and an offline non-root Docker image. Native FP32→PyTorch/reload→ONNX probability parity passed tolerance1e-4; maximum observed difference was about1.26e-6. Clean CPU PyTorch, Docker, candidate HTTP and production HTTPS were independently exercised, including native class orders and legacy behavior.

BF16 research extraction and FP32 serving are not bit-identical. On eight fixed allowed samples per cohort, max probability differences were `.01138` HAM/`.002368` PAD, with no sample argmax changes. This small numerical audit is not full-cohort equivalence or new diagnostic validation. Complete model/data/code hashes are in the bundle manifest, provenance and release-validation artifacts.

## Intended use and exclusions
Research, software reproducibility and controlled thesis experiments only. Not a medical device; not for diagnosis, treatment decisions or calibrated clinical-risk estimates. Outputs must not substitute for a dermatologist.

PAD evaluation is transductive: evaluation images entered SSL without labels. HAM development was repeatedly used for selection. No sealed test was opened. Generalization, clinical validity, fairness, subgroup safety and absence of overfitting are not established. Tonal E06–E08/T0/M1/M1−T0 are not executed or certified by this classifier release.

## License and privacy
Project source is Apache2.0; derived weights are subject to the project's CC BY-NC4.0/research restrictions and dataset attributions in LICENSE-WEIGHTS. PAD attribution is retained. No clinical images, patient/group identifiers or per-image predictions are included in the curated release assets. Synthetic fixtures have no diagnostic ground truth.

Reproduction and usage: `scripts/release_v10/USAGE.md`; recipe/source recovery: `docs/42-moco-e01-for-pau.md`. Publication: https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/v10.0.0.
