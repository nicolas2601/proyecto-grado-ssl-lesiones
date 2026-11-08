# 41. Proposed MoCo release V10: development results and status

## Status

**Proposed; not published and not deployed.** The generated tables describe development out-of-fold (OOF) readouts. They are not a sealed test, an independent clinical validation, or the performance of a globally fitted deployment reader. The complete package, inference parity gates, release publication, and API deployment remain separate release gates.

The current proposed method is **MoCo-v3-style continued self-supervised learning on DINOv2 ViT-B/14**, using the predeclared quota of 12 source images plus 4 unlabeled PAD-UFES-20 images per batch. The proposed classifier concatenates normalized CLS tokens from blocks 8–11 (3,072 dimensions), then L2-normalizes each feature row. SimSiam remains a historical comparator, not the V10 proposal. The raw seed-42 checkpoint retains a legacy filename containing `simsiam`; that filename is not the method identity.

The model takes RGB images with shape `(N, 3, 224, 224)`. It has two dataset-specific output heads: PAD6 (`ACK`, `BCC`, `MEL`, `NEV`, `SCC`, `SEK`) and HAM7 (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`). The labels are output classes, not input nodes; the two ontologies are not merged.

## Results

The [generated PAD6 and HAM7 tables](../resultados/clasificacion/v10/README.md) include per-class precision, recall, F1, image and group/component support, plus accuracy, macro-F1, balanced accuracy, macro one-vs-rest AUC, and ECE. Each MoCo variant aggregates three independently trained encoder seeds, each evaluated with three repeated grouped five-fold CV runs. The values are averaged readout results: they are **not an ensemble of the three encoders**. The public DINOv2 reference has one encoder checkpoint and the same three CV repetitions.

| Cohort | Images / groups | Proposed macro-F1 | M0 CLS macro-F1 | Descriptive difference |
|---|---:|---:|---:|---:|
| PAD-UFES-20 development | 1,134 / 682 | 0.674628 | 0.643762 | +0.030866 |
| HAM10000 development | 1,528 / 1,152 | 0.620537 | 0.602002 | +0.018535 |

The difference is descriptive; no significance test is asserted. In the proposed OOF rows, PAD MEL support is 27 images from 18 groups. HAM `df` support is 18 images from 6 groups. SCC support/recall is also weak; all are visible in the generated tables rather than hidden in pooled metrics. Component/group counts can overlap across classes and therefore are not additive.

Comparators in the generated tables include the public DINOv2 reference, three-seed VICReg and Barlow Twins + VICReg readouts, an explicitly unidentified legacy baseline (not attributed to a method without source identity), and a frozen SimSiam ResNet-50 bank result shown as architecture context. The SimSiam ResNet-50 is not a matched DINOv2 backbone. Fine-tuned bank comparators are excluded from this frozen-readout table because their evaluation/training procedure differs.

## Interpretation limits

- PAD is transductive: all 1,134 images and 682 groups in `pad_adapt` were present as unlabeled SSL inputs. Its label-aware reader uses grouped development CV.
- HAM metrics use `ham_dev`; this cohort was exposed to development selection and is not an independent validation set.
- The sealed HAM holdout and PAD `target_test` were not accessed for these tables.
- Conditional uncertainty estimates from fixed OOF predictions, if reported elsewhere, do not include encoder-training or model-selection uncertainty and are not p-values.
- These results do not establish clinical utility, generalization, fairness, or absence of overfitting. This is not a medical device and must not be used for diagnosis or clinical decisions.
- GPU availability is observational, not reserved capacity. The OOF table is not a GPU reservation or a reproducible wall-time estimate.

## Provenance

- Evidence plan SHA-256: `e4589916523799fc653ce55a8c6a9754dea44827533985e0a6fd591b7cf88e1d`.
- Aggregate input-receipt/OOF SHA-256: `b0df0e23a998a405d06d880d389df9ad84f0977f3fc0c348419856372c30e7b1`.
- Protocol SHA-256: `1144e857e768a94e4e8df5129e5ec0b5f9f73d3f5c2a5f2982b0e42e9e335414`.
- Predeclared seed-42 quota4 student checkpoint SHA-256: `fe64610560880e2df1185bf17fbe49e2da46a31e0644ddad76bfa7e8651aab1e`. It was fixed for the proposed bundle, not chosen for having the best score.
- Source plan and summary receipts, plus development OOF JSONs, are retained locally under the ignored `results/moco-v10-evidence-inputs/`; the raw image identifiers, group identifiers, and per-image probabilities are not included in the generated JSON/CSV/Markdown outputs.

Regenerate or check the aggregate artifacts with:

```bash
python3 -I scripts/build_moco_results.py
python3 -I scripts/build_moco_results.py --check
```
