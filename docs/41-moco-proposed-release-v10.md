# 41. Proposed MoCo release V10: development results and status

## Status

**Complete inference artifacts verified; research API deployed as V10.** Publication/checksum evidence is delivered with the release. The generated tables describe development out-of-fold (OOF) readouts. They are not a sealed test, an independent clinical validation, or the performance of a globally fitted deployment reader. Native FP32/PyTorch/ONNX/Docker and candidate/production HTTPS parity are separate software gates, not new diagnostic-performance estimates.

The current proposed method is **MoCo-v3-style continued self-supervised learning on DINOv2 ViT-B/14**, using the predeclared quota of 12 source images plus 4 unlabeled PAD-UFES-20 images per batch. The proposed classifier concatenates normalized CLS tokens from blocks 8–11 (3,072 dimensions), then L2-normalizes each feature row. SimSiam remains a historical comparator, not the V10 proposal. The raw seed-42 checkpoint retains a legacy filename containing `simsiam`; that filename is not the method identity.

The model takes RGB images with shape `(N, 3, 224, 224)`. It has two dataset-specific output heads: PAD6 (`ACK`, `BCC`, `MEL`, `NEV`, `SCC`, `SEK`) and HAM7 (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`). The labels are output classes, not input nodes; the two ontologies are not merged.

## Results

The [generated PAD6 and HAM7 tables](../resultados/clasificacion/v10/README.md) include per-class precision, recall, F1, image and group/component support, plus accuracy, macro-F1, balanced accuracy, macro one-vs-rest AUC, and ECE. Each MoCo variant aggregates three independently trained encoder seeds, each evaluated with three repeated grouped five-fold CV runs. The values are averaged readout results: they are **not an ensemble of the three encoders**. The public DINOv2 reference has one encoder checkpoint and the same three CV repetitions.

| Cohort | Images / groups | Proposed macro-F1 | M0 CLS macro-F1 | Descriptive difference |
|---|---:|---:|---:|---:|
| PAD-UFES-20 development | 1,134 / 682 | 0.674628 | 0.643762 | +0.030866 |
| HAM10000 development | 1,528 / 1,152 | 0.620537 | 0.602002 | +0.018535 |

The difference is descriptive; no significance test is asserted. In the proposed OOF rows, PAD MEL support is 27 images from 18 groups. HAM `df` support is 18 images from 6 groups. SCC support/recall is also weak; all are visible in the generated tables rather than hidden in pooled metrics. Component/group counts can overlap across classes and therefore are not additive.

Comparators in the generated tables include the public DINOv2 reference, three-seed VICReg and Barlow Twins + VICReg readouts, an explicitly unidentified legacy baseline (not attributed to a method without source identity), and a frozen SimSiam ResNet-50 bank result shown as architecture context. The SimSiam ResNet-50 is not a matched DINOv2 backbone. Fine-tuned bank comparators are excluded from this frozen-readout table because their evaluation/training procedure differs. They are provided separately in [the architecture-bank supplement](../resultados/clasificacion/v10/BANK.md), with per-lesion values in `BANK.csv`; no matched-training or independent-validation claim is made.

## Final development readers (fit complete; performance not independently measured)

Two CPU readers were fit on the fixed seed-42 BF16 CLS4 encoder using all authorized development records: HAM7 (`1,528` images / `1,152` groups) and PAD6 (`1,134` images / `682` groups). C was selected by grouped three-fold CV over seeds `20261025`, `20261026`, and `20261027`, using macro-F1 over C = 0.3, 1, 3, 10, 30, and 100 with the smaller-C tie break. Both selected C = 30, then refit on all allowed development rows; the fit records report 143 HAM7 and 144 PAD6 solver iterations. These are reader-fit settings, not independent performance scores. They must not replace the three-encoder OOF table values. Final model hashes and numerical parity are recorded in [the E01 recipe](42-moco-e01-for-pau.md) and the bundle manifest. PyTorch CPU clean-environment and offline Docker checks passed12cases each; candidate and production HTTPS passed12MoCo and4legacy checks plus error/CORS contracts. API default is `MOCO_HAM7`; PAD6 is explicit, and A71/B71 retain their V9 identities. A consumer explicitly requesting B71 still receives V9 rather than silently renamed MoCo.

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
- Aggregate API report: [`api-report.json`](../resultados/clasificacion/v10/api-report.json). This describes the tables and reader-fit protocol; it is not the eventual bundle `manifest.json` and contains no PT/ONNX hashes.
- Predeclared seed-42 quota4 student checkpoint SHA-256: `fe64610560880e2df1185bf17fbe49e2da46a31e0644ddad76bfa7e8651aab1e`. It was fixed for the proposed bundle, not chosen for having the best score.
- Source plan and summary receipts, plus development OOF JSONs, are retained locally under the ignored `results/moco-v10-evidence-inputs/`; the raw image identifiers, group identifiers, and per-image probabilities are not included in the generated JSON/CSV/Markdown outputs.

Regenerate or check the aggregate artifacts with:

```bash
python3 -I scripts/build_moco_results.py
python3 -I scripts/build_moco_results.py --check
```
