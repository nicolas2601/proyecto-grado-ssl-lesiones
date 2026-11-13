# DDI malignant-vs-benign AUC against published dermatology AI

Same task and set as Daneshjou et al. 2022 (arXiv:2203.08807, Sci. Adv.): all 656 DDI images,
DDI `malignant` label. Our score = p(mel) + p(bcc) + p(akiec) of the HAM7 head; 2,000 bootstrap
resamples (seed 0). DDI was never used for training or selection of our readers.

| Model | All (n = 656) | FST I-II (n = 208) | FST V-VI (n = 207) |
|---|---|---|---|
| DeepDerm (published) | 0.56 (0.51-0.61) | 0.61 | 0.50 |
| ModelDerm (published) | 0.65 (0.61-0.70) | 0.64 | 0.55 |
| HAM10000 model (published) | 0.67 (0.62-0.71) | 0.72 | 0.57 |
| v10.1.0 HAM7 head (ours) | 0.666 (0.613-0.716) | 0.726 (0.643-0.807) | 0.494 (0.392-0.598) |
| v10.4.0 HAM7 head (ours) | 0.743 (0.695-0.787) | 0.715 (0.629-0.797) | 0.705 (0.614-0.794) |

Published values: Daneshjou et al., Results, first and second paragraphs (full text read). The same
paper reports 0.72-0.77 after supervised fine-tuning on DDI itself. Intervals overlap with ModelDerm
at the upper end; read as a point-estimate improvement, not a significance claim.
