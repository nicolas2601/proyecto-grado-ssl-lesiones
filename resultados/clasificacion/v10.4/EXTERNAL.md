| External set | n | bacc v10.1 → v10.4 | non-nevus→nevus v10.1 → v10.4 | mel→nevus v10.1 → v10.4 | malignant AUC v10.1 → v10.4 |
|---|---|---|---|---|---|
| MILK10k clinical, test lesions | 1005 | 0.16 → 0.68 | 0.88 → 0.03 | 0.95 → 0.15 | 0.74 → 0.87 |
| MILK10k dermoscopic, test lesions | 1015 | 0.53 → 0.64 | 0.06 → 0.01 | 0.23 → 0.11 | 0.80 → 0.89 |
| PAD-UFES-20 test (smartphone) | 1155 | 0.31 → 0.69 | 0.32 → 0.02 | 0.48 → 0.20 | 0.75 → 0.98 |
| DDI (clinical, diverse skin tones) | 390 | 0.15 → 0.35 | 0.90 → 0.22 | 0.95 → 0.24 | 0.77 → 0.78 |
| Fitzpatrick17k atlas subset (web) | 612 | 0.28 → 0.69 | 0.55 → 0.04 | 0.54 → 0.11 | 0.77 → 0.97 |
| Wikimedia Commons, reviewed (web) | 76 | 0.35 → 0.55 | 0.69 → 0.07 | 0.84 → 0.11 | 0.73 → 0.84 |
| BOSQUE (Colombia) | 151 | 0.28 → 0.54 | 0.40 → 0.02 | 0.47 → 0.11 | 0.88 → 0.90 |
| HAM10000 holdout (dermoscopy) | 2003 | 0.66 → 0.70 | 0.12 → 0.12 | 0.17 → 0.19 | 0.92 → 0.91 |

Reader: `cell:2:3`; selection rule: max dev_clin_bacc + 0.5*dev_derm_bacc on milk_dev.
Training rows: ham_dev 1528, ham_fit 6249, isic2019_noham 15316, milk_train 6196, pad_adapt 1134.
