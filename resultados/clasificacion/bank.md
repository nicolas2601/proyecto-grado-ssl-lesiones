# Frozen-backbone bank (full)

## Probe: knn

| backbone | fraction | headline_metric | mean |
|---|---|---|---|
| dinov2_public_vitb14 | 10% | auc | 0.8121 |
| dinov2_public_vitb14 | 100% | auc | 0.8692 |
| dinov2_public_vits14 | 10% | auc | 0.8094 |
| dinov2_public_vits14 | 100% | auc | 0.8749 |
| ssl_plain_vitb14 | 10% | auc | 0.7198 |
| ssl_plain_vitb14 | 100% | auc | 0.7335 |
| ssl_plain_vits14 | 10% | auc | 0.7339 |
| ssl_plain_vits14 | 100% | auc | 0.7579 |
| ssl_proposed_vitb14 | 10% | auc | 0.7117 |
| ssl_proposed_vitb14 | 100% | auc | 0.7440 |
| ssl_proposed_vits14 | 10% | auc | 0.7084 |
| ssl_proposed_vits14 | 100% | auc | 0.7323 |

## Probe: logreg

| backbone | fraction | headline_metric | mean |
|---|---|---|---|
| dinov2_public_vitb14 | 10% | macro_f1 | 0.4486 |
| dinov2_public_vitb14 | 100% | macro_f1 | 0.6158 |
| dinov2_public_vits14 | 10% | macro_f1 | 0.4699 |
| dinov2_public_vits14 | 100% | macro_f1 | 0.5829 |
| ssl_plain_vitb14 | 10% | macro_f1 | 0.4460 |
| ssl_plain_vitb14 | 100% | macro_f1 | 0.5856 |
| ssl_plain_vits14 | 10% | macro_f1 | 0.3350 |
| ssl_plain_vits14 | 100% | macro_f1 | 0.5307 |
| ssl_proposed_vitb14 | 10% | macro_f1 | 0.3633 |
| ssl_proposed_vitb14 | 100% | macro_f1 | 0.5237 |
| ssl_proposed_vits14 | 10% | macro_f1 | 0.3186 |
| ssl_proposed_vits14 | 100% | macro_f1 | 0.4840 |

## Probe: logreg_balanced

| backbone | fraction | headline_metric | mean |
|---|---|---|---|
| dinov2_public_vitb14 | 10% | macro_f1 | 0.4694 |
| dinov2_public_vitb14 | 100% | macro_f1 | 0.6131 |
| dinov2_public_vits14 | 10% | macro_f1 | 0.4931 |
| dinov2_public_vits14 | 100% | macro_f1 | 0.5578 |
| ssl_plain_vitb14 | 10% | macro_f1 | 0.4654 |
| ssl_plain_vitb14 | 100% | macro_f1 | 0.5544 |
| ssl_plain_vits14 | 10% | macro_f1 | 0.4093 |
| ssl_plain_vits14 | 100% | macro_f1 | 0.5087 |
| ssl_proposed_vitb14 | 10% | macro_f1 | 0.4200 |
| ssl_proposed_vitb14 | 100% | macro_f1 | 0.5368 |
| ssl_proposed_vits14 | 10% | macro_f1 | 0.4017 |
| ssl_proposed_vits14 | 100% | macro_f1 | 0.5064 |

## Strata: dinov2_public_vits14 (tone_tercile, probe=logreg_balanced, fraction=100%)

| label | n | macro_f1 | auc |
|---|---|---|---|
| dark | 820 | 0.5221 | 0.9281 |
| light | 183 | 0.4833 | 0.9083 |
| medium | 957 | 0.5743 | 0.9138 |
| unavailable | 44 | 0.4014 | 0.8760 |

## Strata: dinov2_public_vitb14 (tone_tercile, probe=logreg_balanced, fraction=100%)

| label | n | macro_f1 | auc |
|---|---|---|---|
| dark | 820 | 0.6034 | 0.9169 |
| light | 183 | 0.6175 | 0.9074 |
| medium | 957 | 0.6036 | 0.9293 |
| unavailable | 44 | 0.4395 | 0.8974 |

## Strata: ssl_plain_vits14 (tone_tercile, probe=logreg_balanced, fraction=100%)

| label | n | macro_f1 | auc |
|---|---|---|---|
| dark | 820 | 0.5249 | 0.9324 |
| light | 183 | 0.4772 | 0.8940 |
| medium | 957 | 0.4841 | 0.9136 |
| unavailable | 44 | 0.3903 | 0.8448 |

## Strata: ssl_proposed_vits14 (tone_tercile, probe=logreg_balanced, fraction=100%)

| label | n | macro_f1 | auc |
|---|---|---|---|
| dark | 820 | 0.5186 | 0.9058 |
| light | 183 | 0.4214 | 0.8519 |
| medium | 957 | 0.5017 | 0.9111 |
| unavailable | 44 | 0.3724 | 0.8258 |

## Strata: ssl_plain_vitb14 (tone_tercile, probe=logreg_balanced, fraction=100%)

| label | n | macro_f1 | auc |
|---|---|---|---|
| dark | 820 | 0.5230 | 0.9246 |
| light | 183 | 0.4635 | 0.8942 |
| medium | 957 | 0.5719 | 0.9347 |
| unavailable | 44 | 0.3985 | 0.9113 |

## Strata: ssl_proposed_vitb14 (tone_tercile, probe=logreg_balanced, fraction=100%)

| label | n | macro_f1 | auc |
|---|---|---|---|
| dark | 820 | 0.5309 | 0.9446 |
| light | 183 | 0.4969 | 0.8845 |
| medium | 957 | 0.5289 | 0.9288 |
| unavailable | 44 | 0.3710 | 0.8714 |

