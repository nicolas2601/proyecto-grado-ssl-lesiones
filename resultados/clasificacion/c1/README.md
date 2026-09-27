# C1: la segmentacion como entrada del clasificador

Generado por `scripts/build_c1_readme.py` desde `c1_cells.json` y `c1_bootstrap.json`. No editar a mano: `--check` lo compara en CI.

Tabla principal en macro-F1 (como la v6). La prediccion registrada se evalua en balanced accuracy, tal como se registro (docs/27 §6). Se reportan siempre las cuatro metricas. ECE: menor es mejor.

## Celdas

| Backbone | Condicion | macro-F1 | bal-acc | AUC | ECE | Gate |
|---|---|---|---|---|---|---|
| `dinov2_public_vitb14` | (0) sin mascara | 0,614 | 0,590 | 0,926 | 0,131 | pasa |
| `dinov2_public_vitb14` | (a) recorte a la caja | 0,604 | 0,570 | 0,932 | 0,147 | pasa |
| `dinov2_public_vitb14` | (b) fondo gris | 0,522 | 0,500 | 0,921 | 0,159 | pasa |
| `dinov2_public_vitb14` | (c) pooling en mascara | 0,625 | 0,599 | 0,925 | 0,134 | pasa |
| `ssl_proposed_vitb14` | (0) sin mascara | 0,526 | 0,480 | 0,944 | 0,027 | pasa |
| `ssl_proposed_vitb14` | (a) recorte a la caja | 0,555 | 0,507 | 0,948 | 0,026 | pasa |
| `ssl_proposed_vitb14` | (b) fondo gris | 0,497 | 0,454 | 0,931 | 0,029 | pasa |
| `ssl_proposed_vitb14` | (c) pooling en mascara | 0,504 | 0,458 | 0,942 | 0,030 | pasa |

## Contrastes pareados contra la condicion (0) del mismo backbone

Bootstrap pareado, 1000 remuestras, n_val = 2004. Una celda que no paso su gate no entra.

| Brazo | Control | Metrica | Delta | IC 95 % | Veredicto |
|---|---|---|---|---|---|
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | macro_f1 | -0,0096 | [-0,0625, 0,0416] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | balanced_accuracy | -0,0203 | [-0,0725, 0,0316] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | auc | 0,0058 | [-0,0086, 0,0205] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | ece | 0,0166 | [-0,0033, 0,0363] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | macro_f1 | 0,0114 | [-0,0259, 0,0522] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | balanced_accuracy | 0,0093 | [-0,0286, 0,0480] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | auc | -0,0008 | [-0,0120, 0,0108] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | ece | 0,0032 | [-0,0156, 0,0202] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | macro_f1 | -0,0916 | [-0,1409, -0,0405] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | balanced_accuracy | -0,0898 | [-0,1392, -0,0368] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | auc | -0,0052 | [-0,0190, 0,0078] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | ece | 0,0289 | [0,0077, 0,0472] | gana `dinov2_public_vitb14__none` |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0286 | [-0,0140, 0,0714] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,0268 | [-0,0084, 0,0651] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | auc | 0,0041 | [-0,0003, 0,0085] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | ece | -0,0005 | [-0,0149, 0,0173] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | macro_f1 | -0,0220 | [-0,0613, 0,0184] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | balanced_accuracy | -0,0220 | [-0,0576, 0,0127] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | auc | -0,0012 | [-0,0050, 0,0022] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | ece | 0,0035 | [-0,0137, 0,0171] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | macro_f1 | -0,0291 | [-0,0821, 0,0251] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | balanced_accuracy | -0,0262 | [-0,0718, 0,0188] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | auc | -0,0127 | [-0,0219, -0,0049] | gana `ssl_proposed_vitb14__none` |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | ece | 0,0020 | [-0,0161, 0,0188] | no se distingue del ruido |

## Prediccion registrada

Contraste `dinov2_public_vitb14__mask_pool` contra `dinov2_public_vitb14__none` en balanced_accuracy: delta 0,0093, IC [-0,0286, 0,0480] -> no se distingue del ruido.
