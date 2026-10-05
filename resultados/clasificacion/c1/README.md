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
| `dinov2_public_vits14` | (0) sin mascara | 0,581 | 0,560 | 0,932 | 0,082 | pasa |
| `dinov2_public_vits14` | (a) recorte a la caja | 0,583 | 0,554 | 0,928 | 0,080 | pasa |
| `dinov2_public_vits14` | (b) fondo gris | 0,558 | 0,533 | 0,927 | 0,096 | pasa |
| `dinov2_public_vits14` | (c) pooling en mascara | 0,594 | 0,568 | 0,932 | 0,094 | pasa |
| `ssl_plain_vitb14` | (0) sin mascara | 0,578 | 0,534 | 0,945 | 0,016 | pasa |
| `ssl_plain_vitb14` | (a) recorte a la caja | 0,589 | 0,548 | 0,949 | 0,023 | pasa |
| `ssl_plain_vitb14` | (b) fondo gris | 0,533 | 0,491 | 0,939 | 0,024 | pasa |
| `ssl_plain_vitb14` | (c) pooling en mascara | 0,574 | 0,526 | 0,940 | 0,019 | pasa |
| `ssl_plain_vits14` | (0) sin mascara | 0,533 | 0,479 | 0,931 | 0,021 | pasa |
| `ssl_plain_vits14` | (a) recorte a la caja | 0,582 | 0,542 | 0,953 | 0,019 | pasa |
| `ssl_plain_vits14` | (b) fondo gris | 0,452 | 0,413 | 0,932 | 0,024 | pasa |
| `ssl_plain_vits14` | (c) pooling en mascara | 0,522 | 0,468 | 0,930 | 0,021 | pasa |
| `ssl_proposed_vitb14` | (0) sin mascara | 0,526 | 0,480 | 0,944 | 0,027 | pasa |
| `ssl_proposed_vitb14` | (a) recorte a la caja | 0,555 | 0,507 | 0,948 | 0,026 | pasa |
| `ssl_proposed_vitb14` | (b) fondo gris | 0,497 | 0,454 | 0,931 | 0,029 | pasa |
| `ssl_proposed_vitb14` | (c) pooling en mascara | 0,504 | 0,458 | 0,942 | 0,030 | pasa |
| `ssl_proposed_vits14` | (0) sin mascara | 0,482 | 0,434 | 0,921 | 0,038 | pasa |
| `ssl_proposed_vits14` | (a) recorte a la caja | 0,474 | 0,432 | 0,921 | 0,018 | pasa |
| `ssl_proposed_vits14` | (b) fondo gris | 0,367 | 0,336 | 0,909 | 0,031 | pasa |
| `ssl_proposed_vits14` | (c) pooling en mascara | 0,471 | 0,424 | 0,915 | 0,027 | pasa |

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
| `dinov2_public_vits14__bbox_crop` | `dinov2_public_vits14__none` | macro_f1 | 0,0021 | [-0,0405, 0,0410] | no se distingue del ruido |
| `dinov2_public_vits14__bbox_crop` | `dinov2_public_vits14__none` | balanced_accuracy | -0,0064 | [-0,0484, 0,0329] | no se distingue del ruido |
| `dinov2_public_vits14__bbox_crop` | `dinov2_public_vits14__none` | auc | -0,0045 | [-0,0177, 0,0066] | no se distingue del ruido |
| `dinov2_public_vits14__bbox_crop` | `dinov2_public_vits14__none` | ece | -0,0023 | [-0,0205, 0,0123] | no se distingue del ruido |
| `dinov2_public_vits14__mask_pool` | `dinov2_public_vits14__none` | macro_f1 | 0,0131 | [-0,0185, 0,0493] | no se distingue del ruido |
| `dinov2_public_vits14__mask_pool` | `dinov2_public_vits14__none` | balanced_accuracy | 0,0075 | [-0,0234, 0,0430] | no se distingue del ruido |
| `dinov2_public_vits14__mask_pool` | `dinov2_public_vits14__none` | auc | -0,0004 | [-0,0099, 0,0092] | no se distingue del ruido |
| `dinov2_public_vits14__mask_pool` | `dinov2_public_vits14__none` | ece | 0,0116 | [-0,0059, 0,0263] | no se distingue del ruido |
| `dinov2_public_vits14__neutral_bg` | `dinov2_public_vits14__none` | macro_f1 | -0,0232 | [-0,0688, 0,0211] | no se distingue del ruido |
| `dinov2_public_vits14__neutral_bg` | `dinov2_public_vits14__none` | balanced_accuracy | -0,0274 | [-0,0732, 0,0146] | no se distingue del ruido |
| `dinov2_public_vits14__neutral_bg` | `dinov2_public_vits14__none` | auc | -0,0052 | [-0,0189, 0,0092] | no se distingue del ruido |
| `dinov2_public_vits14__neutral_bg` | `dinov2_public_vits14__none` | ece | 0,0135 | [-0,0072, 0,0287] | no se distingue del ruido |
| `ssl_plain_vitb14__bbox_crop` | `ssl_plain_vitb14__none` | macro_f1 | 0,0106 | [-0,0203, 0,0489] | no se distingue del ruido |
| `ssl_plain_vitb14__bbox_crop` | `ssl_plain_vitb14__none` | balanced_accuracy | 0,0138 | [-0,0145, 0,0509] | no se distingue del ruido |
| `ssl_plain_vitb14__bbox_crop` | `ssl_plain_vitb14__none` | auc | 0,0037 | [0,0000, 0,0081] | gana `ssl_plain_vitb14__bbox_crop` |
| `ssl_plain_vitb14__bbox_crop` | `ssl_plain_vitb14__none` | ece | 0,0071 | [-0,0115, 0,0193] | no se distingue del ruido |
| `ssl_plain_vitb14__mask_pool` | `ssl_plain_vitb14__none` | macro_f1 | -0,0049 | [-0,0413, 0,0338] | no se distingue del ruido |
| `ssl_plain_vitb14__mask_pool` | `ssl_plain_vitb14__none` | balanced_accuracy | -0,0088 | [-0,0431, 0,0286] | no se distingue del ruido |
| `ssl_plain_vitb14__mask_pool` | `ssl_plain_vitb14__none` | auc | -0,0056 | [-0,0106, -0,0004] | gana `ssl_plain_vitb14__none` |
| `ssl_plain_vitb14__mask_pool` | `ssl_plain_vitb14__none` | ece | 0,0029 | [-0,0135, 0,0161] | no se distingue del ruido |
| `ssl_plain_vitb14__neutral_bg` | `ssl_plain_vitb14__none` | macro_f1 | -0,0455 | [-0,0928, 0,0014] | no se distingue del ruido |
| `ssl_plain_vitb14__neutral_bg` | `ssl_plain_vitb14__none` | balanced_accuracy | -0,0438 | [-0,0878, -0,0003] | gana `ssl_plain_vitb14__none` |
| `ssl_plain_vitb14__neutral_bg` | `ssl_plain_vitb14__none` | auc | -0,0068 | [-0,0146, 0,0014] | no se distingue del ruido |
| `ssl_plain_vitb14__neutral_bg` | `ssl_plain_vitb14__none` | ece | 0,0080 | [-0,0105, 0,0213] | no se distingue del ruido |
| `ssl_plain_vits14__bbox_crop` | `ssl_plain_vits14__none` | macro_f1 | 0,0492 | [0,0035, 0,0965] | gana `ssl_plain_vits14__bbox_crop` |
| `ssl_plain_vits14__bbox_crop` | `ssl_plain_vits14__none` | balanced_accuracy | 0,0624 | [0,0228, 0,1029] | gana `ssl_plain_vits14__bbox_crop` |
| `ssl_plain_vits14__bbox_crop` | `ssl_plain_vits14__none` | auc | 0,0222 | [0,0125, 0,0349] | gana `ssl_plain_vits14__bbox_crop` |
| `ssl_plain_vits14__bbox_crop` | `ssl_plain_vits14__none` | ece | -0,0019 | [-0,0188, 0,0149] | no se distingue del ruido |
| `ssl_plain_vits14__mask_pool` | `ssl_plain_vits14__none` | macro_f1 | -0,0118 | [-0,0585, 0,0301] | no se distingue del ruido |
| `ssl_plain_vits14__mask_pool` | `ssl_plain_vits14__none` | balanced_accuracy | -0,0114 | [-0,0496, 0,0241] | no se distingue del ruido |
| `ssl_plain_vits14__mask_pool` | `ssl_plain_vits14__none` | auc | -0,0011 | [-0,0063, 0,0049] | no se distingue del ruido |
| `ssl_plain_vits14__mask_pool` | `ssl_plain_vits14__none` | ece | -0,0002 | [-0,0153, 0,0151] | no se distingue del ruido |
| `ssl_plain_vits14__neutral_bg` | `ssl_plain_vits14__none` | macro_f1 | -0,0812 | [-0,1279, -0,0325] | gana `ssl_plain_vits14__none` |
| `ssl_plain_vits14__neutral_bg` | `ssl_plain_vits14__none` | balanced_accuracy | -0,0663 | [-0,1063, -0,0281] | gana `ssl_plain_vits14__none` |
| `ssl_plain_vits14__neutral_bg` | `ssl_plain_vits14__none` | auc | 0,0009 | [-0,0101, 0,0144] | no se distingue del ruido |
| `ssl_plain_vits14__neutral_bg` | `ssl_plain_vits14__none` | ece | 0,0031 | [-0,0136, 0,0204] | no se distingue del ruido |
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
| `ssl_proposed_vits14__bbox_crop` | `ssl_proposed_vits14__none` | macro_f1 | -0,0087 | [-0,0480, 0,0320] | no se distingue del ruido |
| `ssl_proposed_vits14__bbox_crop` | `ssl_proposed_vits14__none` | balanced_accuracy | -0,0023 | [-0,0338, 0,0305] | no se distingue del ruido |
| `ssl_proposed_vits14__bbox_crop` | `ssl_proposed_vits14__none` | auc | 0,0002 | [-0,0089, 0,0105] | no se distingue del ruido |
| `ssl_proposed_vits14__bbox_crop` | `ssl_proposed_vits14__none` | ece | -0,0199 | [-0,0303, 0,0023] | no se distingue del ruido |
| `ssl_proposed_vits14__mask_pool` | `ssl_proposed_vits14__none` | macro_f1 | -0,0111 | [-0,0481, 0,0250] | no se distingue del ruido |
| `ssl_proposed_vits14__mask_pool` | `ssl_proposed_vits14__none` | balanced_accuracy | -0,0098 | [-0,0412, 0,0202] | no se distingue del ruido |
| `ssl_proposed_vits14__mask_pool` | `ssl_proposed_vits14__none` | auc | -0,0061 | [-0,0134, 0,0006] | no se distingue del ruido |
| `ssl_proposed_vits14__mask_pool` | `ssl_proposed_vits14__none` | ece | -0,0116 | [-0,0258, 0,0080] | no se distingue del ruido |
| `ssl_proposed_vits14__neutral_bg` | `ssl_proposed_vits14__none` | macro_f1 | -0,1157 | [-0,1622, -0,0691] | gana `ssl_proposed_vits14__none` |
| `ssl_proposed_vits14__neutral_bg` | `ssl_proposed_vits14__none` | balanced_accuracy | -0,0980 | [-0,1377, -0,0600] | gana `ssl_proposed_vits14__none` |
| `ssl_proposed_vits14__neutral_bg` | `ssl_proposed_vits14__none` | auc | -0,0121 | [-0,0219, -0,0016] | gana `ssl_proposed_vits14__none` |
| `ssl_proposed_vits14__neutral_bg` | `ssl_proposed_vits14__none` | ece | -0,0071 | [-0,0214, 0,0150] | no se distingue del ruido |

## Prediccion registrada

Contraste `dinov2_public_vitb14__mask_pool` contra `dinov2_public_vitb14__none` en balanced_accuracy: delta 0,0093, IC [-0,0286, 0,0480] -> no se distingue del ruido.

## Contrastes entre backbones (pregunta del profe)

Pareado sobre las mismas imagenes de validacion. Primero el DINOv2 publico en cada condicion contra el propuesto en (0), con (0) contra (0) como referencia que reproduce el signo de la v6; despues los dos backbones en la MISMA condicion, con segmentacion en ambos lados.

| Brazo | Control | Metrica | Delta | IC 95 % | Veredicto |
|---|---|---|---|---|---|
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0991 | [0,0510, 0,1476] | gana `dinov2_public_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,1192 | [0,0706, 0,1672] | gana `dinov2_public_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | auc | -0,0184 | [-0,0313, -0,0078] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | ece | 0,1070 | [0,0804, 0,1217] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0780 | [0,0213, 0,1360] | gana `dinov2_public_vitb14__bbox_crop` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,0895 | [0,0350, 0,1414] | gana `dinov2_public_vitb14__bbox_crop` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | auc | -0,0118 | [-0,0251, 0,0001] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | ece | 0,1203 | [0,0943, 0,1368] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | macro_f1 | -0,0040 | [-0,0603, 0,0466] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,0201 | [-0,0262, 0,0656] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | auc | -0,0228 | [-0,0333, -0,0126] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | ece | 0,1326 | [0,1054, 0,1479] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0877 | [0,0319, 0,1389] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,1098 | [0,0580, 0,1583] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | auc | -0,0176 | [-0,0305, -0,0065] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | ece | 0,1038 | [0,0771, 0,1185] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | macro_f1 | 0,0495 | [-0,0054, 0,1097] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | balanced_accuracy | 0,0627 | [0,0095, 0,1179] | gana `dinov2_public_vitb14__bbox_crop` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | auc | -0,0158 | [-0,0280, -0,0045] | gana `ssl_proposed_vitb14__bbox_crop` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | ece | 0,1209 | [0,0942, 0,1334] | gana `ssl_proposed_vitb14__bbox_crop` |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | macro_f1 | 0,0252 | [-0,0228, 0,0690] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | balanced_accuracy | 0,0462 | [0,0046, 0,0880] | gana `dinov2_public_vitb14__neutral_bg` |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | auc | -0,0101 | [-0,0223, 0,0024] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | ece | 0,1306 | [0,1046, 0,1449] | gana `ssl_proposed_vitb14__neutral_bg` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | macro_f1 | 0,1211 | [0,0674, 0,1743] | gana `dinov2_public_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | balanced_accuracy | 0,1411 | [0,0876, 0,1941] | gana `dinov2_public_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | auc | -0,0173 | [-0,0287, -0,0075] | gana `ssl_proposed_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | ece | 0,1034 | [0,0797, 0,1192] | gana `ssl_proposed_vitb14__mask_pool` |

## Estratificado por lesion conservada tras el recorte (umbral 0.9)

Confusor declarado en docs/27: (a) no pasa por el recorte central, asi que en lesiones cerca del borde ve mas lesion que (0), (b) y (c).

### `retained_ge_0.9` (n = 1814)

| Celda | macro-F1 | bal-acc | AUC | ECE |
|---|---|---|---|---|
| `dinov2_public_vitb14__bbox_crop` | 0,606 | 0,571 | 0,930 | 0,137 |
| `dinov2_public_vitb14__mask_pool` | 0,627 | 0,602 | 0,929 | 0,125 |
| `dinov2_public_vitb14__neutral_bg` | 0,526 | 0,505 | 0,922 | 0,150 |
| `dinov2_public_vitb14__none` | 0,609 | 0,583 | 0,926 | 0,125 |
| `ssl_proposed_vitb14__bbox_crop` | 0,554 | 0,506 | 0,950 | 0,027 |
| `ssl_proposed_vitb14__mask_pool` | 0,509 | 0,462 | 0,946 | 0,029 |
| `ssl_proposed_vitb14__neutral_bg` | 0,510 | 0,462 | 0,934 | 0,028 |
| `ssl_proposed_vitb14__none` | 0,528 | 0,480 | 0,946 | 0,033 |

| Brazo | Control | Metrica | Delta | IC 95 % | Veredicto |
|---|---|---|---|---|---|
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | macro_f1 | -0,0030 | [-0,0544, 0,0497] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | balanced_accuracy | -0,0119 | [-0,0644, 0,0475] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | auc | 0,0042 | [-0,0116, 0,0185] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | ece | 0,0120 | [-0,0068, 0,0323] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | macro_f1 | 0,0177 | [-0,0259, 0,0577] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | balanced_accuracy | 0,0190 | [-0,0244, 0,0588] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | auc | 0,0023 | [-0,0100, 0,0153] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | ece | -0,0001 | [-0,0187, 0,0183] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | macro_f1 | -0,0837 | [-0,1400, -0,0279] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | balanced_accuracy | -0,0778 | [-0,1341, -0,0233] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | auc | -0,0043 | [-0,0197, 0,0099] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | ece | 0,0247 | [0,0011, 0,0441] | gana `dinov2_public_vitb14__none` |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0257 | [-0,0178, 0,0742] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,0261 | [-0,0142, 0,0671] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | auc | 0,0044 | [-0,0008, 0,0098] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | ece | -0,0061 | [-0,0190, 0,0158] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | macro_f1 | -0,0186 | [-0,0580, 0,0217] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | balanced_accuracy | -0,0174 | [-0,0538, 0,0161] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | auc | 0,0001 | [-0,0034, 0,0038] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | ece | -0,0044 | [-0,0192, 0,0111] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | macro_f1 | -0,0183 | [-0,0795, 0,0409] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | balanced_accuracy | -0,0183 | [-0,0710, 0,0290] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | auc | -0,0115 | [-0,0214, -0,0027] | gana `ssl_proposed_vitb14__none` |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | ece | -0,0047 | [-0,0208, 0,0161] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0993 | [0,0464, 0,1515] | gana `dinov2_public_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,1220 | [0,0686, 0,1730] | gana `dinov2_public_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | auc | -0,0171 | [-0,0303, -0,0059] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | ece | 0,0920 | [0,0651, 0,1103] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0786 | [0,0156, 0,1354] | gana `dinov2_public_vitb14__bbox_crop` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,0911 | [0,0357, 0,1464] | gana `dinov2_public_vitb14__bbox_crop` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | auc | -0,0152 | [-0,0291, -0,0026] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | ece | 0,1041 | [0,0782, 0,1232] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | macro_f1 | -0,0021 | [-0,0599, 0,0528] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,0252 | [-0,0241, 0,0737] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | auc | -0,0237 | [-0,0354, -0,0131] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | ece | 0,1169 | [0,0901, 0,1354] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0816 | [0,0229, 0,1394] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,1029 | [0,0477, 0,1562] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | auc | -0,0194 | [-0,0322, -0,0066] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | ece | 0,0921 | [0,0668, 0,1110] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | macro_f1 | 0,0529 | [-0,0102, 0,1089] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | balanced_accuracy | 0,0650 | [0,0034, 0,1174] | gana `dinov2_public_vitb14__bbox_crop` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | auc | -0,0197 | [-0,0330, -0,0078] | gana `ssl_proposed_vitb14__bbox_crop` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | ece | 0,1103 | [0,0813, 0,1233] | gana `ssl_proposed_vitb14__bbox_crop` |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | macro_f1 | 0,0161 | [-0,0319, 0,0636] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | balanced_accuracy | 0,0435 | [-0,0003, 0,0849] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | auc | -0,0122 | [-0,0263, 0,0002] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | ece | 0,1215 | [0,0947, 0,1366] | gana `ssl_proposed_vitb14__neutral_bg` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | macro_f1 | 0,1179 | [0,0618, 0,1785] | gana `dinov2_public_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | balanced_accuracy | 0,1394 | [0,0815, 0,1974] | gana `dinov2_public_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | auc | -0,0172 | [-0,0291, -0,0067] | gana `ssl_proposed_vitb14__mask_pool` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | ece | 0,0965 | [0,0709, 0,1143] | gana `ssl_proposed_vitb14__mask_pool` |


### `retained_lt_0.9` (n = 188)

| Celda | macro-F1 | bal-acc | AUC | ECE |
|---|---|---|---|---|
| `dinov2_public_vitb14__bbox_crop` | 0,556 | 0,578 | 0,942 | 0,251 |
| `dinov2_public_vitb14__mask_pool` | 0,611 | 0,608 | 0,904 | 0,219 |
| `dinov2_public_vitb14__neutral_bg` | 0,480 | 0,543 | 0,920 | 0,249 |
| `dinov2_public_vitb14__none` | 0,614 | 0,659 | 0,933 | 0,197 |
| `ssl_proposed_vitb14__bbox_crop` | 0,461 | 0,449 | 0,927 | 0,120 |
| `ssl_proposed_vitb14__mask_pool` | 0,434 | 0,406 | 0,918 | 0,089 |
| `ssl_proposed_vitb14__neutral_bg` | 0,374 | 0,381 | 0,916 | 0,093 |
| `ssl_proposed_vitb14__none` | 0,447 | 0,429 | 0,919 | 0,084 |

| Brazo | Control | Metrica | Delta | IC 95 % | Veredicto |
|---|---|---|---|---|---|
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | macro_f1 | -0,0579 | [-0,1762, 0,0610] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | balanced_accuracy | -0,0802 | [-0,1880, 0,0265] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | auc | 0,0093 | [-0,0165, 0,0336] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `dinov2_public_vitb14__none` | ece | 0,0539 | [-0,0230, 0,1219] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | macro_f1 | -0,0026 | [-0,1193, 0,0732] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | balanced_accuracy | -0,0509 | [-0,1255, 0,0334] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | auc | -0,0290 | [-0,0653, 0,0021] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `dinov2_public_vitb14__none` | ece | 0,0225 | [-0,0454, 0,0860] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | macro_f1 | -0,1339 | [-0,2342, -0,0098] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | balanced_accuracy | -0,1159 | [-0,2127, -0,0270] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | auc | -0,0129 | [-0,0437, 0,0136] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `dinov2_public_vitb14__none` | ece | 0,0525 | [-0,0166, 0,1214] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0136 | [-0,0553, 0,0805] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,0203 | [-0,0481, 0,0918] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | auc | 0,0084 | [-0,0057, 0,0229] | no se distingue del ruido |
| `ssl_proposed_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | ece | 0,0361 | [-0,0233, 0,0789] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | macro_f1 | -0,0133 | [-0,1066, 0,1015] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | balanced_accuracy | -0,0232 | [-0,1194, 0,0856] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | auc | -0,0009 | [-0,0153, 0,0096] | no se distingue del ruido |
| `ssl_proposed_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | ece | 0,0054 | [-0,0411, 0,0548] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | macro_f1 | -0,0735 | [-0,1514, 0,0238] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | balanced_accuracy | -0,0483 | [-0,1276, 0,0387] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | auc | -0,0033 | [-0,0238, 0,0120] | no se distingue del ruido |
| `ssl_proposed_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | ece | 0,0091 | [-0,0416, 0,0633] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | macro_f1 | 0,1641 | [-0,0562, 0,2654] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,1784 | [-0,0465, 0,2842] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | auc | -0,0152 | [-0,0601, 0,0210] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__none` | ece | 0,1355 | [0,0485, 0,1768] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | macro_f1 | 0,1089 | [-0,0522, 0,2299] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,1491 | [-0,0613, 0,2442] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | auc | 0,0231 | [-0,0007, 0,0473] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__none` | ece | 0,1669 | [0,0752, 0,2080] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | macro_f1 | 0,0328 | [-0,0940, 0,1596] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,1135 | [-0,1022, 0,2028] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | auc | 0,0010 | [-0,0279, 0,0252] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__none` | ece | 0,1656 | [0,0793, 0,2074] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | macro_f1 | 0,1668 | [0,0090, 0,2816] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | balanced_accuracy | 0,2294 | [0,0101, 0,3262] | gana `dinov2_public_vitb14__none` |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | auc | 0,0139 | [-0,0132, 0,0412] | no se distingue del ruido |
| `dinov2_public_vitb14__none` | `ssl_proposed_vitb14__none` | ece | 0,1130 | [0,0284, 0,1590] | gana `ssl_proposed_vitb14__none` |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | macro_f1 | 0,0953 | [-0,0644, 0,2126] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | balanced_accuracy | 0,1288 | [-0,0848, 0,2198] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | auc | 0,0147 | [-0,0088, 0,0379] | no se distingue del ruido |
| `dinov2_public_vitb14__bbox_crop` | `ssl_proposed_vitb14__bbox_crop` | ece | 0,1308 | [0,0552, 0,1831] | gana `ssl_proposed_vitb14__bbox_crop` |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | macro_f1 | 0,1063 | [-0,0329, 0,2204] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | balanced_accuracy | 0,1617 | [-0,0456, 0,2436] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | auc | 0,0043 | [-0,0253, 0,0346] | no se distingue del ruido |
| `dinov2_public_vitb14__neutral_bg` | `ssl_proposed_vitb14__neutral_bg` | ece | 0,1565 | [0,0658, 0,2056] | gana `ssl_proposed_vitb14__neutral_bg` |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | macro_f1 | 0,1774 | [-0,0669, 0,2947] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | balanced_accuracy | 0,2017 | [-0,0351, 0,3173] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | auc | -0,0143 | [-0,0521, 0,0195] | no se distingue del ruido |
| `dinov2_public_vitb14__mask_pool` | `ssl_proposed_vitb14__mask_pool` | ece | 0,1301 | [0,0498, 0,1688] | gana `ssl_proposed_vitb14__mask_pool` |


### `empty_mask` (n = 2)

- empty_mask: n=2 < 30, un IC ahi no dice nada
