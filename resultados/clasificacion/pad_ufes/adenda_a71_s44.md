# PAD-UFES-20, adenda descriptiva: Modelo de control y semilla 44 (docs/30 §7)

Generado por `scripts/build_pad_ufes_addendum_table.py` desde `pad_ufes_addendum_a71_s44.json`; no editar a mano (`--check` lo compara y `tests/test_build_pad_ufes_addendum_table.py` lo corre en CI). `code_revision` 0c9526fa6707af739a1f1683f2424cce5cd305e4, registro `76aa91e7b68cbfd55bd93aa857f6587efd5717b7`.

**Solo descriptivo.** No hay predicción ni regla de decisión: un contraste se lee a lo sumo como "el IC pareado excluye el 0" o "no se distingue". Bootstrap pareado agrupado por paciente, semilla 20260923, 1000 remuestras.

## Modelos y protocolos

- `public`: logreg on CLS @224, no TTA (docs/30 2.3, as the first pass)
- `plain_s44`: logreg on CLS @224, no TTA (docs/30 2.3, as the first pass)
- `proposed_s44`: logreg on CLS @224, no TTA (docs/30 2.3, as the first pass)
- `A71`: A_soup_a1.0@224|cls_attn|la0.5|tta
- `public_A71`: public@224|cls_attn|la0.5|tta

## Primaria descriptiva (bcc, mel, nv)

1141 imágenes, 720 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] | Argmax fuera de la lectura |
|---|---|---|---|---|---|
| `public` | 0,4244 [0,3963; 0,4543] | 0,5105 [0,4758; 0,5459] | 0,7560 [0,7195; 0,7880] | 0,3576 [0,3289; 0,3924] | 0,2962 |
| `plain_s44` | 0,3417 [0,3075; 0,3770] | 0,4798 [0,4383; 0,5238] | 0,7922 [0,7547; 0,8305] | 0,4110 [0,3770; 0,4443] | 0,4303 |
| `proposed_s44` | 0,3052 [0,2761; 0,3352] | 0,4563 [0,4097; 0,5079] | 0,7408 [0,7038; 0,7772] | 0,4570 [0,4258; 0,4912] | 0,4531 |
| `A71` | 0,3499 [0,3177; 0,3821] | 0,5258 [0,4699; 0,5831] | 0,7885 [0,7525; 0,8236] | 0,4224 [0,3894; 0,4561] | 0,3655 |
| `public_A71` | 0,4251 [0,3937; 0,4579] | 0,5165 [0,4773; 0,5587] | 0,7554 [0,7150; 0,7914] | 0,3128 [0,2817; 0,3440] | 0,2770 |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `A71` − `public_A71` | Macro-F1 | -0,0752 | [-0,1048; -0,0429] | el IC excluye el 0 |
| `A71` − `public_A71` | Balanced acc. | 0,0093 | [-0,0473; 0,0733] | no se distingue |
| `A71` − `public_A71` | AUC | 0,0332 | [-0,0044; 0,0714] | no se distingue |
| `A71` − `public_A71` | ECE | 0,1096 | [0,0765; 0,1459] | el IC excluye el 0 |
| `plain_s44` − `public` | Macro-F1 | -0,0827 | [-0,1141; -0,0491] | el IC excluye el 0 |
| `plain_s44` − `public` | Balanced acc. | -0,0306 | [-0,0766; 0,0198] | no se distingue |
| `plain_s44` − `public` | AUC | 0,0363 | [-0,0062; 0,0770] | no se distingue |
| `plain_s44` − `public` | ECE | 0,0534 | [0,0175; 0,0861] | el IC excluye el 0 |
| `proposed_s44` − `public` | Macro-F1 | -0,1192 | [-0,1462; -0,0933] | el IC excluye el 0 |
| `proposed_s44` − `public` | Balanced acc. | -0,0542 | [-0,0993; -0,0088] | el IC excluye el 0 |
| `proposed_s44` − `public` | AUC | -0,0152 | [-0,0494; 0,0183] | no se distingue |
| `proposed_s44` − `public` | ECE | 0,0994 | [0,0667; 0,1306] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | Macro-F1 | -0,0365 | [-0,0594; -0,0135] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | Balanced acc. | -0,0236 | [-0,0638; 0,0149] | no se distingue |
| `proposed_s44` − `plain_s44` | AUC | -0,0514 | [-0,0775; -0,0236] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | ECE | 0,0460 | [0,0202; 0,0728] | el IC excluye el 0 |

## Secundaria (akiec = ACK + SCC, bcc, mel, nv)

2063 imágenes, 1246 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] | Argmax fuera de la lectura |
|---|---|---|---|---|---|
| `public` | 0,2665 [0,2474; 0,2876] | 0,3954 [0,3697; 0,4233] | 0,6522 [0,6261; 0,6753] | 0,5648 [0,5405; 0,5896] | 0,3097 |
| `plain_s44` | 0,2524 [0,2300; 0,2756] | 0,3829 [0,3526; 0,4140] | 0,6981 [0,6694; 0,7237] | 0,4948 [0,4703; 0,5171] | 0,3151 |
| `proposed_s44` | 0,2371 [0,2174; 0,2568] | 0,3815 [0,3446; 0,4223] | 0,6624 [0,6354; 0,6903] | 0,5159 [0,4939; 0,5383] | 0,4556 |
| `A71` | 0,2444 [0,2223; 0,2649] | 0,4194 [0,3730; 0,4607] | 0,6931 [0,6665; 0,7182] | 0,5228 [0,5001; 0,5469] | 0,3863 |
| `public_A71` | 0,2810 [0,2624; 0,3015] | 0,4034 [0,3745; 0,4371] | 0,6570 [0,6262; 0,6852] | 0,4902 [0,4675; 0,5136] | 0,3088 |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `A71` − `public_A71` | Macro-F1 | -0,0365 | [-0,0571; -0,0158] | el IC excluye el 0 |
| `A71` − `public_A71` | Balanced acc. | 0,0159 | [-0,0269; 0,0563] | no se distingue |
| `A71` − `public_A71` | AUC | 0,0361 | [0,0081; 0,0610] | el IC excluye el 0 |
| `A71` − `public_A71` | ECE | 0,0326 | [0,0078; 0,0589] | el IC excluye el 0 |
| `plain_s44` − `public` | Macro-F1 | -0,0141 | [-0,0377; 0,0085] | no se distingue |
| `plain_s44` − `public` | Balanced acc. | -0,0125 | [-0,0471; 0,0207] | no se distingue |
| `plain_s44` − `public` | AUC | 0,0460 | [0,0135; 0,0750] | el IC excluye el 0 |
| `plain_s44` − `public` | ECE | -0,0700 | [-0,0996; -0,0439] | el IC excluye el 0 |
| `proposed_s44` − `public` | Macro-F1 | -0,0295 | [-0,0495; -0,0124] | el IC excluye el 0 |
| `proposed_s44` − `public` | Balanced acc. | -0,0138 | [-0,0504; 0,0230] | no se distingue |
| `proposed_s44` − `public` | AUC | 0,0102 | [-0,0147; 0,0341] | no se distingue |
| `proposed_s44` − `public` | ECE | -0,0489 | [-0,0715; -0,0231] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | Macro-F1 | -0,0154 | [-0,0361; 0,0032] | no se distingue |
| `proposed_s44` − `plain_s44` | Balanced acc. | -0,0013 | [-0,0337; 0,0358] | no se distingue |
| `proposed_s44` − `plain_s44` | AUC | -0,0357 | [-0,0549; -0,0163] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | ECE | 0,0211 | [0,0007; 0,0428] | el IC excluye el 0 |

## Exploratoria (akiec, bcc, bkl = SEK, mel, nv)

2298 imágenes, 1373 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] | Argmax fuera de la lectura |
|---|---|---|---|---|---|
| `public` | 0,2255 [0,2063; 0,2447] | 0,3321 [0,3059; 0,3599] | 0,6318 [0,6088; 0,6540] | 0,6133 [0,5932; 0,6354] | 0,0426 |
| `plain_s44` | 0,2123 [0,1913; 0,2327] | 0,3216 [0,2969; 0,3451] | 0,6723 [0,6467; 0,6938] | 0,5022 [0,4806; 0,5235] | 0,0400 |
| `proposed_s44` | 0,1782 [0,1598; 0,1970] | 0,3135 [0,2865; 0,3421] | 0,6421 [0,6177; 0,6644] | 0,5553 [0,5357; 0,5750] | 0,0596 |
| `A71` | 0,2025 [0,1833; 0,2211] | 0,3378 [0,3033; 0,3712] | 0,6699 [0,6474; 0,6891] | 0,5220 [0,5025; 0,5441] | 0,1540 |
| `public_A71` | 0,2311 [0,2118; 0,2502] | 0,3376 [0,3109; 0,3680] | 0,6271 [0,6010; 0,6510] | 0,4980 [0,4755; 0,5217] | 0,0735 |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `A71` − `public_A71` | Macro-F1 | -0,0287 | [-0,0479; -0,0075] | el IC excluye el 0 |
| `A71` − `public_A71` | Balanced acc. | 0,0002 | [-0,0370; 0,0384] | no se distingue |
| `A71` − `public_A71` | AUC | 0,0428 | [0,0161; 0,0668] | el IC excluye el 0 |
| `A71` − `public_A71` | ECE | 0,0240 | [-0,0004; 0,0490] | no se distingue |
| `plain_s44` − `public` | Macro-F1 | -0,0132 | [-0,0352; 0,0072] | no se distingue |
| `plain_s44` − `public` | Balanced acc. | -0,0104 | [-0,0415; 0,0188] | no se distingue |
| `plain_s44` − `public` | AUC | 0,0405 | [0,0121; 0,0678] | el IC excluye el 0 |
| `plain_s44` − `public` | ECE | -0,1111 | [-0,1340; -0,0883] | el IC excluye el 0 |
| `proposed_s44` − `public` | Macro-F1 | -0,0473 | [-0,0652; -0,0310] | el IC excluye el 0 |
| `proposed_s44` − `public` | Balanced acc. | -0,0185 | [-0,0452; 0,0088] | no se distingue |
| `proposed_s44` − `public` | AUC | 0,0103 | [-0,0103; 0,0306] | no se distingue |
| `proposed_s44` − `public` | ECE | -0,0580 | [-0,0807; -0,0338] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | Macro-F1 | -0,0341 | [-0,0514; -0,0176] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | Balanced acc. | -0,0081 | [-0,0327; 0,0174] | no se distingue |
| `proposed_s44` − `plain_s44` | AUC | -0,0302 | [-0,0478; -0,0138] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | ECE | 0,0531 | [0,0359; 0,0713] | el IC excluye el 0 |

## Estratos por fototipo (lectura primaria, exploratorio)

Regla (b) de docs/30 §6: el valor puntual se reporta siempre; si alguna remuestra da un valor no finito, el IC queda "no calculable" con el número exacto k/1000 de remuestras no finitas. No se descarta ninguna remuestra.

El grupo III-VI es 84 % tipo III (392/465), así que NO representa piel oscura. IV (62), V (10) y VI (1) van solo como descriptivo, sin veredicto. PAD-UFES-20 no permite concluir sobre FST V-VI.

### I

108 imágenes, 61 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] |
|---|---|---|---|---|
| `public` | 0,2760 [0,1905; 0,3775] | 0,5096 [0,1460; 0,6667] | 0,7024 no calculable (6/1000) | 0,4644 [0,3543; 0,5761] |
| `plain_s44` | 0,2344 [0,1175; 0,3719] | 0,6381 [0,3192; 0,7755] | 0,8008 no calculable (6/1000) | 0,5340 [0,4412; 0,6492] |
| `proposed_s44` | 0,2055 [0,1162; 0,3156] | 0,5547 [0,0880; 0,7589] | 0,6806 no calculable (6/1000) | 0,5280 [0,4389; 0,6531] |
| `A71` | 0,2155 [0,1206; 0,3252] | 0,4879 [0,0741; 0,7356] | 0,7583 no calculable (6/1000) | 0,4974 [0,4035; 0,6153] |
| `public_A71` | 0,2880 [0,1921; 0,4101] | 0,6843 [0,3620; 0,8222] | 0,7793 no calculable (6/1000) | 0,4323 [0,3423; 0,5285] |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `A71` − `public_A71` | Macro-F1 | -0,0725 | [-0,1519; -0,0111] | el IC excluye el 0 |
| `A71` − `public_A71` | Balanced acc. | -0,1964 | [-0,3919; -0,0143] | el IC excluye el 0 |
| `A71` − `public_A71` | AUC | -0,0210 | n/d | no calculable (6/1000) |
| `A71` − `public_A71` | ECE | 0,0651 | [-0,0290; 0,1694] | no se distingue |
| `plain_s44` − `public` | Macro-F1 | -0,0415 | [-0,1423; 0,0590] | no se distingue |
| `plain_s44` − `public` | Balanced acc. | 0,1284 | [-0,0860; 0,4157] | no se distingue |
| `plain_s44` − `public` | AUC | 0,0984 | n/d | no calculable (6/1000) |
| `plain_s44` − `public` | ECE | 0,0697 | [-0,0311; 0,2043] | no se distingue |
| `proposed_s44` − `public` | Macro-F1 | -0,0704 | [-0,1312; -0,0155] | el IC excluye el 0 |
| `proposed_s44` − `public` | Balanced acc. | 0,0451 | [-0,0947; 0,1218] | no se distingue |
| `proposed_s44` − `public` | AUC | -0,0218 | n/d | no calculable (6/1000) |
| `proposed_s44` − `public` | ECE | 0,0637 | [-0,0243; 0,1870] | no se distingue |
| `proposed_s44` − `plain_s44` | Macro-F1 | -0,0289 | [-0,1047; 0,0449] | no se distingue |
| `proposed_s44` − `plain_s44` | Balanced acc. | -0,0833 | [-0,3423; 0,0275] | no se distingue |
| `proposed_s44` − `plain_s44` | AUC | -0,1202 | n/d | no calculable (6/1000) |
| `proposed_s44` − `plain_s44` | ECE | -0,0060 | [-0,0935; 0,0904] | no se distingue |

### II

571 imágenes, 348 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] |
|---|---|---|---|---|
| `public` | 0,3345 [0,2958; 0,3761] | 0,5079 [0,4438; 0,5715] | 0,7385 [0,6722; 0,8014] | 0,4009 [0,3562; 0,4439] |
| `plain_s44` | 0,2467 [0,2100; 0,2850] | 0,4718 [0,4209; 0,5245] | 0,7843 [0,7264; 0,8312] | 0,4829 [0,4398; 0,5291] |
| `proposed_s44` | 0,2101 [0,1742; 0,2478] | 0,4613 [0,4012; 0,5268] | 0,7095 [0,6451; 0,7653] | 0,5475 [0,5068; 0,5897] |
| `A71` | 0,2507 [0,2060; 0,2933] | 0,5226 [0,4432; 0,6048] | 0,7658 [0,7080; 0,8172] | 0,4944 [0,4475; 0,5420] |
| `public_A71` | 0,3388 [0,2962; 0,3794] | 0,5168 [0,4501; 0,5885] | 0,7314 [0,6667; 0,7963] | 0,3355 [0,2952; 0,3833] |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `A71` − `public_A71` | Macro-F1 | -0,0881 | [-0,1285; -0,0504] | el IC excluye el 0 |
| `A71` − `public_A71` | Balanced acc. | 0,0059 | [-0,0728; 0,0792] | no se distingue |
| `A71` − `public_A71` | AUC | 0,0344 | [-0,0189; 0,0820] | no se distingue |
| `A71` − `public_A71` | ECE | 0,1589 | [0,1062; 0,2106] | el IC excluye el 0 |
| `plain_s44` − `public` | Macro-F1 | -0,0878 | [-0,1270; -0,0494] | el IC excluye el 0 |
| `plain_s44` − `public` | Balanced acc. | -0,0361 | [-0,1117; 0,0276] | no se distingue |
| `plain_s44` − `public` | AUC | 0,0459 | [-0,0192; 0,1018] | no se distingue |
| `plain_s44` − `public` | ECE | 0,0820 | [0,0309; 0,1366] | el IC excluye el 0 |
| `proposed_s44` − `public` | Macro-F1 | -0,1244 | [-0,1556; -0,0896] | el IC excluye el 0 |
| `proposed_s44` − `public` | Balanced acc. | -0,0465 | [-0,1136; 0,0176] | no se distingue |
| `proposed_s44` − `public` | AUC | -0,0289 | [-0,0789; 0,0166] | no se distingue |
| `proposed_s44` − `public` | ECE | 0,1466 | [0,1000; 0,1926] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | Macro-F1 | -0,0366 | [-0,0634; -0,0083] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | Balanced acc. | -0,0104 | [-0,0650; 0,0481] | no se distingue |
| `proposed_s44` − `plain_s44` | AUC | -0,0748 | [-0,1139; -0,0304] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | ECE | 0,0646 | [0,0261; 0,1025] | el IC excluye el 0 |

### III-VI

293 imágenes, 177 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] |
|---|---|---|---|---|
| `public` | 0,3407 [0,2844; 0,3975] | 0,4838 [0,4357; 0,5542] | 0,7367 [0,6774; 0,7947] | 0,4383 [0,3775; 0,5029] |
| `plain_s44` | 0,2867 [0,2261; 0,3463] | 0,4774 [0,4207; 0,5469] | 0,7494 [0,6721; 0,8393] | 0,4519 [0,3893; 0,5166] |
| `proposed_s44` | 0,2308 [0,1812; 0,2819] | 0,3935 [0,3514; 0,4478] | 0,6855 [0,6324; 0,7495] | 0,5154 [0,4558; 0,5740] |
| `A71` | 0,2863 [0,2279; 0,3429] | 0,5052 [0,4205; 0,6107] | 0,7244 [0,6573; 0,7960] | 0,4932 [0,4319; 0,5645] |
| `public_A71` | 0,3280 [0,2757; 0,3766] | 0,4586 [0,4246; 0,4907] | 0,7335 [0,6727; 0,8032] | 0,3888 [0,3270; 0,4563] |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `A71` − `public_A71` | Macro-F1 | -0,0418 | [-0,0876; 0,0035] | no se distingue |
| `A71` − `public_A71` | Balanced acc. | 0,0467 | [-0,0382; 0,1497] | no se distingue |
| `A71` − `public_A71` | AUC | -0,0091 | [-0,0901; 0,0635] | no se distingue |
| `A71` − `public_A71` | ECE | 0,1045 | [0,0394; 0,1765] | el IC excluye el 0 |
| `plain_s44` − `public` | Macro-F1 | -0,0539 | [-0,1107; 0,0048] | no se distingue |
| `plain_s44` − `public` | Balanced acc. | -0,0064 | [-0,0976; 0,0800] | no se distingue |
| `plain_s44` − `public` | AUC | 0,0127 | [-0,0777; 0,1002] | no se distingue |
| `plain_s44` − `public` | ECE | 0,0137 | [-0,0524; 0,0806] | no se distingue |
| `proposed_s44` − `public` | Macro-F1 | -0,1098 | [-0,1555; -0,0699] | el IC excluye el 0 |
| `proposed_s44` − `public` | Balanced acc. | -0,0903 | [-0,1691; -0,0155] | el IC excluye el 0 |
| `proposed_s44` − `public` | AUC | -0,0511 | [-0,1115; 0,0114] | no se distingue |
| `proposed_s44` − `public` | ECE | 0,0771 | [0,0211; 0,1334] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | Macro-F1 | -0,0559 | [-0,1089; -0,0072] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | Balanced acc. | -0,0839 | [-0,1677; -0,0050] | el IC excluye el 0 |
| `proposed_s44` − `plain_s44` | AUC | -0,0639 | [-0,1300; 0,0010] | no se distingue |
| `proposed_s44` − `plain_s44` | ECE | 0,0634 | [0,0049; 0,1125] | el IC excluye el 0 |

### sin_fst

169 imágenes, 134 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] |
|---|---|---|---|---|
| `public` | 0,3179 [0,3094; 0,3253] | 0,3037 [0,2886; 0,3177] | nan no calculable (1000/1000) | 0,0678 [0,0392; 0,1148] |
| `plain_s44` | 0,3242 [0,3168; 0,3302] | 0,3156 [0,3018; 0,3272] | nan no calculable (1000/1000) | 0,0537 [0,0258; 0,0961] |
| `proposed_s44` | 0,3221 [0,3148; 0,3280] | 0,3116 [0,2982; 0,3229] | nan no calculable (1000/1000) | 0,0414 [0,0248; 0,0856] |
| `A71` | 0,3168 [0,3077; 0,3240] | 0,3018 [0,2857; 0,3152] | nan no calculable (1000/1000) | 0,0564 [0,0337; 0,1067] |
| `public_A71` | 0,3189 [0,3091; 0,3262] | 0,3057 [0,2882; 0,3193] | nan no calculable (1000/1000) | 0,0473 [0,0243; 0,0900] |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `A71` − `public_A71` | Macro-F1 | -0,0022 | [-0,0131; 0,0095] | no se distingue |
| `A71` − `public_A71` | Balanced acc. | -0,0039 | [-0,0238; 0,0173] | no se distingue |
| `A71` − `public_A71` | AUC | nan | n/d | no calculable (1000/1000) |
| `A71` − `public_A71` | ECE | 0,0090 | [-0,0326; 0,0602] | no se distingue |
| `plain_s44` − `public` | Macro-F1 | 0,0064 | [-0,0034; 0,0164] | no se distingue |
| `plain_s44` − `public` | Balanced acc. | 0,0118 | [-0,0064; 0,0302] | no se distingue |
| `plain_s44` − `public` | AUC | nan | n/d | no calculable (1000/1000) |
| `plain_s44` − `public` | ECE | -0,0141 | [-0,0650; 0,0375] | no se distingue |
| `proposed_s44` − `public` | Macro-F1 | 0,0043 | [-0,0050; 0,0137] | no se distingue |
| `proposed_s44` − `public` | Balanced acc. | 0,0079 | [-0,0089; 0,0252] | no se distingue |
| `proposed_s44` − `public` | AUC | nan | n/d | no calculable (1000/1000) |
| `proposed_s44` − `public` | ECE | -0,0264 | [-0,0709; 0,0240] | no se distingue |
| `proposed_s44` − `plain_s44` | Macro-F1 | -0,0021 | [-0,0098; 0,0053] | no se distingue |
| `proposed_s44` − `plain_s44` | Balanced acc. | -0,0039 | [-0,0183; 0,0100] | no se distingue |
| `proposed_s44` − `plain_s44` | AUC | nan | n/d | no calculable (1000/1000) |
| `proposed_s44` − `plain_s44` | ECE | -0,0123 | [-0,0374; 0,0250] | no se distingue |

## Datos

- Pacheco et al., *PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones*, Mendeley Data, v1, doi [10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1); artículo en *Data in Brief* 32 (2020) 106221, doi [10.1016/j.dib.2020.106221](https://doi.org/10.1016/j.dib.2020.106221). Licencia CC BY 4.0 (<http://creativecommons.org/licenses/by/4.0>). Cambios: el mapeo de etiquetas de docs/30 §2.2, la exclusión de SEK de las lecturas primaria y secundaria y el preprocesado del C1.
- PAD-UFES-20 no permite concluir sobre FST V-VI.
