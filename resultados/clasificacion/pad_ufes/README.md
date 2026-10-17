# PAD-UFES-20 zero-shot, primera pasada (docs/30 §2-3)

Generado por `scripts/build_pad_ufes_table.py` desde `pad_ufes_eval.json`; no editar a mano (`--check` lo compara y `tests/test_build_pad_ufes_table.py` lo corre en CI). Protocolo y predicción: docs/30 §2-3. `code_revision` 0c9526fa6707af739a1f1683f2424cce5cd305e4, registro `0c11d722e22fe3ad333cc8e3626ac6d55efcaebb`.

PAD completo: 2298 imágenes, 1373 pacientes. Probe `logreg`. Bootstrap pareado agrupado por paciente, semilla 20260923, 1000 remuestras. Hilos: OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=4, MKL_NUM_THREADS=4 (sidecar de entorno).

Fototipo: El grupo III-VI es 84 % tipo III (392/465), así que NO representa piel oscura. IV (62), V (10) y VI (1) van solo como descriptivo, sin veredicto. PAD-UFES-20 no permite concluir sobre FST V-VI.

## Modelos

- `public`: `dinov2_vitb14:public`
- `plain_s42`: `dinov2_vitb14:/home/jovyan/work/proyectonp/salidas/v7/final_A/simsiam_dinov2_backbone_final.pt`
- `proposed_s42`: `dinov2_vitb14:/home/jovyan/work/proyectonp/salidas/v7/final_B/simsiam_dinov2_backbone_final.pt`
- `plain_s43`: `dinov2_vitb14:/home/jovyan/work/proyectonp/salidas/v7/final_A_s43/simsiam_dinov2_backbone_final.pt`
- `proposed_s43`: `dinov2_vitb14:/home/jovyan/work/proyectonp/salidas/v7/final_B_s43/simsiam_dinov2_backbone_final.pt`

## Lectura primaria (bcc, mel, nv)

1141 imágenes, 720 pacientes.

Veredictos registrados (docs/30 §3), copiados del JSON:

- **PAD-P1**: cumplida (Δ puntual macro-F1 Proposed sin segmentación − público ≤ 0 con las dos semillas)
- **PAD-P2**: cumplida (Δ puntual macro-F1 Modelo de control − público ≤ 0 con las dos semillas)
- **PAD-P3**: no cumplida (Δ puntual ECE Proposed sin segmentación − público < 0 con las dos semillas)
- **PAD-P4**: no cumplida (IC de macro-F1 Proposed sin segmentación − Modelo de control cruza el 0 con las dos semillas)

| Predicción | Semilla | Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---|---|---:|---|---|
| PAD-P1 | repetición 42 | `proposed_s42` − `public` | Macro-F1 | -0,1070 | [-0,1350; -0,0789] | a favor de `public` (IC excluye 0) |
| PAD-P1 | repetición 43 | `proposed_s43` − `public` | Macro-F1 | -0,1312 | [-0,1628; -0,0984] | a favor de `public` (IC excluye 0) |
| PAD-P2 | repetición 42 | `plain_s42` − `public` | Macro-F1 | -0,0859 | [-0,1112; -0,0572] | a favor de `public` (IC excluye 0) |
| PAD-P2 | repetición 43 | `plain_s43` − `public` | Macro-F1 | -0,1085 | [-0,1351; -0,0825] | a favor de `public` (IC excluye 0) |
| PAD-P3 | repetición 42 | `proposed_s42` − `public` | ECE | 0,1128 | [0,0797; 0,1453] | a favor de `public` (IC excluye 0) |
| PAD-P3 | repetición 43 | `proposed_s43` − `public` | ECE | 0,0978 | [0,0637; 0,1307] | a favor de `public` (IC excluye 0) |
| PAD-P4 | repetición 42 | `proposed_s42` − `plain_s42` | Macro-F1 | -0,0212 | [-0,0456; 0,0007] | indistinguible |
| PAD-P4 | repetición 43 | `proposed_s43` − `plain_s43` | Macro-F1 | -0,0227 | [-0,0437; -0,0005] | a favor de `plain_s43` (IC excluye 0) |

Mejora robusta a la semilla (IC excluye 0 a favor del brazo con las dos semillas, según el JSON):

- `plain-public/macro_f1`: no
- `plain-public/balanced_accuracy`: no
- `plain-public/auc`: no
- `plain-public/ece`: no
- `proposed-public/macro_f1`: no
- `proposed-public/balanced_accuracy`: no
- `proposed-public/auc`: no
- `proposed-public/ece`: no
- `proposed-plain/macro_f1`: no
- `proposed-plain/balanced_accuracy`: no
- `proposed-plain/auc`: no
- `proposed-plain/ece`: no

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] | Argmax fuera de la lectura |
|---|---|---|---|---|---|
| `public` | 0,4244 [0,3963; 0,4543] | 0,5105 [0,4758; 0,5459] | 0,7560 [0,7195; 0,7880] | 0,3576 [0,3289; 0,3924] | 0,2962 |
| `plain_s42` | 0,3385 [0,3084; 0,3710] | 0,4780 [0,4348; 0,5242] | 0,7566 [0,7154; 0,7959] | 0,3923 [0,3588; 0,4236] | 0,4224 |
| `proposed_s42` | 0,3173 [0,2859; 0,3508] | 0,4886 [0,4391; 0,5372] | 0,7860 [0,7497; 0,8225] | 0,4704 [0,4356; 0,5047] | 0,2831 |
| `plain_s43` | 0,3159 [0,2860; 0,3475] | 0,4552 [0,4161; 0,5005] | 0,7553 [0,7169; 0,7933] | 0,4170 [0,3845; 0,4519] | 0,3628 |
| `proposed_s43` | 0,2932 [0,2620; 0,3267] | 0,4497 [0,4029; 0,5012] | 0,7488 [0,7088; 0,7884] | 0,4554 [0,4230; 0,4895] | 0,2770 |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `plain_s42` − `public` | Macro-F1 | -0,0859 | [-0,1112; -0,0572] | a favor de `public` (IC excluye 0) |
| `plain_s42` − `public` | Balanced acc. | -0,0325 | [-0,0735; 0,0098] | indistinguible |
| `plain_s42` − `public` | AUC | 0,0006 | [-0,0403; 0,0376] | indistinguible |
| `plain_s42` − `public` | ECE | 0,0347 | [0,0018; 0,0660] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `public` | Macro-F1 | -0,1070 | [-0,1350; -0,0789] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `public` | Balanced acc. | -0,0219 | [-0,0709; 0,0243] | indistinguible |
| `proposed_s42` − `public` | AUC | 0,0301 | [-0,0107; 0,0678] | indistinguible |
| `proposed_s42` − `public` | ECE | 0,1128 | [0,0797; 0,1453] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | Macro-F1 | -0,0212 | [-0,0456; 0,0007] | indistinguible |
| `proposed_s42` − `plain_s42` | Balanced acc. | 0,0106 | [-0,0271; 0,0471] | indistinguible |
| `proposed_s42` − `plain_s42` | AUC | 0,0295 | [0,0047; 0,0537] | mejora de `proposed_s42` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | ECE | 0,0780 | [0,0501; 0,1087] | a favor de `plain_s42` (IC excluye 0) |
| `plain_s43` − `public` | Macro-F1 | -0,1085 | [-0,1351; -0,0825] | a favor de `public` (IC excluye 0) |
| `plain_s43` − `public` | Balanced acc. | -0,0553 | [-0,0984; -0,0135] | a favor de `public` (IC excluye 0) |
| `plain_s43` − `public` | AUC | -0,0006 | [-0,0393; 0,0342] | indistinguible |
| `plain_s43` − `public` | ECE | 0,0594 | [0,0263; 0,0900] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `public` | Macro-F1 | -0,1312 | [-0,1628; -0,0984] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `public` | Balanced acc. | -0,0608 | [-0,1149; -0,0038] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `public` | AUC | -0,0072 | [-0,0448; 0,0288] | indistinguible |
| `proposed_s43` − `public` | ECE | 0,0978 | [0,0637; 0,1307] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | Macro-F1 | -0,0227 | [-0,0437; -0,0005] | a favor de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | Balanced acc. | -0,0055 | [-0,0362; 0,0319] | indistinguible |
| `proposed_s43` − `plain_s43` | AUC | -0,0065 | [-0,0301; 0,0158] | indistinguible |
| `proposed_s43` − `plain_s43` | ECE | 0,0384 | [0,0134; 0,0644] | a favor de `plain_s43` (IC excluye 0) |

## Lectura secundaria (akiec = ACK + SCC, bcc, mel, nv)

2063 imágenes, 1246 pacientes.

Veredictos registrados (docs/30 §3), copiados del JSON:

- **PAD-P1**: cumplida (Δ puntual macro-F1 Proposed sin segmentación − público ≤ 0 con las dos semillas)
- **PAD-P2**: cumplida (Δ puntual macro-F1 Modelo de control − público ≤ 0 con las dos semillas)
- **PAD-P3**: no cumplida (Δ puntual ECE Proposed sin segmentación − público < 0 con las dos semillas)
- **PAD-P4**: no cumplida (IC de macro-F1 Proposed sin segmentación − Modelo de control cruza el 0 con las dos semillas)

| Predicción | Semilla | Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---|---|---:|---|---|
| PAD-P1 | repetición 42 | `proposed_s42` − `public` | Macro-F1 | -0,0711 | [-0,0921; -0,0525] | a favor de `public` (IC excluye 0) |
| PAD-P1 | repetición 43 | `proposed_s43` − `public` | Macro-F1 | -0,0865 | [-0,1087; -0,0669] | a favor de `public` (IC excluye 0) |
| PAD-P2 | repetición 42 | `plain_s42` − `public` | Macro-F1 | -0,0660 | [-0,0855; -0,0481] | a favor de `public` (IC excluye 0) |
| PAD-P2 | repetición 43 | `plain_s43` − `public` | Macro-F1 | -0,0433 | [-0,0631; -0,0248] | a favor de `public` (IC excluye 0) |
| PAD-P3 | repetición 42 | `proposed_s42` − `public` | ECE | 0,0295 | [0,0073; 0,0539] | a favor de `public` (IC excluye 0) |
| PAD-P3 | repetición 43 | `proposed_s43` − `public` | ECE | 0,0114 | [-0,0100; 0,0348] | indistinguible |
| PAD-P4 | repetición 42 | `proposed_s42` − `plain_s42` | Macro-F1 | -0,0051 | [-0,0214; 0,0099] | indistinguible |
| PAD-P4 | repetición 43 | `proposed_s43` − `plain_s43` | Macro-F1 | -0,0432 | [-0,0599; -0,0267] | a favor de `plain_s43` (IC excluye 0) |

Mejora robusta a la semilla (IC excluye 0 a favor del brazo con las dos semillas, según el JSON):

- `plain-public/macro_f1`: no
- `plain-public/balanced_accuracy`: no
- `plain-public/auc`: no
- `plain-public/ece`: no
- `proposed-public/macro_f1`: no
- `proposed-public/balanced_accuracy`: no
- `proposed-public/auc`: no
- `proposed-public/ece`: no
- `proposed-plain/macro_f1`: no
- `proposed-plain/balanced_accuracy`: no
- `proposed-plain/auc`: no
- `proposed-plain/ece`: no

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] | Argmax fuera de la lectura |
|---|---|---|---|---|---|
| `public` | 0,2665 [0,2474; 0,2876] | 0,3954 [0,3697; 0,4233] | 0,6522 [0,6261; 0,6753] | 0,5648 [0,5405; 0,5896] | 0,3097 |
| `plain_s42` | 0,2005 [0,1831; 0,2187] | 0,3593 [0,3302; 0,3944] | 0,6533 [0,6222; 0,6829] | 0,5485 [0,5266; 0,5708] | 0,4639 |
| `proposed_s42` | 0,1954 [0,1743; 0,2153] | 0,3738 [0,3353; 0,4137] | 0,6702 [0,6399; 0,6975] | 0,5944 [0,5721; 0,6180] | 0,3286 |
| `plain_s43` | 0,2232 [0,2032; 0,2434] | 0,3638 [0,3341; 0,3983] | 0,6679 [0,6389; 0,6943] | 0,5124 [0,4901; 0,5356] | 0,3160 |
| `proposed_s43` | 0,1801 [0,1595; 0,1994] | 0,3447 [0,3055; 0,3877] | 0,6433 [0,6131; 0,6715] | 0,5762 [0,5555; 0,5988] | 0,3015 |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `plain_s42` − `public` | Macro-F1 | -0,0660 | [-0,0855; -0,0481] | a favor de `public` (IC excluye 0) |
| `plain_s42` − `public` | Balanced acc. | -0,0360 | [-0,0666; -0,0037] | a favor de `public` (IC excluye 0) |
| `plain_s42` − `public` | AUC | 0,0011 | [-0,0284; 0,0314] | indistinguible |
| `plain_s42` − `public` | ECE | -0,0164 | [-0,0372; 0,0072] | indistinguible |
| `proposed_s42` − `public` | Macro-F1 | -0,0711 | [-0,0921; -0,0525] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `public` | Balanced acc. | -0,0216 | [-0,0598; 0,0180] | indistinguible |
| `proposed_s42` − `public` | AUC | 0,0181 | [-0,0116; 0,0446] | indistinguible |
| `proposed_s42` − `public` | ECE | 0,0295 | [0,0073; 0,0539] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | Macro-F1 | -0,0051 | [-0,0214; 0,0099] | indistinguible |
| `proposed_s42` − `plain_s42` | Balanced acc. | 0,0144 | [-0,0157; 0,0412] | indistinguible |
| `proposed_s42` − `plain_s42` | AUC | 0,0170 | [-0,0026; 0,0366] | indistinguible |
| `proposed_s42` − `plain_s42` | ECE | 0,0459 | [0,0274; 0,0654] | a favor de `plain_s42` (IC excluye 0) |
| `plain_s43` − `public` | Macro-F1 | -0,0433 | [-0,0631; -0,0248] | a favor de `public` (IC excluye 0) |
| `plain_s43` − `public` | Balanced acc. | -0,0316 | [-0,0664; -0,0005] | a favor de `public` (IC excluye 0) |
| `plain_s43` − `public` | AUC | 0,0158 | [-0,0139; 0,0415] | indistinguible |
| `plain_s43` − `public` | ECE | -0,0524 | [-0,0760; -0,0293] | mejora de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `public` | Macro-F1 | -0,0865 | [-0,1087; -0,0669] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `public` | Balanced acc. | -0,0506 | [-0,0927; -0,0079] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `public` | AUC | -0,0089 | [-0,0361; 0,0173] | indistinguible |
| `proposed_s43` − `public` | ECE | 0,0114 | [-0,0100; 0,0348] | indistinguible |
| `proposed_s43` − `plain_s43` | Macro-F1 | -0,0432 | [-0,0599; -0,0267] | a favor de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | Balanced acc. | -0,0191 | [-0,0437; 0,0099] | indistinguible |
| `proposed_s43` − `plain_s43` | AUC | -0,0246 | [-0,0417; -0,0066] | a favor de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | ECE | 0,0638 | [0,0451; 0,0845] | a favor de `plain_s43` (IC excluye 0) |

## Lectura exploratoria (akiec, bcc, bkl = SEK, mel, nv), sin veredicto

2298 imágenes, 1373 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] | Argmax fuera de la lectura |
|---|---|---|---|---|---|
| `public` | 0,2255 [0,2063; 0,2447] | 0,3321 [0,3059; 0,3599] | 0,6318 [0,6088; 0,6540] | 0,6133 [0,5932; 0,6354] | 0,0426 |
| `plain_s42` | 0,1671 [0,1470; 0,1859] | 0,2993 [0,2679; 0,3307] | 0,6318 [0,6021; 0,6549] | 0,5787 [0,5596; 0,5991] | 0,0326 |
| `proposed_s42` | 0,1734 [0,1545; 0,1912] | 0,3311 [0,2982; 0,3606] | 0,6575 [0,6318; 0,6802] | 0,5917 [0,5714; 0,6121] | 0,0131 |
| `plain_s43` | 0,1860 [0,1667; 0,2062] | 0,3052 [0,2785; 0,3327] | 0,6413 [0,6144; 0,6640] | 0,5171 [0,4961; 0,5383] | 0,0366 |
| `proposed_s43` | 0,1573 [0,1373; 0,1774] | 0,2980 [0,2647; 0,3334] | 0,6237 [0,5975; 0,6484] | 0,5562 [0,5359; 0,5768] | 0,0470 |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `plain_s42` − `public` | Macro-F1 | -0,0584 | [-0,0787; -0,0384] | a favor de `public` (IC excluye 0) |
| `plain_s42` − `public` | Balanced acc. | -0,0327 | [-0,0628; -0,0017] | a favor de `public` (IC excluye 0) |
| `plain_s42` − `public` | AUC | -0,0000 | [-0,0276; 0,0267] | indistinguible |
| `plain_s42` − `public` | ECE | -0,0346 | [-0,0578; -0,0128] | a favor de `plain_s42` (IC excluye 0) |
| `proposed_s42` − `public` | Macro-F1 | -0,0521 | [-0,0714; -0,0331] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `public` | Balanced acc. | -0,0010 | [-0,0336; 0,0327] | indistinguible |
| `proposed_s42` − `public` | AUC | 0,0258 | [0,0000; 0,0501] | a favor de `proposed_s42` (IC excluye 0) |
| `proposed_s42` − `public` | ECE | -0,0216 | [-0,0421; -0,0002] | a favor de `proposed_s42` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | Macro-F1 | 0,0063 | [-0,0109; 0,0242] | indistinguible |
| `proposed_s42` − `plain_s42` | Balanced acc. | 0,0318 | [-0,0011; 0,0662] | indistinguible |
| `proposed_s42` − `plain_s42` | AUC | 0,0258 | [0,0077; 0,0437] | a favor de `proposed_s42` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | ECE | 0,0130 | [-0,0055; 0,0318] | indistinguible |
| `plain_s43` − `public` | Macro-F1 | -0,0395 | [-0,0579; -0,0217] | a favor de `public` (IC excluye 0) |
| `plain_s43` − `public` | Balanced acc. | -0,0268 | [-0,0537; 0,0005] | indistinguible |
| `plain_s43` − `public` | AUC | 0,0095 | [-0,0165; 0,0345] | indistinguible |
| `plain_s43` − `public` | ECE | -0,0962 | [-0,1168; -0,0745] | a favor de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `public` | Macro-F1 | -0,0682 | [-0,0901; -0,0489] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `public` | Balanced acc. | -0,0341 | [-0,0724; 0,0024] | indistinguible |
| `proposed_s43` − `public` | AUC | -0,0081 | [-0,0331; 0,0150] | indistinguible |
| `proposed_s43` − `public` | ECE | -0,0571 | [-0,0776; -0,0351] | a favor de `proposed_s43` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | Macro-F1 | -0,0288 | [-0,0463; -0,0111] | a favor de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | Balanced acc. | -0,0073 | [-0,0356; 0,0258] | indistinguible |
| `proposed_s43` − `plain_s43` | AUC | -0,0176 | [-0,0350; -0,0001] | a favor de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | ECE | 0,0391 | [0,0224; 0,0567] | a favor de `plain_s43` (IC excluye 0) |

## Estratos por fototipo (lectura primaria, exploratorio, sin veredicto)

Regla (b) de docs/30 §6: el valor puntual se reporta siempre; si alguna remuestra da un valor no finito, el IC queda "no calculable" con el número exacto k/1000 de remuestras no finitas. Un estrato con menos de 100 imágenes lleva el rótulo "sin potencia".

### I

108 imágenes, 61 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] |
|---|---|---|---|---|
| `public` | 0,2760 [0,1905; 0,3775] | 0,5096 [0,1460; 0,6667] | 0,7024 no calculable (6/1000) | 0,4644 [0,3543; 0,5761] |
| `plain_s42` | 0,2349 [0,1362; 0,3567] | 0,6480 [0,3367; 0,7882] | 0,7703 no calculable (6/1000) | 0,4520 [0,3433; 0,5864] |
| `proposed_s42` | 0,1689 [0,0764; 0,2790] | 0,5283 [0,0556; 0,7327] | 0,7026 no calculable (6/1000) | 0,6305 [0,5157; 0,7305] |
| `plain_s43` | 0,2702 [0,1554; 0,3939] | 0,6678 [0,3659; 0,8084] | 0,8248 no calculable (6/1000) | 0,4134 [0,2983; 0,5483] |
| `proposed_s43` | 0,2053 [0,1119; 0,3184] | 0,6282 [0,3158; 0,7633] | 0,7349 no calculable (6/1000) | 0,5239 [0,4109; 0,6351] |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `plain_s42` − `public` | Macro-F1 | -0,0411 | [-0,1142; 0,0324] | indistinguible |
| `plain_s42` − `public` | Balanced acc. | 0,1383 | [-0,0749; 0,4234] | indistinguible |
| `plain_s42` − `public` | AUC | 0,0679 | n/d | no calculable (6/1000) |
| `plain_s42` − `public` | ECE | -0,0123 | [-0,1146; 0,1250] | indistinguible |
| `proposed_s42` − `public` | Macro-F1 | -0,1071 | [-0,1832; -0,0255] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `public` | Balanced acc. | 0,0187 | [-0,4197; 0,3878] | indistinguible |
| `proposed_s42` − `public` | AUC | 0,0002 | n/d | no calculable (6/1000) |
| `proposed_s42` − `public` | ECE | 0,1661 | [0,0327; 0,2868] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | Macro-F1 | -0,0659 | [-0,1336; -0,0035] | a favor de `plain_s42` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | Balanced acc. | -0,1196 | [-0,3800; -0,0074] | a favor de `plain_s42` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | AUC | -0,0677 | n/d | no calculable (6/1000) |
| `proposed_s42` − `plain_s42` | ECE | 0,1785 | [0,0501; 0,2742] | a favor de `plain_s42` (IC excluye 0) |
| `plain_s43` − `public` | Macro-F1 | -0,0057 | [-0,0866; 0,0759] | indistinguible |
| `plain_s43` − `public` | Balanced acc. | 0,1581 | [-0,0551; 0,4400] | indistinguible |
| `plain_s43` − `public` | AUC | 0,1224 | n/d | no calculable (6/1000) |
| `plain_s43` − `public` | ECE | -0,0510 | [-0,1487; 0,0741] | indistinguible |
| `proposed_s43` − `public` | Macro-F1 | -0,0707 | [-0,1435; 0,0046] | indistinguible |
| `proposed_s43` − `public` | Balanced acc. | 0,1185 | [-0,0906; 0,4044] | indistinguible |
| `proposed_s43` − `public` | AUC | 0,0324 | n/d | no calculable (6/1000) |
| `proposed_s43` − `public` | ECE | 0,0595 | [-0,0520; 0,1748] | indistinguible |
| `proposed_s43` − `plain_s43` | Macro-F1 | -0,0650 | [-0,1292; -0,0014] | a favor de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | Balanced acc. | -0,0396 | [-0,0784; 0,0000] | indistinguible |
| `proposed_s43` − `plain_s43` | AUC | -0,0900 | n/d | no calculable (6/1000) |
| `proposed_s43` − `plain_s43` | ECE | 0,1104 | [-0,0178; 0,2208] | indistinguible |

### II

571 imágenes, 348 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] |
|---|---|---|---|---|
| `public` | 0,3345 [0,2958; 0,3761] | 0,5079 [0,4438; 0,5715] | 0,7385 [0,6722; 0,8014] | 0,4009 [0,3562; 0,4439] |
| `plain_s42` | 0,2437 [0,2042; 0,2861] | 0,4757 [0,4159; 0,5382] | 0,7319 [0,6668; 0,7893] | 0,4689 [0,4230; 0,5150] |
| `proposed_s42` | 0,2294 [0,1870; 0,2684] | 0,5047 [0,4448; 0,5636] | 0,7640 [0,7077; 0,8141] | 0,5569 [0,5173; 0,6007] |
| `plain_s43` | 0,2028 [0,1687; 0,2392] | 0,4302 [0,3856; 0,4809] | 0,7107 [0,6512; 0,7648] | 0,5238 [0,4838; 0,5697] |
| `proposed_s43` | 0,2013 [0,1625; 0,2401] | 0,4341 [0,3679; 0,4984] | 0,7271 [0,6688; 0,7734] | 0,5458 [0,5024; 0,5907] |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `plain_s42` − `public` | Macro-F1 | -0,0908 | [-0,1244; -0,0557] | a favor de `public` (IC excluye 0) |
| `plain_s42` − `public` | Balanced acc. | -0,0322 | [-0,1067; 0,0345] | indistinguible |
| `plain_s42` − `public` | AUC | -0,0066 | [-0,0735; 0,0470] | indistinguible |
| `plain_s42` − `public` | ECE | 0,0680 | [0,0181; 0,1193] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `public` | Macro-F1 | -0,1050 | [-0,1416; -0,0683] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `public` | Balanced acc. | -0,0032 | [-0,0832; 0,0708] | indistinguible |
| `proposed_s42` − `public` | AUC | 0,0256 | [-0,0405; 0,0874] | indistinguible |
| `proposed_s42` − `public` | ECE | 0,1560 | [0,1107; 0,2072] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | Macro-F1 | -0,0143 | [-0,0422; 0,0138] | indistinguible |
| `proposed_s42` − `plain_s42` | Balanced acc. | 0,0289 | [-0,0288; 0,0857] | indistinguible |
| `proposed_s42` − `plain_s42` | AUC | 0,0322 | [-0,0129; 0,0808] | indistinguible |
| `proposed_s42` − `plain_s42` | ECE | 0,0880 | [0,0468; 0,1269] | a favor de `plain_s42` (IC excluye 0) |
| `plain_s43` − `public` | Macro-F1 | -0,1316 | [-0,1633; -0,1001] | a favor de `public` (IC excluye 0) |
| `plain_s43` − `public` | Balanced acc. | -0,0777 | [-0,1464; -0,0144] | a favor de `public` (IC excluye 0) |
| `plain_s43` − `public` | AUC | -0,0278 | [-0,0799; 0,0213] | indistinguible |
| `plain_s43` − `public` | ECE | 0,1229 | [0,0802; 0,1733] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `public` | Macro-F1 | -0,1332 | [-0,1727; -0,0902] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `public` | Balanced acc. | -0,0738 | [-0,1626; 0,0066] | indistinguible |
| `proposed_s43` − `public` | AUC | -0,0114 | [-0,0639; 0,0368] | indistinguible |
| `proposed_s43` − `public` | ECE | 0,1449 | [0,0986; 0,1943] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | Macro-F1 | -0,0016 | [-0,0285; 0,0270] | indistinguible |
| `proposed_s43` − `plain_s43` | Balanced acc. | 0,0039 | [-0,0452; 0,0546] | indistinguible |
| `proposed_s43` − `plain_s43` | AUC | 0,0164 | [-0,0209; 0,0478] | indistinguible |
| `proposed_s43` − `plain_s43` | ECE | 0,0220 | [-0,0154; 0,0573] | indistinguible |

### III-VI

293 imágenes, 177 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] |
|---|---|---|---|---|
| `public` | 0,3407 [0,2844; 0,3975] | 0,4838 [0,4357; 0,5542] | 0,7367 [0,6774; 0,7947] | 0,4383 [0,3775; 0,5029] |
| `plain_s42` | 0,2702 [0,2204; 0,3242] | 0,4418 [0,3996; 0,4948] | 0,6903 [0,6128; 0,7730] | 0,4417 [0,3832; 0,5071] |
| `proposed_s42` | 0,2521 [0,1954; 0,3087] | 0,4401 [0,3766; 0,5143] | 0,7205 [0,6558; 0,7886] | 0,5183 [0,4564; 0,5871] |
| `plain_s43` | 0,2621 [0,2104; 0,3176] | 0,4608 [0,4074; 0,5365] | 0,7132 [0,6498; 0,7828] | 0,4604 [0,3953; 0,5195] |
| `proposed_s43` | 0,2236 [0,1747; 0,2802] | 0,4388 [0,3850; 0,5126] | 0,6987 [0,6269; 0,7903] | 0,5263 [0,4632; 0,5852] |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `plain_s42` − `public` | Macro-F1 | -0,0704 | [-0,1211; -0,0192] | a favor de `public` (IC excluye 0) |
| `plain_s42` − `public` | Balanced acc. | -0,0420 | [-0,1239; 0,0306] | indistinguible |
| `plain_s42` − `public` | AUC | -0,0464 | [-0,1174; 0,0313] | indistinguible |
| `plain_s42` − `public` | ECE | 0,0034 | [-0,0604; 0,0691] | indistinguible |
| `proposed_s42` − `public` | Macro-F1 | -0,0886 | [-0,1410; -0,0414] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `public` | Balanced acc. | -0,0437 | [-0,1322; 0,0444] | indistinguible |
| `proposed_s42` − `public` | AUC | -0,0162 | [-0,0909; 0,0463] | indistinguible |
| `proposed_s42` − `public` | ECE | 0,0800 | [0,0265; 0,1374] | a favor de `public` (IC excluye 0) |
| `proposed_s42` − `plain_s42` | Macro-F1 | -0,0182 | [-0,0622; 0,0275] | indistinguible |
| `proposed_s42` − `plain_s42` | Balanced acc. | -0,0017 | [-0,0575; 0,0727] | indistinguible |
| `proposed_s42` − `plain_s42` | AUC | 0,0302 | [-0,0217; 0,0757] | indistinguible |
| `proposed_s42` − `plain_s42` | ECE | 0,0766 | [0,0100; 0,1423] | a favor de `plain_s42` (IC excluye 0) |
| `plain_s43` − `public` | Macro-F1 | -0,0785 | [-0,1355; -0,0281] | a favor de `public` (IC excluye 0) |
| `plain_s43` − `public` | Balanced acc. | -0,0229 | [-0,1142; 0,0672] | indistinguible |
| `plain_s43` − `public` | AUC | -0,0234 | [-0,0992; 0,0509] | indistinguible |
| `plain_s43` − `public` | ECE | 0,0221 | [-0,0442; 0,0906] | indistinguible |
| `proposed_s43` − `public` | Macro-F1 | -0,1170 | [-0,1701; -0,0682] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `public` | Balanced acc. | -0,0449 | [-0,1318; 0,0440] | indistinguible |
| `proposed_s43` − `public` | AUC | -0,0380 | [-0,1071; 0,0323] | indistinguible |
| `proposed_s43` − `public` | ECE | 0,0880 | [0,0215; 0,1496] | a favor de `public` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | Macro-F1 | -0,0385 | [-0,0694; -0,0049] | a favor de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | Balanced acc. | -0,0220 | [-0,0411; -0,0026] | a favor de `plain_s43` (IC excluye 0) |
| `proposed_s43` − `plain_s43` | AUC | -0,0145 | [-0,0680; 0,0368] | indistinguible |
| `proposed_s43` − `plain_s43` | ECE | 0,0659 | [0,0110; 0,1156] | a favor de `plain_s43` (IC excluye 0) |

### sin_fst

169 imágenes, 134 pacientes.

| Modelo | Macro-F1 [IC 95 %] | Balanced acc. [IC 95 %] | AUC [IC 95 %] | ECE [IC 95 %] |
|---|---|---|---|---|
| `public` | 0,3179 [0,3094; 0,3253] | 0,3037 [0,2886; 0,3177] | nan no calculable (1000/1000) | 0,0678 [0,0392; 0,1148] |
| `plain_s42` | 0,3221 [0,3149; 0,3281] | 0,3116 [0,2984; 0,3230] | nan no calculable (1000/1000) | 0,0496 [0,0283; 0,0876] |
| `proposed_s42` | 0,3263 [0,3212; 0,3312] | 0,3195 [0,3099; 0,3291] | nan no calculable (1000/1000) | 0,0278 [0,0192; 0,0585] |
| `plain_s43` | 0,3263 [0,3205; 0,3305] | 0,3195 [0,3086; 0,3277] | nan no calculable (1000/1000) | 0,0381 [0,0266; 0,0754] |
| `proposed_s43` | 0,3242 [0,3167; 0,3302] | 0,3156 [0,3016; 0,3270] | nan no calculable (1000/1000) | 0,0465 [0,0228; 0,0881] |

| Contraste | Métrica | Δ puntual | IC 95 % pareado | Lectura |
|---|---|---:|---|---|
| `plain_s42` − `public` | Macro-F1 | 0,0043 | [-0,0056; 0,0143] | indistinguible |
| `plain_s42` − `public` | Balanced acc. | 0,0079 | [-0,0103; 0,0261] | indistinguible |
| `plain_s42` − `public` | AUC | nan | n/d | no calculable (1000/1000) |
| `plain_s42` − `public` | ECE | -0,0182 | [-0,0671; 0,0273] | indistinguible |
| `proposed_s42` − `public` | Macro-F1 | 0,0084 | [0,0000; 0,0180] | indistinguible |
| `proposed_s42` − `public` | Balanced acc. | 0,0158 | [0,0000; 0,0331] | indistinguible |
| `proposed_s42` − `public` | AUC | nan | n/d | no calculable (1000/1000) |
| `proposed_s42` − `public` | ECE | -0,0401 | [-0,0759; 0,0001] | indistinguible |
| `proposed_s42` − `plain_s42` | Macro-F1 | 0,0042 | [-0,0020; 0,0109] | indistinguible |
| `proposed_s42` − `plain_s42` | Balanced acc. | 0,0079 | [-0,0038; 0,0206] | indistinguible |
| `proposed_s42` − `plain_s42` | AUC | nan | n/d | no calculable (1000/1000) |
| `proposed_s42` − `plain_s42` | ECE | -0,0218 | [-0,0509; 0,0117] | indistinguible |
| `plain_s43` − `public` | Macro-F1 | 0,0084 | [0,0000; 0,0177] | indistinguible |
| `plain_s43` − `public` | Balanced acc. | 0,0158 | [0,0000; 0,0328] | indistinguible |
| `plain_s43` − `public` | AUC | nan | n/d | no calculable (1000/1000) |
| `plain_s43` − `public` | ECE | -0,0297 | [-0,0688; 0,0144] | indistinguible |
| `proposed_s43` − `public` | Macro-F1 | 0,0064 | [-0,0032; 0,0161] | indistinguible |
| `proposed_s43` − `public` | Balanced acc. | 0,0118 | [-0,0061; 0,0296] | indistinguible |
| `proposed_s43` − `public` | AUC | nan | n/d | no calculable (1000/1000) |
| `proposed_s43` − `public` | ECE | -0,0214 | [-0,0697; 0,0260] | indistinguible |
| `proposed_s43` − `plain_s43` | Macro-F1 | -0,0021 | [-0,0088; 0,0041] | indistinguible |
| `proposed_s43` − `plain_s43` | Balanced acc. | -0,0039 | [-0,0166; 0,0079] | indistinguible |
| `proposed_s43` − `plain_s43` | AUC | nan | n/d | no calculable (1000/1000) |
| `proposed_s43` − `plain_s43` | ECE | 0,0084 | [-0,0258; 0,0348] | indistinguible |
