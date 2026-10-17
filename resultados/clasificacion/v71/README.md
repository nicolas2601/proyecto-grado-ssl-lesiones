# Lectura v7.1 (val, docs/29)

n = 2004, `code_revision` 88e434f4576f1aa7dc89c27cdff571ddb3efefdc. Generado desde `v71_final_eval.json`; no editar a mano (`--check` lo compara en CI). La selección se hizo en el holdout interno y validación se leyó una sola vez. El primario v7 no cambia.

| Lectura | Configuración | Macro-F1 | Bal-acc | Acc | AUC | ECE | Sens. mel | AUC mel |
|---|---|---|---|---|---|---|---|---|
| Modelo de control | `A_soup_a1.0@224|cls_attn|la0.5|tta` | 0,7229 [0,6795; 0,7592] | 0,7305 [0,6844; 0,7719] | 0,8398 [0,8229; 0,8553] | 0,9644 [0,9558; 0,9718] | 0,0154 [0,0135; 0,0339] | 0,5856 [0,5195; 0,6438] | 0,9167 [0,8986; 0,9316] |
| Proposed sin segmentación | `B_s42_a1.0@224|cls4|balanced|tta` | 0,7141 [0,6703; 0,7547] | 0,6964 [0,6504; 0,7408] | 0,8378 [0,8218; 0,8533] | 0,9602 [0,9516; 0,9680] | 0,0529 [0,0428; 0,0695] | 0,5315 [0,4663; 0,5938] | 0,9021 [0,8811; 0,9198] |
| Público, protocolo de Modelo de control | `public@224|cls_attn|la0.5|tta` | 0,6320 [0,5875; 0,6710] | 0,6174 [0,5714; 0,6632] | 0,7959 [0,7774; 0,8124] | 0,9432 [0,9339; 0,9519] | 0,0806 [0,0684; 0,0987] | 0,4595 [0,3951; 0,5287] | 0,8458 [0,8207; 0,8723] |
| Público, protocolo de Proposed sin segmentación | `public@224|cls4|balanced|tta` | 0,6837 [0,6381; 0,7202] | 0,6659 [0,6176; 0,7102] | 0,8278 [0,8119; 0,8438] | 0,9619 [0,9549; 0,9680] | 0,0688 [0,0573; 0,0846] | 0,5135 [0,4449; 0,5775] | 0,8931 [0,8725; 0,9124] |
| Modelo de control, protocolo v7 (primario) | `A_s42_a1.0@224|cls|none|id` | 0,6464 [0,5980; 0,6894] | 0,6021 [0,5565; 0,6488] | 0,8318 [0,8154; 0,8468] | 0,9619 [0,9563; 0,9671] | 0,0135 [0,0138; 0,0336] | 0,4865 [0,4235; 0,5490] | 0,8985 [0,8784; 0,9168] |
| Proposed sin segmentación, protocolo v7 (primario) | `B_s42_a1.0@224|cls|none|id` | 0,6606 [0,6157; 0,7040] | 0,6327 [0,5871; 0,6838] | 0,8313 [0,8149; 0,8478] | 0,9629 [0,9563; 0,9690] | 0,0325 [0,0253; 0,0497] | 0,5045 [0,4422; 0,5683] | 0,9100 [0,8917; 0,9261] |
| Público v7 (primario) | `public@224|cls|none|id` | 0,6158 [0,5690; 0,6550] | 0,5924 [0,5459; 0,6376] | 0,7819 [0,7655; 0,7994] | 0,9257 [0,9123; 0,9370] | 0,1307 [0,1159; 0,1476] | 0,4279 [0,3632; 0,4978] | 0,8331 [0,8054; 0,8603] |
| Público, su mejor protocolo | `public@224|cls4|la0.5|tta` | 0,6857 [0,6412; 0,7208] | 0,6622 [0,6160; 0,7058] | 0,8268 [0,8104; 0,8423] | 0,9598 [0,9516; 0,9665] | 0,0702 [0,0605; 0,0871] | 0,5090 [0,4392; 0,5690] | 0,8891 [0,8675; 0,9090] |

## Puntos de operación de melanoma (umbral fijado en el holdout)

Metricas con umbral de mel (sensibilidad y especificidad a los umbrales de especificidad 80/90 % y de sensibilidad 95 %): en val se evaluan con el MISMO probe entrenado en el 80 % del fit, donde se fijaron los umbrales, porque reentrenar cambia la escala de los scores. Metricas sin umbral (macro-F1 por argmax, AUC, bal-acc, ECE, sensibilidad de mel por argmax): probe reentrenado en el train completo, como en v7. Los puntos de operacion re-ajustados sobre val no se reportan (decision de Paula, 2026-09-28; docs/29 §2.7 item 1).

| Lectura | Sens. / esp. (esp. 80 %) | Sens. / esp. (esp. 90 %) | Sens. / esp. (sens. 95 %) |
|---|---|---|---|
| Modelo de control | 0,8604 [0,8143; 0,9056] / 0,8042 [0,7854; 0,8217] | 0,6892 [0,6281; 0,7488] / 0,9029 [0,8886; 0,9161] | 0,9685 [0,9432; 0,9902] / 0,6336 [0,6113; 0,6551] |
| Proposed sin segmentación | 0,8333 [0,7843; 0,8802] / 0,8008 [0,7813; 0,8196] | 0,6757 [0,6113; 0,7328] / 0,9102 [0,8964; 0,9230] | 0,9189 [0,8827; 0,9522] / 0,6723 [0,6509; 0,6936] |
| Público, protocolo de Modelo de control | 0,6937 [0,6348; 0,7510] / 0,8042 [0,7845; 0,8215] | 0,5450 [0,4808; 0,6068] / 0,9147 [0,9012; 0,9270] | 0,9685 [0,9417; 0,9904] / 0,3378 [0,3166; 0,3587] |
| Público, protocolo de Proposed sin segmentación | 0,7477 [0,6917; 0,8037] / 0,8019 [0,7825; 0,8195] | 0,6577 [0,5915; 0,7186] / 0,8967 [0,8826; 0,9099] | 0,9369 [0,9034; 0,9673] / 0,5466 [0,5226; 0,5699] |
| Modelo de control, protocolo v7 (primario) | 0,8468 [0,7990; 0,8934] / 0,7907 [0,7721; 0,8082] | 0,6351 [0,5726; 0,6942] / 0,9074 [0,8951; 0,9210] | 0,9234 [0,8837; 0,9565] / 0,5993 [0,5761; 0,6218] |
| Proposed sin segmentación, protocolo v7 (primario) | 0,8468 [0,7975; 0,8914] / 0,8053 [0,7869; 0,8225] | 0,6802 [0,6176; 0,7426] / 0,9024 [0,8878; 0,9149] | 0,9189 [0,8807; 0,9520] / 0,6611 [0,6407; 0,6824] |
| Público v7 (primario) | 0,6937 [0,6333; 0,7510] / 0,8098 [0,7919; 0,8274] | 0,4730 [0,4101; 0,5369] / 0,9085 [0,8944; 0,9214] | 0,9550 [0,9269; 0,9804] / 0,3098 [0,2873; 0,3322] |
| Público, su mejor protocolo | 0,7387 [0,6809; 0,7957] / 0,8002 [0,7816; 0,8174] | 0,6351 [0,5695; 0,6974] / 0,8973 [0,8825; 0,9105] | 0,9279 [0,8930; 0,9615] / 0,5443 [0,5205; 0,5671] |

## Contrastes pareados (bootstrap, semilla 20260923)

| Contraste | Δ Macro-F1 | Δ Bal-acc | Δ AUC | Δ ECE | Δ Sens. mel | Δ AUC mel |
|---|---|---|---|---|---|---|
| Modelo de control − Público, protocolo de Modelo de control | 0,0909 [0,0499; 0,1321] | 0,1131 [0,0728; 0,1553] | 0,0212 [0,0108; 0,0303] | -0,0651 [-0,0770; -0,0414] | 0,1261 [0,0524; 0,1949] | 0,0709 [0,0488; 0,0929] |
| Proposed sin segmentación − Público, protocolo de Proposed sin segmentación | 0,0304 [-0,0074; 0,0686] | 0,0305 [-0,0097; 0,0712] | -0,0018 [-0,0081; 0,0043] | -0,0159 [-0,0296; -0,0008] | 0,0180 [-0,0417; 0,0833] | 0,0090 [-0,0084; 0,0255] |
| Modelo de control − Modelo de control, protocolo v7 (primario) | 0,0765 [0,0362; 0,1215] | 0,1284 [0,0864; 0,1737] | 0,0025 [-0,0045; 0,0084] | 0,0019 [-0,0147; 0,0138] | 0,0991 [0,0424; 0,1568] | 0,0181 [0,0053; 0,0305] |
| Proposed sin segmentación − Proposed sin segmentación, protocolo v7 (primario) | 0,0535 [0,0105; 0,0913] | 0,0637 [0,0166; 0,1054] | -0,0028 [-0,0090; 0,0030] | 0,0204 [0,0032; 0,0322] | 0,0270 [-0,0348; 0,0880] | -0,0079 [-0,0220; 0,0072] |
| Modelo de control − Público v7 (primario) | 0,1071 [0,0675; 0,1475] | 0,1381 [0,0976; 0,1792] | 0,0387 [0,0279; 0,0493] | -0,1153 [-0,1266; -0,0878] | 0,1577 [0,0829; 0,2286] | 0,0836 [0,0571; 0,1086] |
| Proposed sin segmentación − Público v7 (primario) | 0,0983 [0,0568; 0,1435] | 0,1040 [0,0597; 0,1503] | 0,0344 [0,0244; 0,0451] | -0,0779 [-0,0935; -0,0581] | 0,1036 [0,0201; 0,1765] | 0,0689 [0,0406; 0,0960] |
| Proposed sin segmentación − Modelo de control | -0,0088 [-0,0457; 0,0270] | -0,0341 [-0,0752; 0,0064] | -0,0042 [-0,0115; 0,0037] | 0,0374 [0,0156; 0,0490] | -0,0541 [-0,1149; 0,0093] | -0,0146 [-0,0300; -0,0013] |
| Público, su mejor protocolo − Público v7 (primario) | 0,0699 [0,0324; 0,1062] | 0,0699 [0,0312; 0,1088] | 0,0341 [0,0246; 0,0446] | -0,0605 [-0,0749; -0,0430] | 0,0811 [0,0092; 0,1508] | 0,0559 [0,0332; 0,0788] |

Fase 2 (docs/29 §2.7 ítem 7): sensibilidad de mel a especificidad 80 % en el holdout: A 0,8764, B 0,8652. Habilitada para proponer: no.

## Contrastes adicionales (`extra_contrasts.json`)

'mejora' solo si el IC pareado excluye el 0; nunca 'gana' si el IC toca el 0.

### Melanoma a umbral propio (comparación justa)

Cada lectura se evalua con el probe entrenado en el 80 % del fit y con SU PROPIO umbral de holdout (el protocolo publicado); el contraste pareado usa el mismo remuestreo para ambas.

| Contraste | Métrica | Brazo | Control | Δ | IC 95 % | Excluye 0 |
|---|---|---|---|---|---|---|
| Modelo de control − Público, protocolo de Modelo de control | Sens. mel (esp. 80 %) | 0,8604 | 0,6937 | 0,1667 | [0,1045; 0,2271] | sí |
| Modelo de control − Público, protocolo de Modelo de control | Sens. mel (esp. 90 %) | 0,6892 | 0,5450 | 0,1441 | [0,0762; 0,2159] | sí |
| Modelo de control − Público, protocolo de Modelo de control | Esp. mel (sens. 95 %) | 0,6336 | 0,3378 | 0,2957 | [0,2714; 0,3192] | sí |
| Proposed sin segmentación − Público, protocolo de Proposed sin segmentación | Sens. mel (esp. 80 %) | 0,8333 | 0,7477 | 0,0856 | [0,0362; 0,1373] | sí |
| Proposed sin segmentación − Público, protocolo de Proposed sin segmentación | Sens. mel (esp. 90 %) | 0,6757 | 0,6577 | 0,0180 | [-0,0337; 0,0708] | no |
| Proposed sin segmentación − Público, protocolo de Proposed sin segmentación | Esp. mel (sens. 95 %) | 0,6723 | 0,5466 | 0,1257 | [0,1026; 0,1468] | sí |
| Modelo de control − Público, su mejor protocolo | Sens. mel (esp. 80 %) | 0,8604 | 0,7387 | 0,1216 | [0,0703; 0,1734] | sí |
| Modelo de control − Público, su mejor protocolo | Sens. mel (esp. 90 %) | 0,6892 | 0,6351 | 0,0541 | [0,0000; 0,1128] | no |
| Modelo de control − Público, su mejor protocolo | Esp. mel (sens. 95 %) | 0,6336 | 0,5443 | 0,0892 | [0,0676; 0,1124] | sí |
| Proposed sin segmentación − Público, su mejor protocolo | Sens. mel (esp. 80 %) | 0,8333 | 0,7387 | 0,0946 | [0,0482; 0,1467] | sí |
| Proposed sin segmentación − Público, su mejor protocolo | Sens. mel (esp. 90 %) | 0,6757 | 0,6351 | 0,0405 | [-0,0139; 0,1000] | no |
| Proposed sin segmentación − Público, su mejor protocolo | Esp. mel (sens. 95 %) | 0,6723 | 0,5443 | 0,1279 | [0,1062; 0,1496] | sí |

### Melanomas clasificados como nevus

Melanomas verdaderos cuyo argmax es nv (probe del train completo), sobre el total de mel verdaderos. Menor es mejor: un delta negativo favorece al brazo.

| Lectura | mel→nv | Tasa |
|---|---|---|
| Modelo de control | 62 | 0,2793 |
| Proposed sin segmentación | 67 | 0,3018 |
| Público, protocolo de Modelo de control | 76 | 0,3423 |
| Público, protocolo de Proposed sin segmentación | 67 | 0,3018 |
| Modelo de control, protocolo v7 (primario) | 86 | 0,3874 |
| Proposed sin segmentación, protocolo v7 (primario) | 75 | 0,3378 |
| Público v7 (primario) | 87 | 0,3919 |
| Público, su mejor protocolo | 68 | 0,3063 |
| Modelo de control, protocolo v7 repetición 43 | 90 | 0,4054 |
| Proposed sin segmentación, protocolo v7 repetición 43 | 98 | 0,4414 |

| Contraste | Métrica | Brazo | Control | Δ | IC 95 % | Excluye 0 |
|---|---|---|---|---|---|---|
| Modelo de control − Público, protocolo de Modelo de control | Tasa mel→nv | 0,2793 | 0,3423 | -0,0631 | [-0,1293; -0,0045] | sí |
| Proposed sin segmentación − Público, protocolo de Proposed sin segmentación | Tasa mel→nv | 0,3018 | 0,3018 | 0,0000 | [-0,0496; 0,0521] | no |
| Modelo de control − Modelo de control, protocolo v7 (primario) | Tasa mel→nv | 0,2793 | 0,3874 | -0,1081 | [-0,1650; -0,0538] | sí |
| Proposed sin segmentación − Proposed sin segmentación, protocolo v7 (primario) | Tasa mel→nv | 0,3018 | 0,3378 | -0,0360 | [-0,0923; 0,0197] | no |

### Contra los primarios v7 repetición 42 y v7 repetición 43

Modelo de control es la sopa de los pesos repetición 42 y repetición 43, asi que su contraste contra Modelo de control, protocolo v7 repetición 43 no es independiente de repetición 43; Proposed sin segmentación usa solo repetición 42. Las filas contra A_v7/B_v7 son repetición 42 y repiten las publicadas.

| Contraste | Métrica | Brazo | Control | Δ | IC 95 % | Excluye 0 |
|---|---|---|---|---|---|---|
| Modelo de control − Modelo de control, protocolo v7 (primario) | Macro-F1 | 0,7229 | 0,6464 | 0,0765 | [0,0362; 0,1215] | sí |
| Modelo de control − Modelo de control, protocolo v7 (primario) | Bal-acc | 0,7305 | 0,6021 | 0,1284 | [0,0864; 0,1737] | sí |
| Modelo de control − Modelo de control, protocolo v7 repetición 43 | Macro-F1 | 0,7229 | 0,6850 | 0,0378 | [0,0023; 0,0761] | sí |
| Modelo de control − Modelo de control, protocolo v7 repetición 43 | Bal-acc | 0,7305 | 0,6457 | 0,0848 | [0,0471; 0,1231] | sí |
| Modelo de control − Proposed sin segmentación, protocolo v7 repetición 43 | Macro-F1 | 0,7229 | 0,6638 | 0,0591 | [0,0200; 0,1009] | sí |
| Modelo de control − Proposed sin segmentación, protocolo v7 repetición 43 | Bal-acc | 0,7305 | 0,6119 | 0,1186 | [0,0759; 0,1612] | sí |
| Proposed sin segmentación − Proposed sin segmentación, protocolo v7 (primario) | Macro-F1 | 0,7141 | 0,6606 | 0,0535 | [0,0105; 0,0913] | sí |
| Proposed sin segmentación − Proposed sin segmentación, protocolo v7 (primario) | Bal-acc | 0,6964 | 0,6327 | 0,0637 | [0,0166; 0,1054] | sí |
| Proposed sin segmentación − Proposed sin segmentación, protocolo v7 repetición 43 | Macro-F1 | 0,7141 | 0,6638 | 0,0503 | [0,0051; 0,0903] | sí |
| Proposed sin segmentación − Proposed sin segmentación, protocolo v7 repetición 43 | Bal-acc | 0,6964 | 0,6119 | 0,0845 | [0,0379; 0,1281] | sí |
| Proposed sin segmentación − Modelo de control, protocolo v7 repetición 43 | Macro-F1 | 0,7141 | 0,6850 | 0,0291 | [-0,0104; 0,0673] | no |
| Proposed sin segmentación − Modelo de control, protocolo v7 repetición 43 | Bal-acc | 0,6964 | 0,6457 | 0,0507 | [0,0130; 0,0901] | sí |

## Limitación: sensibilidad del probe a los hilos BLAS

Limitación (no errata): el probe lineal tiene una sensibilidad de implementación de hasta 0,014 de macro-F1 según el número de hilos BLAS por proceso (4 vs 8 hilos; probe_jobs 10 vs 6; cambian 48 de 60 celdas en cada una de las 3 unidades, con hasta 18 imágenes que cambian de clase predicha por celda); con la configuración registrada (4 hilos) reproduce la grilla publicada bit a bit (máx. |Δ| 0). El efecto SSL (Modelo de control − público con el mismo protocolo, macro-F1 +0,091 [+0,050; +0,132]) es ~6,5× esa sensibilidad. No es falta de convergencia: los 30 ajustes publicados convergen por tolerancia (0/30 alcanzan max_iter=2000; n_iter 165–862); la sensibilidad es de redondeo en punto flotante dentro de la tolerancia del optimizador. En Prueba de constancia de color el umbral de Predicción de anclaje (≤ 0,01) está por debajo de esa sensibilidad, así que Predicción de anclaje solo se lee pareado dentro de la misma corrida con hilos fijos. Evidencia: commits 0a691eb (reproducción) y 278efd2 (convergencia).

Generado desde `resultados/clasificacion/t4/diagnostico_reproduccion/identity_j10_vs_j10.json`, `identity_j10_vs_j6.json`, `t4/diagnostico_convergencia/probe_convergence.json`, el contraste pareado `A71 − public_A71` de `v71_final_eval.json` y `t4_sog.P1_MAX_ABS_DELTA`; no editar a mano.
