# T3 — C1-v7 por tercil de luminancia (exploratorio)

Generado por `scripts/t3_c1_luminance.py` desde `t3_c1_luminance.json`; no editar a mano. Registro, predicción y salvedades: docs/28, bloque T3.

Los bins son terciles de la mediana de L\* de la piel sobre el pool (`tone_tercile`, manifiesto sha256 `57df1e65e88d…`): luminancia, no es tono de piel. Está confundida con el diagnóstico, el dermatoscopio y la iluminación; ningún Δ se atribuye al tono.

Δ = métrica(condición) − métrica(`none`), bootstrap pareado por imagen dentro del tercil, 1000 remuestreos, semilla 20260923, IC percentil 95 %. Con n < 100: sin potencia.

Cobertura de val: oscuro 820, medio 957, claro 183, sin tercil 44, fuera del manifiesto 0, total 2004.

## Predicción de Paula: **cumplida**

Celdas clave: 12. Mejoran: ninguna. Robustas: ninguna. Empeoran: 0.

Multiplicidad: 270 contrastes en total, unos 13.5 IC sin cero esperables por azar; en las 12 celdas clave, unas 0.3 «mejora» por azar.

## Celdas clave (tercil oscuro)

| Segmentador | Backbone | Tercil | Condición | n | n mel | Δ macro-F1 [IC 95 %] | Δ bal-acc [IC 95 %] | Δ sens. mel [IC 95 %] |
|---|---|---|---|---:|---:|---|---|---|
| student_r3 | dinov2_public_vitb14 | oscuro | bbox_crop | 820 | 99 | -0,0242 [-0,1102; +0,0608] indistinguible | -0,0476 [-0,1417; +0,0481] indistinguible | +0,1010 [+0,0000; +0,2062] indistinguible |
| student_r3 | dinov2_public_vitb14 | oscuro | mask_pool | 820 | 99 | +0,0246 [-0,0280; +0,0784] indistinguible | +0,0060 [-0,0465; +0,0573] indistinguible | +0,0606 [-0,0471; +0,1786] indistinguible |
| student_r3 | v7_astar_s42 | oscuro | bbox_crop | 820 | 99 | +0,0034 [-0,0718; +0,0788] indistinguible | -0,0019 [-0,0737; +0,0694] indistinguible | +0,0202 [-0,0809; +0,1250] indistinguible |
| student_r3 | v7_astar_s42 | oscuro | mask_pool | 820 | 99 | +0,0194 [-0,0515; +0,0863] indistinguible | +0,0190 [-0,0458; +0,0843] indistinguible | -0,0606 [-0,1294; +0,0002] indistinguible |
| student_r3 | v7_astar_s43 | oscuro | bbox_crop | 820 | 99 | -0,0220 [-0,0970; +0,0431] indistinguible | -0,0049 [-0,0831; +0,0764] indistinguible | +0,0202 [-0,0544; +0,0938] indistinguible |
| student_r3 | v7_astar_s43 | oscuro | mask_pool | 820 | 99 | -0,0221 [-0,0717; +0,0269] indistinguible | -0,0235 [-0,0729; +0,0314] indistinguible | +0,0303 [-0,0495; +0,1075] indistinguible |
| a2 | dinov2_public_vitb14 | oscuro | bbox_crop | 820 | 99 | -0,0499 [-0,1293; +0,0285] indistinguible | -0,0757 [-0,1661; +0,0140] indistinguible | +0,0303 [-0,0599; +0,1300] indistinguible |
| a2 | dinov2_public_vitb14 | oscuro | mask_pool | 820 | 99 | +0,0261 [-0,0333; +0,0847] indistinguible | -0,0098 [-0,0682; +0,0451] indistinguible | +0,0707 [-0,0342; +0,1744] indistinguible |
| a2 | v7_astar_s42 | oscuro | bbox_crop | 820 | 99 | +0,0205 [-0,0437; +0,0924] indistinguible | +0,0206 [-0,0447; +0,0916] indistinguible | +0,0101 [-0,0883; +0,1098] indistinguible |
| a2 | v7_astar_s42 | oscuro | mask_pool | 820 | 99 | +0,0357 [-0,0337; +0,0999] indistinguible | +0,0299 [-0,0310; +0,0953] indistinguible | -0,0505 [-0,1177; +0,0112] indistinguible |
| a2 | v7_astar_s43 | oscuro | bbox_crop | 820 | 99 | -0,0366 [-0,1199; +0,0395] indistinguible | -0,0331 [-0,1155; +0,0586] indistinguible | +0,0000 [-0,0889; +0,0870] indistinguible |
| a2 | v7_astar_s43 | oscuro | mask_pool | 820 | 99 | -0,0123 [-0,0661; +0,0427] indistinguible | -0,0083 [-0,0631; +0,0533] indistinguible | +0,0505 [-0,0323; +0,1346] indistinguible |

## Descriptivos (sin veredicto de predicción)

| Segmentador | Backbone | Tercil | Condición | n | n mel | Δ macro-F1 [IC 95 %] | Δ bal-acc [IC 95 %] | Δ sens. mel [IC 95 %] |
|---|---|---|---|---:|---:|---|---|---|
| student_r3 | dinov2_public_vitb14 | oscuro | neutral_bg | 820 | 99 | -0,0500 [-0,1433; +0,0512] indistinguible | -0,0870 [-0,1928; +0,0285] indistinguible | +0,0404 [-0,0706; +0,1613] indistinguible |
| student_r3 | dinov2_public_vitb14 | medio | bbox_crop | 957 | 82 | -0,0097 [-0,0879; +0,0690] indistinguible | -0,0160 [-0,0907; +0,0611] indistinguible | -0,0732 [-0,1974; +0,0494] indistinguible |
| student_r3 | dinov2_public_vitb14 | medio | neutral_bg | 957 | 82 | -0,1000 [-0,1718; -0,0268] empeora | -0,0754 [-0,1464; -0,0083] empeora | -0,0488 [-0,1867; +0,0824] indistinguible |
| student_r3 | dinov2_public_vitb14 | medio | mask_pool | 957 | 82 | +0,0197 [-0,0511; +0,0908] indistinguible | +0,0260 [-0,0410; +0,0936] indistinguible | -0,0854 [-0,2055; +0,0256] indistinguible |
| student_r3 | dinov2_public_vitb14 | claro | bbox_crop | 183 | 35 | -0,0440 [-0,1704; +0,1063] indistinguible | -0,0368 [-0,1829; +0,0969] indistinguible | +0,0571 [-0,1379; +0,2446] indistinguible |
| student_r3 | dinov2_public_vitb14 | claro | neutral_bg | 183 | 35 | -0,1062 [-0,2708; +0,0669] indistinguible | -0,0757 [-0,2547; +0,0898] indistinguible | -0,1714 [-0,3637; +0,0000] indistinguible |
| student_r3 | dinov2_public_vitb14 | claro | mask_pool | 183 | 35 | -0,0089 [-0,1350; +0,1377] indistinguible | -0,0382 [-0,1642; +0,0929] indistinguible | -0,0286 [-0,2000; +0,1380] indistinguible |
| student_r3 | v7_astar_s42 | oscuro | neutral_bg | 820 | 99 | -0,0188 [-0,1112; +0,0768] indistinguible | -0,0218 [-0,1156; +0,0774] indistinguible | +0,0101 [-0,0714; +0,0938] indistinguible |
| student_r3 | v7_astar_s42 | medio | bbox_crop | 957 | 82 | -0,0156 [-0,0778; +0,0450] indistinguible | -0,0200 [-0,0776; +0,0391] indistinguible | -0,0366 [-0,1216; +0,0520] indistinguible |
| student_r3 | v7_astar_s42 | medio | neutral_bg | 957 | 82 | -0,0870 [-0,1394; -0,0325] empeora | -0,0791 [-0,1299; -0,0288] empeora | -0,1220 [-0,2180; -0,0260] empeora |
| student_r3 | v7_astar_s42 | medio | mask_pool | 957 | 82 | -0,0118 [-0,0670; +0,0493] indistinguible | -0,0131 [-0,0695; +0,0471] indistinguible | -0,0854 [-0,1618; -0,0169] empeora |
| student_r3 | v7_astar_s42 | claro | bbox_crop | 183 | 35 | -0,0055 [-0,0518; +0,0389] indistinguible | +0,0234 [-0,0183; +0,0740] indistinguible | +0,0571 [+0,0000; +0,1464] indistinguible |
| student_r3 | v7_astar_s42 | claro | neutral_bg | 183 | 35 | +0,0939 [-0,0224; +0,2403] indistinguible | +0,1218 [-0,0028; +0,2572] indistinguible | +0,1714 [+0,0526; +0,3104] mejora |
| student_r3 | v7_astar_s42 | claro | mask_pool | 183 | 35 | +0,0151 [-0,0206; +0,0536] indistinguible | +0,0378 [+0,0002; +0,0821] mejora | +0,0000 [-0,0833; +0,0833] indistinguible |
| student_r3 | v7_astar_s43 | oscuro | neutral_bg | 820 | 99 | -0,0535 [-0,1275; +0,0304] indistinguible | -0,0605 [-0,1402; +0,0329] indistinguible | +0,0808 [-0,0171; +0,1837] indistinguible |
| student_r3 | v7_astar_s43 | medio | bbox_crop | 957 | 82 | -0,0094 [-0,0713; +0,0487] indistinguible | -0,0170 [-0,0750; +0,0409] indistinguible | +0,0244 [-0,0641; +0,1098] indistinguible |
| student_r3 | v7_astar_s43 | medio | neutral_bg | 957 | 82 | -0,0752 [-0,1323; -0,0202] empeora | -0,0650 [-0,1210; -0,0147] empeora | -0,1463 [-0,2501; -0,0361] empeora |
| student_r3 | v7_astar_s43 | medio | mask_pool | 957 | 82 | -0,0121 [-0,0640; +0,0391] indistinguible | +0,0044 [-0,0436; +0,0491] indistinguible | +0,0488 [-0,0695; +0,1609] indistinguible |
| student_r3 | v7_astar_s43 | claro | bbox_crop | 183 | 35 | +0,0453 [-0,0282; +0,1697] indistinguible | +0,0379 [-0,0599; +0,1596] indistinguible | +0,0571 [-0,0690; +0,2000] indistinguible |
| student_r3 | v7_astar_s43 | claro | neutral_bg | 183 | 35 | +0,0907 [-0,0463; +0,2209] indistinguible | +0,0889 [-0,0325; +0,2214] indistinguible | +0,0286 [-0,1250; +0,1875] indistinguible |
| student_r3 | v7_astar_s43 | claro | mask_pool | 183 | 35 | -0,0153 [-0,0855; +0,0434] indistinguible | -0,0465 [-0,1111; +0,0114] indistinguible | -0,0571 [-0,2000; +0,0857] indistinguible |
| student_r3 | v7_bstar_s42 | oscuro | bbox_crop | 820 | 99 | +0,0464 [-0,0207; +0,1198] indistinguible | +0,0369 [-0,0427; +0,1205] indistinguible | +0,0404 [-0,0488; +0,1358] indistinguible |
| student_r3 | v7_bstar_s42 | oscuro | neutral_bg | 820 | 99 | +0,0010 [-0,0728; +0,0792] indistinguible | -0,0125 [-0,0873; +0,0672] indistinguible | +0,0505 [-0,0691; +0,1667] indistinguible |
| student_r3 | v7_bstar_s42 | oscuro | mask_pool | 820 | 99 | -0,0209 [-0,0647; +0,0152] indistinguible | -0,0298 [-0,0760; +0,0073] indistinguible | -0,0707 [-0,1429; +0,0000] indistinguible |
| student_r3 | v7_bstar_s42 | medio | bbox_crop | 957 | 82 | +0,0152 [-0,0389; +0,0738] indistinguible | +0,0202 [-0,0353; +0,0841] indistinguible | -0,1098 [-0,2262; +0,0000] indistinguible |
| student_r3 | v7_bstar_s42 | medio | neutral_bg | 957 | 82 | -0,0261 [-0,0810; +0,0300] indistinguible | -0,0314 [-0,0840; +0,0234] indistinguible | -0,1098 [-0,2273; +0,0225] indistinguible |
| student_r3 | v7_bstar_s42 | medio | mask_pool | 957 | 82 | -0,0261 [-0,0650; +0,0188] indistinguible | -0,0239 [-0,0641; +0,0183] indistinguible | -0,0244 [-0,1333; +0,0834] indistinguible |
| student_r3 | v7_bstar_s42 | claro | bbox_crop | 183 | 35 | -0,0779 [-0,1766; +0,0215] indistinguible | -0,0611 [-0,1571; +0,0095] indistinguible | +0,0000 [-0,1430; +0,1471] indistinguible |
| student_r3 | v7_bstar_s42 | claro | neutral_bg | 183 | 35 | +0,0130 [-0,0706; +0,1185] indistinguible | +0,0116 [-0,0696; +0,1147] indistinguible | +0,0286 [-0,1463; +0,2059] indistinguible |
| student_r3 | v7_bstar_s42 | claro | mask_pool | 183 | 35 | -0,0295 [-0,1666; +0,1064] indistinguible | -0,0623 [-0,1633; +0,0727] indistinguible | -0,0286 [-0,1667; +0,0969] indistinguible |
| student_r3 | v7_bstar_s43 | oscuro | bbox_crop | 820 | 99 | +0,0209 [-0,0400; +0,0814] indistinguible | +0,0224 [-0,0441; +0,0920] indistinguible | +0,1010 [+0,0000; +0,1864] indistinguible |
| student_r3 | v7_bstar_s43 | oscuro | neutral_bg | 820 | 99 | +0,0004 [-0,0738; +0,0764] indistinguible | -0,0029 [-0,0743; +0,0747] indistinguible | +0,0707 [-0,0435; +0,1868] indistinguible |
| student_r3 | v7_bstar_s43 | oscuro | mask_pool | 820 | 99 | -0,0113 [-0,0650; +0,0411] indistinguible | -0,0058 [-0,0557; +0,0444] indistinguible | +0,1010 [+0,0000; +0,1900] indistinguible |
| student_r3 | v7_bstar_s43 | medio | bbox_crop | 957 | 82 | -0,0263 [-0,0802; +0,0283] indistinguible | -0,0183 [-0,0713; +0,0328] indistinguible | -0,0122 [-0,1000; +0,0732] indistinguible |
| student_r3 | v7_bstar_s43 | medio | neutral_bg | 957 | 82 | -0,0347 [-0,0848; +0,0132] indistinguible | -0,0133 [-0,0671; +0,0353] indistinguible | -0,0610 [-0,1628; +0,0362] indistinguible |
| student_r3 | v7_bstar_s43 | medio | mask_pool | 957 | 82 | -0,0117 [-0,0517; +0,0308] indistinguible | -0,0009 [-0,0398; +0,0396] indistinguible | -0,0122 [-0,1139; +0,0886] indistinguible |
| student_r3 | v7_bstar_s43 | claro | bbox_crop | 183 | 35 | +0,0285 [-0,0529; +0,1602] indistinguible | +0,0706 [+0,0051; +0,1716] mejora | +0,1429 [+0,0323; +0,2759] mejora |
| student_r3 | v7_bstar_s43 | claro | neutral_bg | 183 | 35 | +0,1116 [-0,0089; +0,2485] indistinguible | +0,1257 [-0,0031; +0,2523] indistinguible | +0,2000 [+0,0333; +0,3794] mejora |
| student_r3 | v7_bstar_s43 | claro | mask_pool | 183 | 35 | +0,0809 [-0,0168; +0,2135] indistinguible | +0,0908 [-0,0163; +0,2194] indistinguible | +0,1429 [+0,0312; +0,2647] mejora |
| a2 | dinov2_public_vitb14 | oscuro | neutral_bg | 820 | 99 | -0,0791 [-0,1837; +0,0157] indistinguible | -0,1223 [-0,2367; -0,0140] empeora | +0,0606 [-0,0516; +0,1721] indistinguible |
| a2 | dinov2_public_vitb14 | medio | bbox_crop | 957 | 82 | -0,0040 [-0,0832; +0,0770] indistinguible | +0,0008 [-0,0728; +0,0727] indistinguible | +0,0000 [-0,1177; +0,1268] indistinguible |
| a2 | dinov2_public_vitb14 | medio | neutral_bg | 957 | 82 | -0,0795 [-0,1499; -0,0046] empeora | -0,0553 [-0,1296; +0,0119] indistinguible | -0,0976 [-0,2470; +0,0506] indistinguible |
| a2 | dinov2_public_vitb14 | medio | mask_pool | 957 | 82 | +0,0139 [-0,0595; +0,0868] indistinguible | +0,0266 [-0,0415; +0,0958] indistinguible | -0,0488 [-0,1691; +0,0639] indistinguible |
| a2 | dinov2_public_vitb14 | claro | bbox_crop | 183 | 35 | +0,0157 [-0,1128; +0,1501] indistinguible | +0,0043 [-0,1347; +0,1172] indistinguible | -0,0571 [-0,2176; +0,1053] indistinguible |
| a2 | dinov2_public_vitb14 | claro | neutral_bg | 183 | 35 | -0,0599 [-0,1810; +0,0757] indistinguible | -0,0702 [-0,1933; +0,0583] indistinguible | -0,2000 [-0,3871; -0,0303] empeora |
| a2 | dinov2_public_vitb14 | claro | mask_pool | 183 | 35 | +0,0173 [-0,1217; +0,1688] indistinguible | -0,0275 [-0,1549; +0,1068] indistinguible | +0,0286 [-0,1740; +0,2334] indistinguible |
| a2 | v7_astar_s42 | oscuro | neutral_bg | 820 | 99 | +0,0163 [-0,0770; +0,1164] indistinguible | -0,0002 [-0,0986; +0,1012] indistinguible | -0,0303 [-0,1225; +0,0612] indistinguible |
| a2 | v7_astar_s42 | medio | bbox_crop | 957 | 82 | -0,0141 [-0,0805; +0,0561] indistinguible | -0,0056 [-0,0658; +0,0599] indistinguible | -0,0610 [-0,1368; +0,0111] indistinguible |
| a2 | v7_astar_s42 | medio | neutral_bg | 957 | 82 | -0,0467 [-0,1069; +0,0195] indistinguible | -0,0439 [-0,1022; +0,0243] indistinguible | -0,0976 [-0,2111; +0,0149] indistinguible |
| a2 | v7_astar_s42 | medio | mask_pool | 957 | 82 | -0,0147 [-0,0695; +0,0416] indistinguible | -0,0101 [-0,0679; +0,0476] indistinguible | -0,0610 [-0,1268; +0,0000] indistinguible |
| a2 | v7_astar_s42 | claro | bbox_crop | 183 | 35 | +0,0455 [-0,0383; +0,1261] indistinguible | +0,0441 [-0,0151; +0,1309] indistinguible | +0,0857 [+0,0000; +0,1843] indistinguible |
| a2 | v7_astar_s42 | claro | neutral_bg | 183 | 35 | +0,0401 [-0,1387; +0,2151] indistinguible | +0,0581 [-0,1178; +0,2311] indistinguible | +0,0571 [-0,0976; +0,2123] indistinguible |
| a2 | v7_astar_s42 | claro | mask_pool | 183 | 35 | +0,0007 [-0,0377; +0,0407] indistinguible | +0,0277 [-0,0068; +0,0719] indistinguible | +0,0000 [-0,0833; +0,0833] indistinguible |
| a2 | v7_astar_s43 | oscuro | neutral_bg | 820 | 99 | -0,0526 [-0,1311; +0,0347] indistinguible | -0,0693 [-0,1525; +0,0297] indistinguible | +0,0606 [-0,0300; +0,1505] indistinguible |
| a2 | v7_astar_s43 | medio | bbox_crop | 957 | 82 | +0,0059 [-0,0421; +0,0574] indistinguible | +0,0084 [-0,0352; +0,0557] indistinguible | +0,0000 [-0,1053; +0,1045] indistinguible |
| a2 | v7_astar_s43 | medio | neutral_bg | 957 | 82 | -0,0314 [-0,0883; +0,0233] indistinguible | -0,0235 [-0,0742; +0,0282] indistinguible | -0,0122 [-0,1194; +0,0946] indistinguible |
| a2 | v7_astar_s43 | medio | mask_pool | 957 | 82 | -0,0094 [-0,0604; +0,0412] indistinguible | +0,0073 [-0,0391; +0,0539] indistinguible | +0,0610 [-0,0366; +0,1625] indistinguible |
| a2 | v7_astar_s43 | claro | bbox_crop | 183 | 35 | +0,0823 [-0,0305; +0,2196] indistinguible | +0,0629 [-0,0376; +0,1979] indistinguible | +0,0000 [-0,1333; +0,1389] indistinguible |
| a2 | v7_astar_s43 | claro | neutral_bg | 183 | 35 | +0,0176 [-0,1001; +0,1559] indistinguible | +0,0294 [-0,0828; +0,1612] indistinguible | -0,0571 [-0,2188; +0,1154] indistinguible |
| a2 | v7_astar_s43 | claro | mask_pool | 183 | 35 | +0,0012 [-0,0745; +0,1322] indistinguible | -0,0113 [-0,1020; +0,1076] indistinguible | -0,0571 [-0,1936; +0,0769] indistinguible |
| a2 | v7_bstar_s42 | oscuro | bbox_crop | 820 | 99 | +0,0266 [-0,0626; +0,1096] indistinguible | +0,0207 [-0,0716; +0,1073] indistinguible | +0,0101 [-0,0879; +0,1019] indistinguible |
| a2 | v7_bstar_s42 | oscuro | neutral_bg | 820 | 99 | -0,0219 [-0,0896; +0,0469] indistinguible | -0,0275 [-0,0989; +0,0485] indistinguible | +0,0505 [-0,0715; +0,1667] indistinguible |
| a2 | v7_bstar_s42 | oscuro | mask_pool | 820 | 99 | -0,0164 [-0,0645; +0,0216] indistinguible | -0,0281 [-0,0820; +0,0152] indistinguible | -0,0303 [-0,1154; +0,0404] indistinguible |
| a2 | v7_bstar_s42 | medio | bbox_crop | 957 | 82 | +0,0389 [-0,0079; +0,0945] indistinguible | +0,0278 [-0,0227; +0,0879] indistinguible | -0,0244 [-0,1370; +0,0737] indistinguible |
| a2 | v7_bstar_s42 | medio | neutral_bg | 957 | 82 | -0,0075 [-0,0625; +0,0586] indistinguible | -0,0123 [-0,0680; +0,0519] indistinguible | -0,1220 [-0,2500; +0,0000] indistinguible |
| a2 | v7_bstar_s42 | medio | mask_pool | 957 | 82 | -0,0119 [-0,0543; +0,0360] indistinguible | -0,0016 [-0,0418; +0,0479] indistinguible | +0,0000 [-0,1081; +0,1067] indistinguible |
| a2 | v7_bstar_s42 | claro | bbox_crop | 183 | 35 | -0,0915 [-0,1824; +0,0027] indistinguible | -0,0577 [-0,1556; +0,0111] indistinguible | -0,0286 [-0,1765; +0,1251] indistinguible |
| a2 | v7_bstar_s42 | claro | neutral_bg | 183 | 35 | -0,0198 [-0,1566; +0,0938] indistinguible | -0,0373 [-0,1613; +0,0862] indistinguible | +0,0286 [-0,1176; +0,1795] indistinguible |
| a2 | v7_bstar_s42 | claro | mask_pool | 183 | 35 | -0,0267 [-0,1612; +0,1070] indistinguible | -0,0651 [-0,1645; +0,0716] indistinguible | -0,0571 [-0,1724; +0,0556] indistinguible |
| a2 | v7_bstar_s43 | oscuro | bbox_crop | 820 | 99 | +0,0536 [-0,0025; +0,1211] indistinguible | +0,0778 [+0,0209; +0,1478] mejora | +0,1010 [+0,0000; +0,1856] indistinguible |
| a2 | v7_bstar_s43 | oscuro | neutral_bg | 820 | 99 | +0,0148 [-0,0591; +0,0829] indistinguible | +0,0191 [-0,0561; +0,0940] indistinguible | +0,0101 [-0,0928; +0,1137] indistinguible |
| a2 | v7_bstar_s43 | oscuro | mask_pool | 820 | 99 | -0,0080 [-0,0641; +0,0433] indistinguible | -0,0047 [-0,0571; +0,0462] indistinguible | +0,0707 [-0,0291; +0,1600] indistinguible |
| a2 | v7_bstar_s43 | medio | bbox_crop | 957 | 82 | -0,0425 [-0,1015; +0,0173] indistinguible | -0,0282 [-0,0842; +0,0263] indistinguible | -0,0244 [-0,1205; +0,0706] indistinguible |
| a2 | v7_bstar_s43 | medio | neutral_bg | 957 | 82 | -0,0388 [-0,0944; +0,0158] indistinguible | -0,0202 [-0,0795; +0,0350] indistinguible | +0,0000 [-0,0923; +0,0909] indistinguible |
| a2 | v7_bstar_s43 | medio | mask_pool | 957 | 82 | -0,0133 [-0,0597; +0,0393] indistinguible | +0,0033 [-0,0469; +0,0560] indistinguible | -0,0366 [-0,1299; +0,0610] indistinguible |
| a2 | v7_bstar_s43 | claro | bbox_crop | 183 | 35 | -0,0245 [-0,0732; +0,0197] indistinguible | +0,0019 [-0,0232; +0,0262] indistinguible | +0,0571 [-0,0716; +0,1892] indistinguible |
| a2 | v7_bstar_s43 | claro | neutral_bg | 183 | 35 | +0,0533 [-0,0578; +0,1927] indistinguible | +0,0695 [-0,0468; +0,2057] indistinguible | +0,1714 [+0,0000; +0,3438] indistinguible |
| a2 | v7_bstar_s43 | claro | mask_pool | 183 | 35 | +0,0794 [-0,0217; +0,2069] indistinguible | +0,0876 [-0,0260; +0,2124] indistinguible | +0,0571 [-0,0370; +0,1667] indistinguible |
