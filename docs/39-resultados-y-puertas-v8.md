# Auditoría modular v8: ejecución y decisiones

Registro de ejecución 2026-10-11 UTC. Continúa docs/35 y protocolos 36–38 sin modificar los criterios después de observar métricas. W0/W1/W2 ejecutados; W3/W4 pendientes con puertas explícitas.

## Procedencia y conservación

Original del clúster: repo, rama feature/fitzpatrick-domain-adaptation-informe, HEAD f19ab56, pendientes pilot.log y wandb/ conservados. Copia anterior repo_audit_v8 en 9c23daf conservada. Trabajo en repo_audit_v8_run_20261011: base 9c23daf y snapshot completo de src/ más entradas necesarias verificadas contra f8b1790f8f6049415eccc92a066d696a48171e4e. No afirmar checkout completo de f8b1790. Módulos comunes adicionales y scripts transferidos con SHA256; checkpoints históricos leídos, no sobrescritos.

Salidas externas: salidas/auditoria_v8/{w0_paired_s42_20261011,w01_w2_s42_20261011,w2_s4_paired_20261011,w2_eval_isic2018_s44_20261011}. Imágenes y registros por ID permanecen en el clúster. Esta rama conserva código, protocolos y agregados numéricos.

## Pool verificado

| Etapa | Imágenes |
|---|---:|
| Pool nativo inicial | 58.457 |
| Cuarentena por área | 3.423 |
| Pares originales/transformadas elegibles | 55.034 |
| Exclusión histórica por ID | 14.128 |
| Duplicados adicionales por contenido | 10 |
| Pool SSL efectivo | 40.896 |

Por dataset, inicial/elegible/cuarentena: HAM p1 5000/4889/111; HAM p2 5015/4934/81; ISIC2019 15316/13781/1535; ISIC2020 33126/31430/1696. La cifra 40.896 corresponde al subconjunto SSL, no al pool inicial. La exclusión efectiva separada conserva el archivo histórico.

## W0: 100 ternas reales, semilla 42

Rutas originales, shift2 y máscaras r2 verificadas; orden de ejecución v1/tone/v2 remapeado al orden de retorno v1/v2/tone. Geometría v1/tone alineada, delta RGB de pseudolesión antes de aumentos cero en los 100 pares. Son nuevas tiradas reproducibles, no reconstrucción de workers históricos.

Retención media v1/tone 0,9056 y v2 0,9122; p10 0,5943 y 0,7240. Una vista v2 sin pseudolesión; un caso v1/tone retiene solo 2,58%. Contar ausencias completas subestima recortes severos. Los estratos por dataset, área, pelo y campo son descriptivos: pocos casos no permiten tasas poblacionales. No cambiar todavía aumentos: registrar como hipótesis para W4 una restricción de retención, separada de la ablación de tono y dominio.

## W1: región ITA

| Variante | ITA disponible /100 | Indeterminado | Anillo insuficiente |
|---|---:|---:|---:|
| I0 complemento histórico | 53 | 47 | 0 |
| I1 anillo, sin pelo excluido | 57 | 33 | 10 |
| I2 mismo anillo, pelo excluido | 57 | 33 | 10 |

I1/I2: 57 pares, cero categorías cambiadas; cambio máximo 1,061°. I0/I1: 49 pares, mediana de cambio absoluto 3,763°, máximo 50,073°, seis categorías cambiadas. La definición del ROI pesa más que excluir pelo en esta muestra. No comparar medias sobre distintos subconjuntos. No hay referencia de color para exactitud; ITA no demuestra fototipo Fitzpatrick. Medición siempre en RGB original, nunca inpainting. Los 100 casos de cuarentena siguen mask_ineligible.

## W2: semillas congeladas

100 IDs de W0 y 100 de cuarentena (semilla 43, 25/dataset); todos los índices, formas, hashes semánticos y correspondencias con exportación verificados. S0 reproduce exactamente los 200 slices históricos de pseudo_r0. S1 span bruto <=0,01 no cambia ninguno. S2 solo omite inversión; S3 solo excluye pelo; S4 vuelve a inferir el maestro con inpainting.

| Brazo | Aceptadas en entrenamiento /100 | Cuarentena /100 |
|---|---:|---:|
| S0 | 90 | 50 |
| S1 | 90 | 50 |
| S2 | 12 | 52 |
| S3 | 90 | 57 |
| S4 | 88 | 53 |

Aceptación no es exactitud anatómica. Ningún caso se reincorpora. S4 superó el control numérico original contra histórico en 200/200 mapas: MAE máxima por mapa 0,000831 y error máximo por píxel 0,005859. Checkpoints, precisión y reglas congelados. Cambios de semillas: 63/100 entrenamiento y 50/100 cuarentena.

## Evaluación exploratoria con referencia experta

ISIC2018 Task1 Training, 100 pares uniformes, semilla 44; predicciones terminadas antes de decodificar referencias. Todos los 100 coinciden por ID canónico con el pool nativo. Benchmark previamente consultado: no conjunto externo intacto, no confirmación de generalización. Coincidencia por ID no sustituye deduplicación por contenido. Métricas de semillas, no del estudiante r2/r3.

IGNORE cuenta como no-lesión sobre todo el frame; abstenciones cuentan como vacías. Dice/IoU incluyen los 100; HD95 indefinido se cuenta separado.

| Brazo | Dice medio | IoU medio | Vacías /100 | Delta Dice vs S0 | IC descriptivo 95% |
|---|---:|---:|---:|---:|---|
| S0 | 0,5501 | 0,4666 | 25 | — | — |
| S1 | 0,5501 | 0,4666 | 25 | 0 | [0; 0] |
| S2 | 0,0111 | 0,0060 | 86 | -0,5389 | [-0,6138; -0,4636] |
| S3 | 0,5409 | 0,4542 | 25 | -0,0091 | [-0,0125; -0,0063] |
| S4 | 0,5541 | 0,4701 | 24 | +0,0041 | [-0,0003; +0,0110] |

Bootstrap pareado de 4000 réplicas, semilla 45, percentiles. S4 no muestra una mejora concluyente. S3 reduce ligeramente el Dice; no adoptarlo por contar más semillas válidas. Mantener S0 como control; no promover S1/S2/S3/S4 a entrenamiento. La inversión está haciendo una función decisiva en este maestro; retirar S2 empeora la lectura local, pero no demuestra que la heurística siempre sea correcta. S1 queda inconcluso, sin mapas que disparen su umbral.

La primera escritura de records falló por np.float32 y np.int64 no serializables. Se corrigió solo serialización y se reanudó evaluación usando los 100 artefactos de predicción ya guardados; sin nueva inferencia ni selección. El hash del script de inferencia previo permanece en provenance.json; recuperación documentada y script corregido conservado.

## Verificación y revisión

W0: ocho pruebas y reporte --check aprobados. Pilotos: cinco pruebas sintéticas aprobadas. Reporte W01/S4: 200 IDs pareados, 800 máscaras 0/1/255 y 1400 archivos S4 verificados. Evaluación: 500 archivos de etiquetas con hashes, métricas binarias recalculadas y bootstrap recalculado. Las 20 láminas W2 y 10 láminas con referencia se revisaron visualmente; revisión ilustrativa, sin convertirla en anotaciones clínicas.

## W3 y W4: siguientes puertas concretas

W3 SLED: revisión oficial 3213eacb65577c0d52f424369158fb56c609238e inspeccionada (README, main, requirements y árbol). Dependencias históricas Python 3.10.6/Java8/JPype/PyOD; clúster actual Python3.13, sin ejecutables java/python3.10 ni paquetes jpype/pyod en la venv usada. No se instaló en el entorno histórico. No se observó archivo de licencia en ese árbol: no redistribuir fuente de terceros hasta resolver procedencia/licencia. No hay piloto SLED ni espectral ejecutado en esta entrega.

Preparar entorno separado y revisión/pesos fijados para W3; reutilizar selección sin máscaras expertas como input. Medir semillas S0 y maestros independientes antes del mismo estudiante. Una evaluación independiente requiere manifiesto fuera del pool (verificación por ID y contenido), protocolo/splits congelados y referencias solo para medir. La muestra actual no satisface esa puerta.

W4 sigue el factorial de docs/35: original-original, original-transformada y cada una con tercera vista de tono. Igualar inicialización, pool/exclusiones, pasos y semillas; reportar costo de tercera vista. Dominio en contraste separado: público intacto, continuación fuente y continuación fuente+target_adapt; nunca target_test. No se lanzó nuevo SSL ni estudiante. No gastar ese bloque de entrenamiento antes de cerrar correspondencia, calidad de semillas y evaluación independiente.

Seguimos con la secuencia registrada, con estas decisiones por componente. No ampliar a una búsqueda indiscriminada ni ajustar umbrales con este benchmark.
