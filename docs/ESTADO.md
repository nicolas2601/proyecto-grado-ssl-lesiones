# ESTADO — avance contra los cuatro objetivos específicos

Fuente de verdad del avance entre sesiones (ver [SESIONES.md](SESIONES.md)). Lo escribe la
sesión maestra. No se edita desde una sesión trabajadora.

- **Fecha de corte:** 2026-10-30
- **Rama:** `experiment/modular-audit-v8`
- **HEAD:** `2daa4ed` (2026-10-27) — *feat(v8): add flat-lr flag for controlled single-factor backbone ablation*
- **Árbol de trabajo:** 67 archivos modificados, 1 borrado, 379 sin seguimiento (`git status --porcelain`)
- **Fuentes consultadas:** este repo, `Paulasaah/Anteproyecto` (local), engram local, engram del
  equipo (VPS), y el remoto `origin` vía `git ls-remote`. El release V9 de Nicolás se incorporó
  en la revisión del 2026-10-30 (ver §V9).

## Cómo leer este documento

Cada afirmación lleva uno de tres estados, sin excepción:

- **VERIFICADO** — se ejecutó o se leyó un artefacto del repo que lo demuestra. Se cita la ruta.
- **NO VERIFICADO** — no se ejecutó nada que lo demuestre en esta sesión. Se indica cómo validarlo.
- **NO EVALUADO** — no se revisó. Se declara el límite.

"Existe el artefacto" y "el resultado es correcto" son afirmaciones distintas. Este documento
audita lo primero y copia veredictos ya registrados; no recalcula cifras.

## Los cuatro objetivos específicos

Texto literal de `Anteproyecto/objetivos.tex` (repo `Paulasaah/Anteproyecto`, no este repo):

1. Caracterizar conjuntos de datos de imágenes dermatológicas representativos de lesiones
   cutáneas relevantes para el contexto clínico del departamento de Santander.
2. Diseñar un algoritmo de clasificación de lesiones cutáneas basado en aprendizaje
   autosupervisado utilizando imágenes dermatológicas.
3. Evaluar el desempeño del algoritmo propuesto mediante métricas reportadas en la literatura
   científica para tareas de análisis de imágenes médicas.
4. Implementar un prototipo de aplicación que integre el algoritmo desarrollado para el apoyo
   en el análisis de imágenes dermatológicas.

`docs/01-contexto-y-objetivos.md` está **desactualizado**: enumera tres objetivos y declara el
cómputo como "Kaggle T4 x2". Ambos puntos quedan superados por `objetivos.tex` y por la
operación en UNABIA. Ese documento no se usa como referencia de objetivos.

---

## Objetivo 1 — Caracterizar los conjuntos de datos

### Qué hay

| Elemento | Estado | Evidencia |
|---|---|---|
| Caracterización documentada de HAM10000, ISIC 2019/2020, PAD-UFES-20, BCN20000, CO2Wounds-V2 | VERIFICADO (existe) | `docs/02-datasets.md`; `Anteproyecto/capitulos/metodologia.tex` §conjuntos-datos |
| Notebooks de EDA versionados | VERIFICADO (existen) | `notebooks/proyecto_ssl_lesiones_v2.ipynb` … `v5.ipynb` |
| Particiones experimentales declaradas con tamaños | VERIFICADO (documentado) | Tabla `tab:particiones-experimentales` en `metodologia.tex` |
| Split disjunto por lesión con `assert` de cero solapamiento | VERIFICADO (regla documentada) | `docs/02-datasets.md`; reglas de higiene en `CLAUDE.md` §5 |
| Manifiesto congelado de evaluación ISIC 2018 (2.594 pares) con recibo y hashes | VERIFICADO (artefacto) | `resultados/preparacion_v8/isic2018_eval_2594_20261022/manifest.json`, SHA-256 `53009a68…fe0f179`; recibo `25f42c62…dc35a8` |
| Membresía congelada HAM10000 para scoring de segmentación (3.132 filas / 2.586 lesiones) | VERIFICADO (artefacto) | `plan-segmentacion/metodos-v8/launch-preparation/manifests/ham10000-segmentation-candidate-20261018-v4.envelope.json` |
| Alineación con el perfil epidemiológico de Santander | VERIFICADO (documentado) | `metodologia.tex` §perfil-epidemiologico-santander, Tabla `tab:patologias-santander` |

### Qué falta

- **Criterios de selección y exclusión, licencias y trazabilidad epidemiológica sin cerrar.**
  Lo declara el propio anteproyecto en `tab:trazabilidad-fases`, fila 1. No es un hallazgo nuevo
  de esta auditoría: es un pendiente ya reconocido.
- **Independencia por paciente no acreditada** en las cohortes de segmentación. El manifiesto
  ISIC 2018 acredita pares 1:1 y cero solapamiento por ID exacto, ID base y SHA-256 de imagen
  contra `selection`/`refit` del pool v2; eso es **preparación**, no independencia clínica.
  Los flags `patient_independence_accredited=false` y
  `historical_ssl_exposure_independence_verified=false` siguen en `false`.
  Evidencia: `odd/tasks/v8-cierre-bloqueos-lanzamiento.md`, sección del 2026-10-18.
- **HAM10000 no es elegible como test integrado de clasificación.** El cruce verificado muestra
  que los 3.132 IDs ya están repartidos entre train (1.772) y validación (1.360) de C1. Sirve
  solo para scoring retrospectivo de segmentación RPI.
- **Cifra invalidada que no se debe reutilizar:** los conteos de solapamiento ISIC 2018 calculados
  comparando solo el sufijo numérico de siete dígitos (entre ellos el histórico 1.917) producen
  falsos positivos. La cohorte vigente es el manifiesto de 2.594 pares.

### No evaluado

- Reproducción de los notebooks de EDA. No se ejecutó ninguno.
- Verificación independiente de los hashes del manifiesto ISIC 2018 (se leyeron los valores
  registrados, no se recalcularon). Se comprueba con `sha256sum` sobre los dos archivos.
- Estado de las licencias de cada dataset frente a su condición de uso declarada.

---

## Objetivo 2 — Diseñar el algoritmo de clasificación SSL

### Qué hay

Tres líneas de trabajo, con estados muy distintos. Conviene no leerlas como un solo bloque.

**Línea de clasificación — v7, cerrada.** VERIFICADO.
Evaluación única en validación (n = 2.004, probe `logreg`, `code_revision` `7fd6245c…`),
tres semillas (42/43/44), bootstrap pareado.

| Brazo | Macro-F1 [IC 95 %] | ECE [IC 95 %] |
|---|---|---|
| A* (SimSiam anclado, plain) | 0,6464 [0,5980; 0,6894] | 0,0135 [0,0138; 0,0336] |
| B* (A* + término de tono) | 0,6606 [0,6157; 0,7040] | 0,0325 [0,0253; 0,0497] |
| DINOv2 público | 0,6158 [0,5690; 0,6550] | 0,1307 [0,1159; 0,1476] |

Fuente: `resultados/clasificacion/v7/README.md`, generado desde
`resultados/clasificacion/v7/final_eval.json`.

**Línea de segmentación — dos generaciones conviven.** VERIFICADO.

| Variante | Dice ISIC 2018 [IC 95 %] | Dice HAM10000 [IC 95 %] | Fuente |
|---|---|---|---|
| `student_r3` (v7, destilación sobre pseudo-máscaras CCAM) | 0,7177 [0,7090; 0,7260] | 0,7847 [0,7812; 0,7883] | `resultados/segmentacion/TABLA_SEGMENTACION.md` |
| S4-UNet A2 (decoder DPT sobre DINOv2 congelado) | 0,7221 [0,7134; 0,7308] | 0,7949 [0,7918; 0,7984] | `resultados/TABLA_CIERRE.md` fila 13 |
| DermLIP-SLIC Run 008 (v8, 2.594 pares nativos) | 0,7904 [0,7824; 0,7983] | — | `odd/tasks/v8-cierre-bloqueos-lanzamiento.md`, 2026-10-23; `resultados/preparacion_v8/evaluation_report_run008.json` |

Run 008 queda **congelado como punto de referencia metodológico**, con 32 casos de Dice cero
(1,23 %) taxonomizados en `docs/58-analisis-casos-fallidos-y-rescate-literatura.md`.

**Línea de tono de piel — estimador ITA, no validado como fototipo.** VERIFICADO.
La estimación por imagen es el ITA con los umbrales fijos de Groh
(`src/dermssl/fitzpatrick.py`), no un FST validado. Rinde **por debajo de la línea base trivial**:
22,9 % frente a 51,3 % de la baseline de mayoría en PAD-UFES-20; 28,5–30,8 % frente a 32,9 %
en Fitzpatrick17k-C. Fuente: nota de regla en `resultados/TABLA_CIERRE.md` y fila 25.
Toda fila del repo que lo usa lleva la nota "estimación ITA no validada (no es FST)" por regla
automática, no por criterio manual. Eso es disciplina correcta y hay que conservarla.

**Código nuevo posterior a v7, commiteado pero sin resultado consolidado.** VERIFICADO (existe
el código; no se verificó resultado):
`0fa90e8` refinamiento SLIC TiSage, `b5d44d6` pipeline CIRCLe+DINOv2, `86b9bf4` Derm-TriplePrior,
`e8771aa` enmascarado de pseudo-etiquetas por banda de incertidumbre, `14164b6` ResNet34 U-Net
preentrenado + copy-paste con jitter de escala + calibración de prototipos PH2 + destilación KL
suave, `2daa4ed` flag de LR plano para ablación de un solo factor.

### Qué falta

- **"El plan SimSiam+DINOv2 sin diagnósticos no se lanzó" quedó obsoleto.** Lo declara el
  anteproyecto en `tab:trazabilidad-fases` fila 2 y en §Análisis de factibilidad, pero esa
  afirmación es del corte anterior. Desde el 2026-10-24 Nicolás corrió una campaña completa de
  unos veinte brazos SSL con evaluación por paciente — ver §Campaña v3. **El anteproyecto está
  desactualizado en este punto y hay que reescribirlo.**
- **Comparación de tres brazos A/B/C (sin segmentación / DermLIP-SLED nuevo / SLED v8) sin
  ejecutar.** Diseño fijado, configuración ejecutable no congelada. La receta `.005` es borrador.
- **Runner y config PAD-DA sin entregar.** Responsabilidad asignada a Nicolás. Es requisito para
  integrar y comparar A/B/C. Debe leer solo `target_adapt`, sin etiquetas y sin `target_test`.
  Fuente: `odd/tasks/v8-cierre-bloqueos-lanzamiento.md`, pendientes prioritarios 5.
- **Binding runner-loader-config al pool v2 sin cerrar** (pendiente 2) y sin prueba de que una
  corrida nueva no reutilice checkpoints, cachés ni pseudo-máscaras legacy (pendiente 3).
- **Gates de ejecución apagados.** `execution_enabled=false`, `launch_ready=false`,
  `training_started=false`. No se cambian sin evidencia real de recursos GPU, binding y
  autorización explícita (pendiente 4).
- **Auditoría modular v8 de cuatro métodos (SLED, USL-Net, RPI-Net, TriDermNet-HAGO) sin plan
  serial ejecutable.** `scripts/run_segmentation_controller_v8.py` es un motor genérico: ejecuta
  un plan JSON que ya exista con rutas reales; no lo genera ni lo infiere.
- **Atribución 32/158 no acreditada.** La frase histórica "reducidos de 158 en baseline SLED" no
  está respaldada: la referencia pareada de Run 008 es Run 007, no una baseline SLED. No
  interpretar 32/158 ni el 79,2 % como ablación controlada ni como mejora causal de SLIC.
  Fuente: addendum de trazabilidad del 2026-10-24.
- **Taxonomía de los 32 casos fallidos es post hoc y no disjunta.** `ISIC_0012179` aparece en dos
  categorías. Falta adjudicación multietiqueta o exclusiva.

### No evaluado

- Ningún entrenamiento ni inferencia se ejecutó en esta sesión.
- La suite `pytest` no se corrió. El repo declara 365 archivos de test; el comando real es
  `pytest -q` con `PYTHONPATH=src`. Pendiente de correr antes de afirmar salud del código.
- Estado real del clúster UNABIA (cuota, GPU libre, autorización). Las instantáneas GPU del
  repo son históricas, no preflight vigente.

---

## Objetivo 3 — Evaluar el desempeño

Este es el objetivo con más evidencia y, a la vez, el que arroja los resultados más incómodos.
No conviene suavizarlos: son el insumo honesto del capítulo de resultados.

### Qué hay

**Protocolo estadístico registrado antes del dato.** VERIFICADO.
`resultados/TABLA_CIERRE.md` documenta la regla de veredicto antes de la tabla: *cumplida* /
*no cumplida* solo cuando existe predicción registrada en `docs/26`, `docs/27`, `docs/28`,
`docs/30`, `docs/31`; *mejora* / *empeora* solo si el IC 95 % pareado excluye el cero.
Las cifras se generan desde JSON por `scripts/build_*`; un test falla si alguien edita un número
a mano (`tests/test_readme_tables_fresh.py`, `tests/test_resultados_readme_fresh.py`).
Esto es lo más sólido del proyecto y hay que defenderlo.

**Veredictos de las predicciones primarias de v7.** VERIFICADO (copiado de
`resultados/clasificacion/v7/README.md` y `resultados/TABLA_CIERRE.md`):

| Predicción | Criterio | Veredicto |
|---|---|---|
| P1 | macro-F1 A* − público ≥ −0,01 | cumplida (3/3 semillas) |
| P2 | ECE B* − A* < 0 con IC que excluye el cero, sin perder macro-F1 | **no cumplida** (3/3 semillas) |
| P3 (objetivo de tesis) | macro-F1 B* − público ≥ 0 y ECE B* − público < 0 | cumplida (3/3 semillas) |
| P1-NI | límite inferior IC macro-F1 A* − público ≥ −0,01 | **no cumplida** (no robusto a la semilla) |
| P3-IC | límite inferior IC macro-F1 B* − público > 0 | cumplida (3/3 semillas) |

**La máscara no ayuda al clasificador.** VERIFICADO. Filas 1–4 de `TABLA_CIERRE.md`:
con `student_r3`, 0 de 15 celdas mejoran contra la condición sin máscara, 4 empeoran; con A2,
0 de 15 mejoran, 2 empeoran. Resultado negativo, medido y pre-registrado.

**Validación externa en PAD-UFES-20: el SSL propio pierde contra DINOv2 público.** VERIFICADO.
`resultados/clasificacion/pad_ufes/README.md`, 1.141 imágenes / 720 pacientes, bootstrap pareado
agrupado por paciente:

| Contraste | Macro-F1 Δ | IC 95 % pareado |
|---|---:|---|
| B* s42 − público | −0,1070 | [−0,1350; −0,0789] |
| B* s43 − público | −0,1312 | [−0,1628; −0,0984] |
| A* s42 − público | −0,0859 | [−0,1112; −0,0572] |
| A* s43 − público | −0,1085 | [−0,1351; −0,0825] |

El ECE también va a favor del público (+0,1128 y +0,0978, IC excluye cero). Es decir: la ventaja
de v7 medida en HAM10000 **no transfiere** a fotografía clínica de celular.

**Y esto ya tiene continuación.** La campaña v3 de Nicolás atacó exactamente este problema con
otro protocolo (CV anidada por paciente sobre `target_adapt`, no zero-shot) y encontró que el
encoder es la palanca: SigLIP2 congelado llega a 0,6668 y con metadatos a 0,7027, contra 0,6063
del SimSiam adaptado. Ver §Campaña v3. **Los dos protocolos no son comparables entre sí**, pero
apuntan en la misma dirección: el SSL propio no es lo mejor disponible en fotografía clínica.

**Evaluación en ISIC 2019 / ISIC 2020.** VERIFICADO (existen tablas generadas):
`resultados/banco_completo/latex/TABLA_ISIC2019.tex` y `TABLA_ISIC2020.tex`, generadas por
`scripts/build_bench_tables.py`. En ISIC 2019 congelado, B71 original alcanza F1 0,6717
[0,6436; 0,6951] contra 0,6583 [0,6311; 0,6825] de DINOv2 público con el mismo protocolo.

**Sesgo por tono: brecha medida.** VERIFICADO. Dice en el tercil oscuro de luminancia es menor
que en el resto: A2 −0,0583 [−0,0652; −0,0515], `student_r3` −0,0616 [−0,0688; −0,0542] en
HAM10000 (filas 26–27). En ISIC 2018 el efecto es "distinguible pero no claro". Las mitigaciones
probadas (T6 umbral por tercil, T7 reentrenamiento con oscurecimiento y sobremuestreo) **no
cerraron la brecha**.

**Limitaciones de implementación documentadas.** VERIFICADO. L1–L3 en `TABLA_CIERRE.md`:
el probe lineal tiene sensibilidad de hasta 0,014 de macro-F1 según el número de hilos BLAS;
decisión registrada de fijar C1 en 4 hilos; con 1 hilo no reproduce bit a bit. El efecto SSL
medido (+0,091) es unas 6,5 veces esa sensibilidad. Registrar esto es buena práctica.

### Qué falta

- **Discrepancia a reconciliar: D1 cross-dominio.** `TABLA_CIERRE.md` fila 10 marca la predicción
  D1 (HAM10000 → ISIC 2019/2020, registrada en `docs/27` FASE 5) como *pendiente / sin datos en el
  repo*. Pero existen tablas generadas de ISIC 2019 y 2020 en `resultados/banco_completo/latex/`.
  **No se resuelve en esta auditoría.** Hay que determinar si el banco completo satisface la
  predicción D1 tal como fue registrada o si es un protocolo distinto. Mientras no se decida,
  D1 se reporta como pendiente.
- **Efecto de la transformación de tono sin medir.** Falta la medición antes/después sobre las
  mismas imágenes. Lo declara el anteproyecto en `tab:trazabilidad-fases` fila 3.
- **C3 (ablación Plain vs Proposed-oráculo) y S5 (destilación DINO → U-Net)**: pendientes sin
  datos en el repo (filas 24 y 16).
- **Cobertura de fototipos sin establecer.** PAD-UFES-20 no permite concluir sobre FST V–VI
  (62 casos IV, 10 V, 1 VI). Las distribuciones de ITA no determinan la composición clínica de
  HAM10000 ni de ISIC.
- **Trazabilidad del piloto de adaptación incompleta.** Los registros no permiten identificar la
  revisión de datos ni la partición usada (riesgo R6 del anteproyecto, declarado materializado
  parcialmente).
- **Evaluador pareado de Run 008 en revisión.** La corrección sigue abierta en T-002
  (`REVISION-1-FIDELIDAD.md`, `PROPUESTA-EVALUADOR-PAIRED-BOOTSTRAP-REV2.md`). Los valores p
  históricos no se trasladan a inferencia confirmatoria.

### No evaluado

- No se recalculó ninguna métrica ni intervalo de confianza. Todos los números de esta sección
  se copiaron de artefactos generados desde JSON.
- No se corrió `--check` de los generadores `build_*`, que es lo que probaría que las tablas no
  divergieron de su fuente. Se comprueba con el job `generated` de CI o corriendo cada
  `scripts/build_*.py --check`.

---

## Campaña v3 del clasificador — trabajo de Nicolás (2026-10-24 al 2026-10-29)

> **Corrección de una afirmación previa de este documento.** La primera versión decía que la
> observación más reciente del proyecto en el VPS era `#1984` del 2026-10-22 y que no había
> trabajo posterior. **Era falso.** Hay unas 270 observaciones posteriores, hasta `#2268` del
> 2026-10-29. No las encontré porque están ruteadas bajo el proyecto `claude-web` aunque
> pertenecen a `proyecto-grado-ssl-lesiones`, y mis búsquedas iban filtradas. La causa está
> registrada por Nicolás en `#2100`. El barrido correcto es por ID, no por filtro de proyecto.

Nicolás abrió una línea de trabajo completa sobre el clasificador que **no está en este repo**.
Vive en `/home/nicolas/Documentos/clasificador1-auditoria/clinical-reference-20261024` (local) y
`/home/jovyan/work/clf1_clinical_reference_20261024` (UNABIA). Todo lo de abajo es
**NO VERIFICADO** por mí: son afirmaciones registradas en el VPS, no artefactos que yo haya leído.
Las cito con su ID para que se puedan auditar.

### Protocolo y por qué no se compara con nuestras cifras

Objetivo: PAD-UFES-20 `target_adapt`, 1.134 imágenes tras el overlay de duplicados, lector lineal
balanceado, **CV anidada agrupada por paciente** (3 semillas CV x 5 outer x 3 inner).
Declarado explícitamente como **desarrollo/transductivo, no validación clínica**. Los tests viejos
(holdout de HAM, `target_test` de PAD) nunca se usaron (`#2224`).

Es un protocolo **distinto** del de nuestro `resultados/clasificacion/pad_ufes/` (lectura
zero-shot, n = 1.141, 3 clases). **No pongas las cifras lado a lado.**

### El resultado que cambia la tesis: el encoder es la palanca, y el SSL propio no es el mejor

De `#2224` y `#2225` (bootstrap por paciente, 500 remuestras, OOF fijo, sin corrección por
multiplicidad):

| Encoder / lector | Macro-F1 PAD | Δ contra `adapted-simsiam` [IC 95 %] |
|---|---:|---|
| kNN k20 (punto de partida) | 0,40–0,43 | — |
| `adapted-simsiam` + lector lineal balanceado | 0,6063 | referencia |
| + metadatos clínicos (sin diagnóstico ni biopsia ni IDs) | 0,6536 | +0,047 [+0,002; +0,092] |
| **SigLIP2-so400m congelado, solo imagen** | **0,6668** | **+0,0605 [+0,028; +0,092]** |
| **SigLIP2-so400m + metadatos** | **0,7027** | **+0,049 [+0,025; +0,074]** contra la fusión anterior |
| SigLIP2-base + metadatos | 0,7029 | +0,049 [+0,026; +0,073] — iguala al so400m y es más barato |
| DINOv2-large | 0,592 | — |
| Clásico (no profundo) | 0,31 | — |

**SigLIP2 es imagen-texto, no SSL.** Ahí está el conflicto: el mejor resultado disponible no
califica como entregable del objetivo general, que exige aprendizaje autosupervisado.

El diagnóstico de Nicolás: *el cuello de botella son las features y el tamaño de los datos, no la
capacidad del lector; las palancas de la receta SSL mueven milésimas, los metadatos y un encoder
más fuerte mueven ~+0,05* (`#2224`).

Techo honesto de literatura con split por paciente: **~0,72 macro-F1**. El 0,80 que circula **no
es reproducible** para 6 clases (`#2224`, `#2230`).

### Dos métodos SSL que sí superan el baseline, provisionalmente

De `#2266` (estado al cierre del 2026-10-27), criterio declarado: media de 3 semillas,
ΔPAD ≥ +0,02 y ΔHAM ≥ −0,01. Baseline SSL, media de 3 semillas: **PAD 0,597** (DE 0,010; la
semilla 42 con 0,606 era alta).

| Brazo | PAD | Δ | HAM | Veredicto |
|---|---:|---|---:|---|
| **VICReg** | 0,638 | +0,041 Welch [+0,019; +0,063] | +0,028 | **cumple, PROVISIONAL** |
| **Barlow Twins** | 0,633 | +0,035 [+0,008; +0,062] | +0,001 | **cumple, PROVISIONAL** |
| Barlow + VICReg (suma, s42) | 0,644 | — | — | no suma sobre VICReg solo |

**Provisional, y la palabra importa:** 3 semillas contra 3, y la prueba de permutación da
**p mínimo = 0,10**. No hay significancia por permutación. Nicolás decidió **no** ampliar a las
semillas 45/46.

**Brazos nulos o que empeoran** (`#2266`): DINO+iBOT, MULAN, BYOL, OOB, freeze4, 336, crop 0.4,
destilación (empeora), recetas v1. Y de los reportes individuales: `learning-rate-x4` +0,003,
`fewer-phone-repeats` −0,009, `constant-predictor-lr` −0,0057 (`#2248`, `#2250`), SkinDisNet
+0,0003 en PAD y −0,0141 en HAM (`#2240`), CLS distillation con fix de memoria −0,0323 en PAD y
−0,0520 en HAM (`#2262`).

Conclusión registrada en `#2250`: *todas las palancas de SimSiam probadas hasta ahí caen dentro de
±0,006 del baseline.*

### Caveats que no se pueden perder al redactar

1. **Fuga transductiva en los brazos con muestreo balanceado por clase.** Las etiquetas de origen
   de ISIC son **binarias** (`mal`/`ben`); PAD usa diagnóstico de 6 clases, que es **también la
   etiqueta de evaluación** (`#2268`). Cualquier brazo CB arrastra ese caveat.
2. **Selección de receta informada por etiquetas, autorizada explícitamente.** Nicolás autorizó
   usar los diagnósticos de HAM dev **solo para elegir la receta final**, sin usarlos en el
   entrenamiento SSL ni abrir los tests reservados (`#2160`). **Hay que declararlo en la tesis**:
   es selección informada por etiquetas, no un pipeline ciego.
3. **El protocolo es de desarrollo y transductivo.** No es validación clínica (`#2224`).
4. **Falta cohorte externa independiente.** DDI requiere que Nicolás acepte el RUA (`#2230`).
5. **Los duplicados cross-role existen y el filtro tiene sensibilidad limitada.** El overlay
   excluye 1.397 filas elegibles; el filtro por embeddings con cos ≥ 0,995 recuperó 0 de 7.945
   enlaces conocidos por ID crudo: **no detecta copias transformadas** (`#2224`).
6. **Los métodos son aproximaciones "estilo X", no reproducciones oficiales** (`#2267`).
7. **ViT-L es NO-GO**: el runtime del projector, el exportador (768 dims) y el runner están atados
   a ViT-B (`#2268`).

### Estado al 2026-10-29 y pedidos de Karen

`#2268` (2026-10-29 03:53, lo más reciente que existe en el VPS): runner desatendido en GPU0 y
GPU1 con colas selladas. GPU0: `dino-ibot-s43`, `s44`, `dino-ibot-cb-s42`, `moco-v3-s42`.
GPU1: `simsiam-cb-s42`, `masked-pixel-s42`, `rotation-s42`.

**Pedidos de Karen, registrados:** DINO+iBOT con 3 semillas; tabla de precisión por clase
(hecha, reproduce macro-F1 0,597 / 0,638 / 0,610); muestreo balanceado por clase (declarado como
carril guiado por etiquetas, **no** SSL puro).

También pendientes sin código al 2026-10-27: MAE de control, rotación/jigsaw, SimCLR/MoCo
(`#2266`). Y un bug que un test atrapó: el decoder de `masked-pixel` era un MLP por token, con
**gradiente cero al encoder** (`#2268`).

**NO EVALUADO:** el resultado de esa cola. No hay observaciones posteriores al 2026-10-29 03:53,
así que el estado de esos siete brazos es desconocido. Es lo primero que hay que preguntarle.

### Segmentación: el trabajo del 24 al 28 de octubre, recuperado

> **Corrección.** Este documento daba Run 008 (2026-10-23) como lo último de segmentación.
> **No lo era.** Hay cuatro días de trabajo posterior que no estaban registrados en el task file
> ni en el VPS, y que viven como archivos `scratch_*` sin seguimiento en la raíz del repo. Se
> recuperaron el 2026-10-30 buscando por fecha de modificación, no por git.

**1. Rescate de los 32 casos de Dice cero — ejecutado, con resultado.** VERIFICADO en
`scratch_rescue_32_results.json` y `scratch_rescue_32_run.log` (2026-10-24 07:56).

| | |
|---|---|
| Casos tratados | 32 (los de Dice cero de Run 008) |
| Dice medio antes | 0,0000 |
| **Dice medio después** | **0,2094** |
| **Casos recuperados** | **11 de 32** |
| Modos | `multispace+standard` (mayoría), `multispace+directional` (2 casos) |

Recuperaciones altas: `ISIC_0015936` 0,9442 · `ISIC_0000218` 0,8640 · `ISIC_0009252` 0,8515 ·
`ISIC_0014677` 0,7353. El **Mecanismo C** (espacio de color multiespacio) del plan de rescate
**funciona en un tercio de los casos fallidos**. Es un resultado real y reportable.

**2. IMA++ descargado y auditado como candidato a holdout independiente.** VERIFICADO: los
archivos están en disco.

*ISIC MultiAnnot++* es el mayor dataset público de segmentación multi-anotador:
**17.684 máscaras para 14.967 imágenes**, de las cuales **2.394 tienen de 2 a 5 anotaciones**
por **16 anotadores distintos** (`A00`–`A15`), con metadatos de herramienta (`T1`–`T3`) y nivel
de destreza (`S1`–`S2`), más máscaras de consenso **STAPLE** y **voto mayoritario**.
Zenodo DOI `10.5281/zenodo.14201692`, arXiv `2512.21472`.

Artefactos en el repo: `scratch_segs.zip` (154,8 MB, las máscaras),
`scratch_IMAplusplus_img_metadata.csv` (2,8 MB), `scratch_seg_metadata.csv` (3,0 MB),
`scratch_iaa_metrics_image.csv` (métricas de acuerdo inter-anotador).

Scripts de auditoría escritos el 2026-10-28: `scripts/check_imapp_cross_unabia.py`,
`scripts/audit_imapp_on_unabia.py`, `scripts/verify_imapp_user_stats.py`. Cruzan los 14.967 IDs
contra el pool histórico de 40.896 (con normalización de `_downsampled`), contra los 2.594 de
evaluación ISIC 2018 y contra el pool de selección de 34.330, y luego deduplican por `lesion_id`
y `patient_id`. Los comentarios del script trabajan sobre **12.235 candidatos** tras exclusión y
**11.807** tras deduplicación interna (428 capturas duplicadas retiradas).

**Por qué esto importa tanto:** IMA++ trae `patient_id` y `lesion_id`. Es exactamente lo que
faltaba para acreditar la independencia por paciente que sigue en `false`. Y la variabilidad
inter-anotador da una **cota superior humana** contra la cual comparar: ningún método puede
superar el acuerdo entre expertos.

**NO VERIFICADO:** que esos conteos de 12.235 y 11.807 se hayan ejecutado y confirmado; en el
script aparecen como comentarios y `print` de contraste contra cifras aportadas por la usuaria.
Se comprueba corriendo `scripts/audit_imapp_on_unabia.py` en UNABIA, donde están los inventarios.

**3. Evaluación `paired-raw-morph-20261027`.** El script de auditoría referencia
`/home/jovyan/work/dermlip_slic_v8/evaluations/paired-raw-morph-20261027/019/0ae35137b699d60f/cruda/cases.jsonl`.
Es una evaluación pareada del **2026-10-27**, posterior a Run 008, que compara crudo contra
morfología. **NO EVALUADO:** sus resultados. Están en el clúster, no en el repo.

**4. Instrumental de diagnóstico construido.** Scripts `scratch_*` para: origen de los ceros
(`trace_zero_origins`, `diagnose_008_zeros`, `investigate_zeros`, `compare_zeros`), vello
(`inspect_hair_rpi`, `pilot_rpi_hair`), viñeteado (`check_vignetting`), barrido de umbral
(`sweep_threshold`), refinamientos (`test_refinements`, `test_robust`, `test_slic_minimal`), y
extracción de método de los papers (`rpi-net.txt`, `triderm_pdf.txt`, `sec4_3_4_4.txt`).

Esto es material de las secciones 4.4 (ablación) y 4.7 (sensibilidad al umbral) del paper, según
[`60-guia-presentacion-cientifica.md`](60-guia-presentacion-cientifica.md). Hoy está en archivos
`scratch_*` sin seguimiento, con nombres que no sobreviven a una auditoría.

### Actividad del 2026-10-29 al 2026-10-30, verificada

Barrido por ID en el VPS y `git log` sobre refs remotas tras `fetch`, en los dos repos:

| Fuente | Resultado |
|---|---|
| VPS, proyecto de grado | Solo `#2267` (29-10 00:34) y `#2268` (29-10 03:53). **Nada después.** |
| VPS, hoy 30-10 | `#2270` y `#2271` (17:59 / 18:00) son de un microsite de teclados mecánicos en `noel/pagina_web`. **No son de la tesis.** |
| `proyecto-grado-ssl-lesiones`, todas las refs remotas | **Cero commits** desde el 2026-10-28. Último de Nicolás: `9ee675c`, 2026-10-21 21:34. |
| `skin-model-pwa`, todas las ramas | **Cero commits** desde el 2026-10-28. Último: `7a0980d`, 2026-10-25 20:49. |

Lectura honesta: **del 29 a hoy el avance en la tesis es el lanzamiento de la cola y nada más.**
El 29 de madrugada relanzó el contraste HAM+PAD, dejó siete brazos corriendo desatendidos y cerró
sesión. Desde entonces no hay commits ni observaciones del proyecto. Hoy trabajó en otra cosa.

Esto refuerza la pregunta abierta: la cola lleva más de un día corriendo sin que se haya
registrado su resultado.

### Un incidente de proceso que vale registrar

`INC-2026-10-27-01`: editar un archivo que un proceso vivo tenía importado invalidó la semilla 43
por guardia de integridad; ~1,3 GPU-h perdidas (`#2266`). Y en `#2267`: un intento no entrenó por
`FileExistsError` al reusar el directorio de salida de un smoke anterior — **no fue un fallo
predictivo del método**. Dos recordatorios de que un resultado nulo puede ser un problema de
plomería, no de ciencia.

---

## Rama `phone-web` — servicio de inferencia V9 + interfaz móvil instalable

Es el eslabón que conecta V9 con un prototipo usable, y **no estaba en este documento**.
**VERIFICADO** con `git diff --stat main...origin/phone-web`: 51 archivos, ~48.878 inserciones.

| Área | Archivos | Qué es |
|---|---|---|
| Interfaz móvil instalable | `demo/phone/index.html`, `manifest.webmanifest`, `sw.js`, `styles.css`, `icon.svg`, `icon-512.png` | PWA con service worker y manifiesto |
| Explicación del modelo | `demo/phone/model-explained.html` (190 líneas) | Pipeline explicado con evidencia de investigación |
| Despliegue | `demo/phone/vercel.json` | Configuración de Vercel |
| Servicio de inferencia | commit `7f6839c`, *feat(backend): version V9 inference service source* | Sirve A71 y B71 con Swagger, con guardas |
| Tests de paridad | `tests/test_phone_f4.py`, `test_phone_golden.py`, `test_phone_frontend.cjs`, `test_phone_segmented_service.py`, `tests/fixtures/v9_preprocess.py` | Paridad HTTP del teléfono contra **referencias float32 independientes** |
| Tests del demo y V9 | `test_demo_v9_service.py`, `test_demo_v9_manifest.py`, `test_demo_identity.py`, `test_demo_request_limits.py`, `test_demo_golden.py` | Contrato del servicio V9 |

Commits relevantes: `ef5c883` sirve A71/B71 con Swagger; `cdba2f8` verifica paridad HTTP contra
float32 independiente; `a7bae49` muestra top scores y **limitaciones explícitas de melanoma**;
`1e8f757` explica el pipeline con evidencia verificada; `bf3ce03` publica diagramas de
arquitectura congelados; `2df9804` corrige contraste de evidencia y matiza la etiqueta de tono de
B71. Fechas: 2026-10-16 y 2026-10-21.

Que haya tests de paridad contra referencias float32 independientes y que la interfaz declare
límites de melanoma explícitos es el nivel de rigor que corresponde en una herramienta de
tamizaje. Eso merece estar en el capítulo de despliegue.

### `phone-web` vs `skin-model-pwa`: dos interfaces móviles, dos repos

Punto que genera confusión y conviene fijar: **`phone-web` es una rama del repo compartido**
`nicolas2601/proyecto-grado-ssl-lesiones`, **no** del repo de la app. VERIFICADO con
`git ls-remote --heads origin` en ambos: `refs/heads/phone-web` existe solo en el compartido.

| | `phone-web` | `skin-model-pwa` |
|---|---|---|
| Ubicación | Rama del repo **compartido** de investigación | Repo **separado** |
| Stack | HTML/CSS/JS **sin framework** (`index.html`, `app.js`, `styles.css`) | **React 19 + Vite + TypeScript**, `radix-ui`, `motion`, `class-variance-authority` |
| Identidad | "DermSSL · Explorador móvil" — *"Una imagen. Siete hipótesis."* | `PRODUCT.md` con `impeccable:product-schema`, `DESIGN.md`, `openspec/` |
| Backend | Servicio de inferencia **V9** con Swagger, A71/B71, en el mismo repo | **NO EVALUADO** qué consume |
| Entrada | Imagen | **Cámara y galería**, flujo de usuario real |
| Extras | `architecture/{experiment-gates,phone-research,split-firewall}.html` | `scripts`: dev, build, typecheck, lint, test, test:coverage, check |
| Despliegue | `vercel.json` | `dist/` presente, procedencia no verificada |

### Resuelto: no hay duplicación de backend, y el prototipo está desplegado

La duda de si `phone-web` y `skin-model-pwa` eran dos pipelines de inferencia redundantes queda
cerrada. **No lo son: son backend y frontend del mismo sistema.** VERIFICADO:

1. `skin-model-pwa/src/features/analysis/prediction.ts:121` llama
   `POST /api/predict?model=B71&mode=base`.
2. `vite.config.ts` proxea `/api` a **`https://derm.nicolasmoreno.site`**.
3. Ese host responde con el servicio que versiona `phone-web` en `demo/backend/`
   (`app.py` expone `/api/status`, `/api/report`, `/api/features`, `/api/predict`, más
   `/docs`; con `v9_service.py`, `segmented_model_service.py`, `f4_service.py`,
   `request_limits.py`, `model_identity.py`).

**El servicio está vivo.** Verificado en vivo el 2026-10-30 con `curl`:

| Comprobación | Resultado |
|---|---|
| `GET /api/status` | **200**, 0,94 s |
| `release` | **v9.0.0** |
| `release_arm` | **B71** |
| `backbone_ready` / `classification_ready` / `reason` | `true` / `true` / `ready` |
| `identity.model_id` | `dermssl-v9-B71` |
| `identity.model_sha256` | `7fd581afc14a5c61da37f8d3873ce78c228276b0186a98e39096e9130285bbd6`, `sha256_verified: true` |
| `recipe` | `feature: cls4`, `tau: 0.0`, `tta: D4`, `aggregation: mean_probabilities`, `n_views: 8` |
| Runtime | `device: cpu`, `threads: 2` |
| `GET /docs` (Swagger) | **200** |
| `models` expuestos | **A71 y B71** |

El frontend es una PWA real: `vite-plugin-pwa` con workbox, `registerType: 'prompt'`,
`navigateFallbackDenylist: /^\/api/` (la navegación a la API nunca devuelve el shell), display
`standalone`, `lang: es`. **La app tiene nombre: `Perma`**, descrita como *"Prototipo experimental
de investigación de imágenes dermatológicas"*. Eso actualiza `#2010`, que declaraba la marca
abierta.

Conclusión para el objetivo 4: **el prototipo no está "pendiente". Está desplegado, sirviendo
V9 y respondiendo.** Esa es la corrección más grande de toda esta auditoría.

### La app está publicada en `https://skin.tikno.pro`

**VERIFICADO el 2026-10-30.** Este es el despliegue público de la app, y faltaba en este
documento.

| Comprobación | Resultado |
|---|---|
| `GET /` | **200**, HTML con bundle de Vite (`/assets/index-DA-l1DrL.js`) |
| `last-modified` | **2026-10-26 01:49** |
| `manifest.webmanifest` | `name: "Image study — research prototype"`, `short_name: "Image study"`, `lang: **en**`, `display: standalone`, `theme_color: #f8f7fb` |
| Descripción | *"Experimental image classification, not a diagnostic service."* |
| `permissions-policy` | `camera=(self), microphone=(), geolocation=()` — **cámara habilitada, micrófono y geolocalización bloqueados** |
| Proxy | Cloudflare; `referrer-policy: no-referrer`, `x-content-type-options: nosniff` |
| **`GET /api/status`** | **200**, y devuelve **exactamente el mismo payload** que `derm.nicolasmoreno.site`: `release v9.0.0`, `arm B71`, `model_sha256: 7fd581af…`, `sha256_verified: true` |

Es decir: **`skin.tikno.pro` sirve el frontend y expone el mismo backend V9**. El objetivo 4 no
solo está implementado: está **publicado en una URL propia, con cámara habilitada y permisos
mínimos bien declarados**.

Nota de dominio: `tikno.pro` es el mismo dominio del engram del equipo
(`engram-mcp.tikno.pro`). O sea, la app no cuelga del dominio personal de Nicolás sino de la
infraestructura compartida — eso mitiga en parte el riesgo institucional señalado más abajo,
aunque sigue siendo infraestructura privada del equipo, no de la UNAB.

**Discrepancia de marca, VERIFICADA.** Producción y las tres ramas del repo
(`main`, `develop`, `feature/camera-ml-motion`) dicen `Image study`, `lang: en`. Pero el
`vite.config.ts` **del disco de esta máquina** dice `Perma`, `lang: es`,
`theme_color: #F5F7FD`. No es una versión vieja: es trabajo local sin commitear (ver abajo).
La app en producción **no se llama Perma**.

### Trabajo local sin commitear en `skin-model-pwa`

**VERIFICADO** con `git status` en `C:/Users/mpaul/Documents/skin-model-pwa`: **15 modificados,
1 borrado, 8 sin seguimiento**; `git diff --stat` reporta 16 archivos con 86 inserciones y
**426 supresiones**.

| Sin seguimiento | Qué sugiere |
|---|---|
| `src/features/perma/` | Módulo completo nuevo |
| `src/styles/perma.css`, `perma-reference.css` | Rediseño visual con hoja de referencia |
| `tests/perma-app.test.tsx` | Suite de la nueva app |
| `src/i18n/portuguese.ts` | **Tercer idioma**: portugués |
| `public/fonts/`, `favicon.ico`, `favicon.svg` | Tipografía e identidad nuevas |

Modificados: `index.html`, `vite.config.ts`, `App.tsx`, `useAnalysis.ts`, `LanguageSwitch.tsx`,
`LocaleProvider.tsx`, `main.tsx`, `app.css`, los cinco iconos y `brand-mark.svg`, más dos tests.

Es un **rediseño de marca e internacionalización completo** que existe solo en este disco. Mismo
patrón de riesgo que los 8 commits sin pushear del repo de investigación, y aquí ni siquiera hay
commits: son cambios en el árbol de trabajo. **Es lo más frágil de todo el proyecto ahora mismo.**

### Qué es `derm.nicolasmoreno.site` y cómo está expuesto

Es **la API de inferencia**, no un sitio web. La raíz devuelve
`{"service":"DermSSL research API","docs":"/docs"}`; la interfaz humana es Swagger en `/docs`, y
la interfaz de usuario es la app `Perma`, que consume esta API.

Configuración de seguridad, **VERIFICADA** con headers y lectura de `demo/backend/app.py`:

| Control | Estado |
|---|---|
| Proxy | **Cloudflare** (`Server: cloudflare`, `CF-RAY`) |
| HSTS | `max-age=31536000` |
| CSP | `default-src 'self'; base-uri 'none'; frame-ancestors 'none'; connect-src 'self'` |
| Cabeceras | `x-content-type-options: nosniff`, `x-frame-options: DENY`, `referrer-policy: no-referrer` |
| CORS | Un **único** origen vía `DEMO_FRONTEND_ORIGIN`; solo `GET`/`POST`; `allow_credentials=False` |
| Rate limiting | Middleware `InferenceLimits`, responde `429 RATE_LIMITED` |

Para un prototipo de investigación expuesto a internet, eso está bien resuelto: no hay CORS
abierto, hay límites de petición y las cabeceras de seguridad están puestas a mano en un
middleware propio, no heredadas por defecto.

**El riesgo no es técnico, es institucional:** `nicolasmoreno.site` es un **dominio personal**.
El prototipo que sustenta el objetivo 4 depende hoy de infraestructura privada de un integrante
del equipo. Si ese dominio caduca o el servicio se apaga, el entregable desaparece. Para una
tesis conviene, como mínimo, dejar documentado cómo se levanta el servicio desde cero —y el
bundle Docker de V9 sirve exactamente para eso— y no depender de que la URL siga viva el día de
la sustentación.

### Tres cosas que el despliegue deja abiertas

1. **Está desplegado B71 por defecto, y B71 es el brazo que pierde en fotografía de celular.**
   En PAD-UFES-20, B\* contra DINOv2 público da −0,1070 y −0,1312 de macro-F1 con IC que excluye
   el cero (ver §Objetivo 3), y la campaña v3 midió SigLIP2 congelado por encima del SimSiam
   adaptado. El servicio expone también A71, así que la elección de B71 como predeterminado es
   una decisión que hay que justificar o revisar. No es un detalle de configuración: es el modelo
   que ve un usuario real.
2. **El `model_sha256` servido no es trazable desde el repo.** VERIFICADO: `7fd581af…` no aparece
   en ningún JSON, MD o TXT de la rama actual ni en los manifiestos de `phone-web`. Puede estar
   registrado en el asset `V9_MANIFEST.json` del release, al que no tenemos acceso — así que
   **"no está en el repo" no significa "no está registrado"**. Pero hoy no se puede auditar desde
   acá qué modelo exacto está en producción.
3. **El `/api/report` desplegado cambió de alcance.** Devuelve `source: {"release": "v9.0.0",
   "scope": "current_inference_runtime"}`: informa el runtime, **no** el reporte científico
   sellado de `resultados/` que servía el `demo/` del repo. Si el prototipo debe mostrar los
   resultados de la tesis, eso es una funcionalidad que se perdió en la transición.

**NO EVALUADO:** si los tests de `phone-web` pasan; si `Perma` está desplegada en una URL pública
propia o solo corre en local contra el backend remoto; el estado de las otras seis ramas de
feature del PWA.

---

## Prototipo PWA — `nicolas2601/skin-model-pwa` (trabajo de Nicolás)

Esto es, hoy, **el objetivo 4 real**, y la primera versión de este documento lo ignoraba por
completo.

### Verificado con git, no solo con la memoria

| Afirmación | Estado | Cómo se comprobó |
|---|---|---|
| Existe el repo `nicolas2601/skin-model-pwa` y está clonado en esta máquina | **VERIFICADO** | `C:/Users/mpaul/Documents/skin-model-pwa`, `git remote -v` |
| Tu clon de `main` está al día con el remoto | **VERIFICADO** | local y `refs/heads/main` = `45abc2a` |
| Punta de `main`: `45abc2a` (2026-10-25), merge del PR #4 desde `develop` | **VERIFICADO** | `git log -1` |
| Siete ramas de feature en el remoto | **VERIFICADO** | `git ls-remote --heads origin`: `bootstrap`, `product-brief`, `pwa-core`, `visual-directions`, `native-camera-assist`, `camera-ml-motion`, `develop` |
| Stack y artefactos: `PRODUCT.md`, `DESIGN.md`, `openspec/`, `src/`, `tests/`, `config/`, `design/`, `dist/` | **VERIFICADO** | listado del directorio |

### Qué registra el VPS sobre él

- `#2010` (2026-10-22): repo creado en worktree aislado, con `PRODUCT.md` de 133 líneas. Flujo de
  usuario real con **cámara/galería**, clasificación declarada **experimental**, revisión por
  especialista **fuera de la app**, límites científicos explícitos. Decisiones de tecnología,
  marca y privacidad declaradas abiertas. "`skin-model-pwa` es etiqueta técnica de trabajo, no
  marca elegida."
- `#2040` (2026-10-23): bugfix de detección offline verificado con Playwright. `navigator.onLine`
  devolvía `true` tras una recarga offline cacheada mientras un HEAD de red fallaba; se agregó un
  HEAD acotado sin credenciales. Verificado además: `/api` no devuelve el shell, inventario de
  caché estático sin cambios, `localStorage`/`IndexedDB` vacíos, y **ningún POST de inferencia**.
- `d89eee8` (2026-10-25): captura cuadrada y guía local de asistencia de cámara.

Que hayan verificado que `localStorage`/`IndexedDB` queden vacíos y que no se dispare inferencia
sin consentimiento, en una app que toca fotos de lesiones de personas reales, es exactamente el
cuidado que corresponde. Conviene que eso llegue al capítulo, no solo al commit.

### Qué falta

- **NO EVALUADO:** si el PWA consume los bundles de V9, el `demo/` de este repo, o nada todavía.
  Es la pregunta que decide si los tres caminos convergen.
- **NO EVALUADO:** estado de `feature/camera-ml-motion` y de las otras seis ramas; si sus tests
  pasan; si hay build desplegado. `dist/` existe en el clon pero no verifiqué su procedencia.
- Decisiones de marca, privacidad y tecnología declaradas abiertas en `#2010`: **NO EVALUADO** si
  se cerraron desde entonces.
- El routing público seguía requiriendo aprobación al 2026-10-23 (`#2040`).

---

## V9 — Bundles de inferencia limpios (trabajo de Nicolás)

Entra por engram del equipo (VPS), observaciones `#1953`, `#1954`, `#1955`. **V9 no es una nueva
generación de modelo**: es el empaquetado de inferencia completa de los clasificadores A71/B71 en
tres formatos. Es el aporte más directo al objetivo 4 que existe hoy y no estaba reflejado aquí.

### Verificado contra el remoto, no solo contra la memoria

| Afirmación | Estado | Cómo se comprobó |
|---|---|---|
| Existe el tag `v9.0.0` en `origin` | **VERIFICADO** | `git ls-remote --tags origin` |
| El tag apunta a `8317d23fd327b614d9f62ba9a4a5ae65c8b99698` | **VERIFICADO** | Coincide exactamente con lo registrado en `#1954` |
| Ese commit es `8317d23` (2026-10-16, *docs: Address second readability audit*) y es la punta de `origin/main` | **VERIFICADO** | `git log -1 origin/main` |
| El commit del tag **no** está en el historial de `experiment/modular-audit-v8` | **VERIFICADO** | `git merge-base --is-ancestor 8317d23f HEAD` → falso |
| Release publicado, 11 assets, ~2,22 GB, SHA-256 comprobados antes y después de publicar | **NO VERIFICADO** | Afirmación de `#1954`. La API de GitHub devuelve 404 para este repo con la cuenta `Paulasaah`: es privado y esta cuenta no tiene acceso de lectura a sus releases. **REQUIERE INTERVENCIÓN** |
| Métricas registradas: A71 F1 0,7025, B71 F1 0,7104 (AMP canónico, delta máximo 0 contra la referencia de 2.004) | **NO VERIFICADO** | Afirmación de `#1954`. Protocolo A71/B71 (banco v7.1), distinto del protocolo A\*/B\* de v7: **no son comparables directamente** con los 0,6464 / 0,6606 de la tabla del objetivo 2 |
| 67 tests verdes, 1 omitida, `ruff` verde | **NO VERIFICADO** | Afirmación de `#1954`, y medida sobre `/home/nicolas/Documentos/proyecto-grado-consolidado`, **no sobre este repo**. En este repo `ruff` está en rojo (ver Hallazgos transversales) |

### El problema de trazabilidad, verificado

El tooling que generó V9 **no está en el repositorio**. VERIFICADO:

- `scripts/release_v9/`, `scripts/build_clean_bundle.py` y `src/dermssl/release/clean_bundle.py`
  no existen ni en la rama de trabajo ni en `origin/main` (`git ls-tree -r origin/main`).
- `docs/23-modelo-descargable-y-uso.md` llega a la §5 en ambas ramas. La §6 que según `#1953`
  documenta la convención de los tres formatos no existe en el repo.
- `src/dermssl/release/` sí existe, con `fuse.py`, `onnx_tools.py`, `package.py`,
  `torch_convert.py`, `torch_export.py`, `torch_forms.py`, `preprocess.py`, `plain_dinov2.py`
  — pero no el módulo de bundle limpio.

Lo registra la propia memoria: no hubo commit ni push, y el overlay de código viaja dentro del
asset `V9_SOURCE.zip` (647 archivos). Consecuencia: **V9 se publicó desde un tag base que no
contiene el código que la produjo.** Reproducirla exige descargar el ZIP del release.
Esto es el riesgo R6 del anteproyecto —pérdida de trazabilidad— materializado en el artefacto
más importante del objetivo 4.

### El motivo de V9 es un defecto del demo que esta auditoría no detectó

`#1953` registra la causa: *"un backbone.pt no incluía cabeza/receta y la app usaba CLS768
simplificado en lugar del clasificador registrado; cambiar extensión no arregla esa divergencia."*

Eso describe una divergencia entre lo que la app servía y el clasificador registrado.
**NO VERIFICADO:** si `demo/` en esta rama todavía tiene esa divergencia. Se comprueba leyendo
`demo/backend/model_service.py` contra `demo/models/ssl_proposed_vitb14_classifier_manifest.json`.
Hasta que eso se revise, la fila "Servicio de modelo — versionado" de la tabla del objetivo 4
acredita que el archivo existe, no que sirva el clasificador correcto.

### Convención establecida por Nicolás

`#1953` la fija como patrón, no como hecho puntual: todo release futuro del clasificador entrega
`.pt` completo (backbone + cabeza fusionada + config + clases + prior + TTA, loader
`weights_only`, sin `pickleModule`), ONNX (con preprocesamiento y runner D4 si el grafo es de una
vista) y Docker (modelo ONNX, misma receta, licencias, versiones fijadas, sin descargas al
arrancar), con verificación de la cadena referencia → PT → ONNX → Docker y de los SHA del
servidor antes de publicar. Advertencia explícita registrada: **no promover los exploratorios
`balanced`/CORAL como mejoras confirmadas.**

### Alcance de la revisión de engram del equipo

La observación más reciente del proyecto en el VPS es `#1984`, del **2026-10-22**. No se
encontraron memorias de Nicolás posteriores a esa fecha. Búsquedas de `DermLIP`, `SLIC`,
`Run008`, `TriplePrior`, `CIRCLe`, `PAD-DA` y `runner` en el VPS no devolvieron resultados: el
trabajo v8 de los últimos días vive en este repo y en engram local, no en el del equipo.
**NO EVALUADO:** si Nicolás tiene trabajo posterior al 2026-10-22 sin registrar en engram.
Eso se resuelve preguntándole.

---

## Objetivo 4 — Implementar el prototipo

> **Leer junto con §Prototipo PWA y §V9.** El PWA `skin-model-pwa` es el objetivo 4 real en
> curso; V9 es el empaquetado de inferencia. El `demo/` de este repo es el tercero de tres
> caminos paralelos. Esta sección audita solo el tercero.

### Qué hay

VERIFICADO (existe el código): hay un prototipo funcional versionado en `demo/`, no un plan.

| Componente | Ruta | Estado |
|---|---|---|
| API FastAPI | `demo/backend/app.py` | Versionada |
| Servicio de modelo (carga perezosa de pesos) | `demo/backend/model_service.py` | Versionado |
| Reporte desde artefactos sellados | `demo/backend/report.py` | Versionado |
| Frontend estático | `demo/frontend/` (`index.html`, `app.js`, `styles.css`) | Versionado |
| Probe y manifiestos | `demo/models/ssl_proposed_vitb14_{probe.npz,classifier_manifest.json,run.json}` | Versionados |
| Imágenes golden | `demo/models/golden_images/` | Versionadas |
| Backbone `.pt` (331 MB) | Externo, vía `MODEL_WEIGHTS_PATH` | **No versionado** |

Endpoints declarados: `GET /api/status`, `GET /api/report`, `POST /api/features`,
`POST /api/predict` (modos `base` y `tta`). Límite de 10 MB por carga, JPEG/PNG/WebP, headers de
seguridad. El `README` declara 13 casos de prueba en
`tests/test_demo_api.py`, `tests/test_demo_model_service.py`, `tests/test_demo_report.py`.
Límite científico declarado en el propio demo: distingue predicción de clase real por SHA-256 de
la imagen golden, no infiere la etiqueta del nombre de archivo.

### Qué falta

- **Discrepancia con el anteproyecto, a resolver con criterio.** `tab:trazabilidad-fases` fila 4
  y §Análisis de factibilidad declaran "Prototipo aún no implementado. Estado: pendiente".
  El repo tiene un backend, un frontend y un contrato de modelo versionados. Las dos
  afirmaciones son compatibles solo si "implementado" se define como *prototipo con el modelo
  final seleccionado*, que es lo que falta. Hay que **fijar esa definición por escrito** antes de
  redactar el capítulo, o el tribunal va a leer una contradicción.
- **El modelo final no está seleccionado.** Es la razón real por la que el objetivo 4 no cierra,
  y se agrava con el resultado de PAD-UFES-20: si el SSL propio pierde contra DINOv2 público en
  fotografía de celular, servir el backbone propio en un prototipo orientado a tamizaje con
  teléfono necesita justificación explícita, no inercia.
- **Preprocesamiento sin fijar** y **criterio de evaluación de uso previsto sin definir con
  especialistas**. Ambos declarados en §Análisis de factibilidad.
- **Riesgo R7 abierto** (discrepancia laboratorio vs. prototipo): V9 existe precisamente porque
  esa discrepancia se detectó (app sirviendo CLS768 simplificado en lugar del clasificador
  registrado). La mitigación está empaquetada en el release; **no está integrada en `demo/`**
  ni en el repo. El riesgo sigue abierto en esta rama.
- **`demo/` ya fue eliminado en `main`** (`d03b17b`, PR #7). Esta sección audita un artefacto que
  sobrevive solo en nuestra rama y que upstream dio de baja. No tiene sentido invertir en él sin
  decidir antes el punto siguiente.
- **Hay cuatro caminos paralelos sin converger**, no dos ni tres:

  | # | Camino | Estado |
  |---|---|---|
  | a | `demo/` en este repo, backbone externo de 331 MB vía `MODEL_WEIGHTS_PATH` | **Eliminado en `main`**; vive solo en `experiment/modular-audit-v8` |
  | b | Bundles V9: `.pt` completo, ONNX, Docker | Publicados; tooling fuera del repo |
  | c | Rama `phone-web`: servicio de inferencia V9 con Swagger + `demo/phone/` instalable con `vercel.json` | En el repo compartido, sin mergear a `main` |
  | d | `nicolas2601/skin-model-pwa`: React/Vite, cámara/galería, siete ramas de feature | Repo aparte, `main` del 2026-10-25 |

  **Actualización verificada:** (c) y (d) **no compiten, se complementan**. `skin-model-pwa`
  (la app `Perma`) es el frontend y consume el servicio de (c) en
  `https://derm.nicolasmoreno.site`, que respondió 200 sirviendo V9 B71 el 2026-10-30 — ver
  §Rama `phone-web`. Los que quedan sin rol son (a), ya eliminado en `main`, y la interfaz
  `demo/phone/` de (c), que se solapa con `Perma`.

  Es decir: **el objetivo 4 está desplegado y funcionando**, con dos pendientes reales — justificar
  o cambiar B71 como modelo predeterminado, y decidir si `demo/phone/` o `Perma` es la interfaz
  de la tesis.

### No evaluado

- **El demo no se ejecutó.** No se verificó que arranque, que responda los endpoints ni que la
  inferencia real funcione. Requiere el checkpoint externo de 331 MB, cuya presencia en esta
  máquina no se comprobó. Se valida con:
  `uvicorn demo.backend.app:app --host 127.0.0.1 --port 8765` y
  `pytest -q tests/test_demo_api.py tests/test_demo_model_service.py tests/test_demo_report.py`.
- Los 13 casos de prueba del demo no se corrieron: el conteo es lo que declara el `README`,
  no una medición de esta sesión.

---

## Hallazgos transversales

### 1. Higiene de código: `ruff` falla

**VERIFICADO.** `python -m ruff check .` devuelve **59 errores**: 58 `F401` (import sin usar) y
1 `F821` (nombre no definido).

El `F821` es un defecto real, no estilo: `scripts/figures/fig3_masks.py:223` usa
`argparse.ArgumentParser` sin importar `argparse`. Ese script **falla al ejecutarse**.
No evaluado: si ese script respalda alguna figura publicada.

La configuración de `ruff.toml` es deliberadamente angosta (`E9, F63, F7, F82, F401`) para que el
lint no sea ruido. Si está en rojo, el mecanismo dejó de cumplir su propósito.

### 2. Volumen de trabajo sin commitear

**VERIFICADO.** 67 modificados, 1 borrado, 379 sin seguimiento. Concentración de los sin
seguimiento: 63 en `plan-segmentacion/metodos-v8`, 17 en `tests/segmentation_v8`, 14 en
`resultados/preparacion_v8`, 7 en `src/dermssl`. El resto son ZIP y TAR de transferencia en la
raíz (`v8-*.zip`, `v8-validation-f1dc0d2.tar`) y `watch_and_report.sh`.

Hay código de producto y tests fuera de git. Para un proyecto cuyo riesgo R6 declarado es
"pérdida de trazabilidad de experimentos", esto es exactamente el riesgo materializándose.
La restricción del repo aplica: no se descarta nada que respalde un número reportado; ante duda,
"no evaluado", nunca "descartar". El veredicto correcto es triaje, no limpieza.

El mismo patrón aparece del lado de Nicolás, y ahí es peor: el tooling de V9 está fuera del repo
por completo, solo dentro de un asset de release (ver §V9). No es un descuido de una persona;
es un problema del flujo de trabajo del equipo.

### 2.b Estado real de las ramas remotas

> **Corrección metodológica.** La versión anterior de esta sección afirmaba que `origin/main`
> estaba detenido en `8317d23` (2026-10-16), leyendo `git log -1 origin/main`. Eso lee la
> **referencia local cacheada**, no el remoto: sin `git fetch` no prueba nada sobre el estado
> upstream. `git ls-remote` es la fuente correcta. Tras hacer `fetch`, el cuadro es otro.

**VERIFICADO** tras `git fetch origin`: `origin/main` está en `9ee675c` (2026-10-21 21:34),
no en `8317d23`. Trae tres commits que no estaban en nuestra vista:

| Commit | Fecha | Qué hace |
|---|---|---|
| `d03b17b` | 2026-10-21 | `chore(demo): remove the local demo app and its tests` |
| `951ccdc` | 2026-10-21 | `docs: point demo links to the standalone app and the phone-web service` |
| `9ee675c` | 2026-10-21 | Merge del PR #7 desde `chore/remove-demo-frontend` |

**El `demo/` que audita este documento ya fue eliminado en `main`.** Sobrevive únicamente en
`experiment/modular-audit-v8`. Los enlaces de documentación ahora apuntan a la app standalone y
al servicio `phone-web` (ver §Objetivo 4).

También aparecieron dos ramas remotas que no teníamos: `chore/remove-demo-frontend` y
**`phone-web`**.

Aun así, el trabajo v8 de segmentación sigue solo en nuestra rama.

Y esa rama tampoco está al día en el remoto: su punta remota es `f717420` (2026-10-21,
*docs(datasets): scope pool exclusions to segmentation*), mientras el HEAD local es `2daa4ed`.
**VERIFICADO:** hay **8 commits locales sin pushear** (`git rev-list --count f717420..HEAD`).

Sumando: 8 commits sin pushear, 379 archivos sin seguimiento y 67 modificados. Todo el trabajo de
segmentación de la última semana —SLIC/TiSage, CIRCLe, Derm-TriplePrior, ResNet34 U-Net,
destilación KL— existe únicamente en esta máquina. Es el punto más urgente del documento:
no es deuda técnica, es exposición a pérdida total.

### 3. CI con disparo solo manual

**VERIFICADO (documentado).** `.github/workflows/ci.yml` corre únicamente por
`workflow_dispatch` desde 2026-09-29 (billing de Actions). El job `generated` —el que impide que
las tablas divergan de su JSON— por lo tanto **no corre automáticamente**. La garantía existe en
el código; la ejecución depende de que alguien la dispare.

### 4. El canal entre sesiones estaba vacío

**VERIFICADO.** `docs/ESTADO.md` no existía antes de este documento y `docs/estado/` está vacío,
aunque `SESIONES.md` los define como canal primario entre sesiones. Este archivo abre el canal;
los contratos por sesión trabajadora siguen pendientes de crear.

### 5. `docs/README.md` no indexa los documentos recientes

**VERIFICADO.** El índice llega hasta el doc 50. Los docs 51–59 —los más recientes, incluidos
`56-propuesta-semisupervisada-dermlip-slic.md`, `58-analisis-casos-fallidos-y-rescate-literatura.md`
y `59-propuesta-arquitectonica-derm-tripleprior.md`— no están listados. Hay que consultarlos por
nombre de archivo.

---

## Resumen por objetivo

| Obj. | Estado honesto | Bloqueo principal |
|---|---|---|
| 1 | Evidencia sustancial; pendientes declarados de licencias, trazabilidad e independencia por paciente | Independencia clínica no acreditada en las cohortes de segmentación |
| 2 | Mucho más avanzado de lo que decía el anteproyecto: v7 cerrada, segmentación con tres generaciones medidas, campaña v3 con ~20 brazos SSL evaluados, tono por debajo de la línea base trivial | VICReg y Barlow Twins ganan solo **provisionalmente** (permutación p = 0,10); el mejor resultado absoluto no es SSL |
| 3 | El objetivo con más evidencia y con los resultados más adversos | Dos protocolos de PAD no comparables; D1 por reconciliar; falta cohorte externa independiente (DDI sin RUA) |
| 4 | **Desplegado y funcionando** (verificado en vivo): la app `Perma` consume el servicio V9 en `derm.nicolasmoreno.site`, que responde 200 sirviendo B71 | Sirve **B71**, el brazo que pierde en fotografía de celular; el SHA del modelo en producción no es trazable desde el repo |

El estado real es **mucho más avanzado** que lo que declara el anteproyecto: hay una campaña de
clasificación completa, y el prototipo del objetivo 4 **está desplegado y respondiendo en
producción**, no pendiente. La frase del anteproyecto *"Prototipo aún no implementado"* es
directamente falsa al 2026-10-30, y está verificada como falsa con una llamada HTTP.

La tensión central del proyecto, ya medida por dos caminos independientes: **el mejor clasificador
disponible no es autosupervisado.** SigLIP2 congelado más metadatos llega a 0,7027 contra 0,6063
del SimSiam adaptado, con IC que excluye el cero. Y el objetivo general exige SSL. Eso no se
resuelve con más corridas: se resuelve decidiendo qué se entrega y cómo se enmarca.

---

## Las dos vías de trabajo

El trabajo del proyecto corre por dos carriles distintos, que **no se mezclan** y tienen reglas
distintas. Confundirlos es lo que produjo documentación sin hilo y pruebas sin registrar.

### Vía A — Documentación

Produce el documento evaluable: el paper y la tesis en
`C:/Users/mpaul/Documents/Anteproyecto`.

- Estándar fijado en [`60-guia-presentacion-cientifica.md`](60-guia-presentacion-cientifica.md).
- **Congelada por decisión del 2026-10-30** mientras siguen las pruebas. Los diagramas no se
  tocan.
- Solo entra material **cerrado**: cifra generada desde JSON, con protocolo registrado antes del
  dato y veredicto escrito. Nada provisional.
- Regla del repo que sigue vigente: las tablas se generan con `build_*`, nunca a mano.

### Vía B — Investigación

Produce decisiones y configuraciones nuevas. No produce documento final.

Tres frentes abiertos:

| Frente | Qué investiga | Dónde se registra |
|---|---|---|
| **App** | Producto, experiencia, contrato técnico con el servicio | `docs/estado/app-ux-producto.md`, `docs/estado/app-tecnica-contrato.md` |
| **Configuraciones de modelo** | Nuevas recetas según resultados y análisis: brazos SSL, encoder, pipeline segmentación→clasificación | Engram del equipo + `odd/tasks/` |
| **Protocolo de evaluación** | Evaluador pareado (T-002), baselines comparables, ablaciones | `plan-segmentacion/metodos-v8/` |

- Acá **sí** entra lo provisional, siempre que se marque como tal. VICReg con permutación
  p = 0,10 es resultado de vía B, no de vía A.
- Regla de registro: **toda prueba o implementación se guarda en el engram del equipo**, por
  decisión del 2026-10-30. Es el canal donde Nicolás y Pau se ven el trabajo.

El flujo es siempre B → A, nunca al revés. Un resultado pasa a documentación cuando tiene
protocolo registrado, veredicto y evidencia reproducible.

---

## Decisiones abiertas para la sesión maestra

Ordenadas por lo que desbloquean, no por esfuerzo:

1. **Salvar el trabajo local, en los dos repos.** Antes que cualquier decisión científica.
   - `ProyectoDeGrado`: 8 commits sin pushear, 379 sin seguimiento, 67 modificados.
   - `skin-model-pwa`: **15 modificados, 1 borrado, 8 sin seguimiento, sin un solo commit** — el
     rediseño `Perma` completo, con portugués, fuentes e identidad nueva.

   El segundo es más urgente: no hay ni commits que recuperar. El triaje va con el agente
   `triaje`, sin borrar nada.
2. **La decisión grande: qué se entrega como algoritmo de la tesis.** Documento preparado en
   [`61-decision-entregable-algoritmo.md`](61-decision-entregable-algoritmo.md), con las cuatro
   opciones, lo que cada una implica por objetivo, lo que falta saber antes de decidir y una
   recomendación argumentada. Es para llevar a la conversación con Nicolás, Andrés y Karen.
   Resumen: el mejor clasificador medido no es autosupervisado y el objetivo general exige SSL.
   La recomendación es la opción B (VICReg o Barlow como entregable SSL, condicionada a
   confirmar con más semillas) y, si no se confirma, la opción A.
3. **Preguntarle a Nicolás el resultado de la cola del 2026-10-29.** Siete brazos quedaron
   corriendo desatendidos (`dino-ibot` s43/s44, `dino-ibot-cb`, `moco-v3`, `simsiam-cb`,
   `masked-pixel`, `rotation`) y no hay ninguna observación posterior. Si `dino-ibot` con 3
   semillas cambia el cuadro, cambia la decisión anterior.
4. **Justificar o cambiar B71 como modelo predeterminado del servicio desplegado.** Es el brazo
   que pierde contra DINOv2 público en fotografía de celular con IC que excluye el cero, y es lo
   que hoy ve un usuario real en `Perma`. El servicio ya expone A71 también, así que el cambio es
   de configuración; lo que falta es la decisión y su justificación escrita. Va junto con el
   punto 2.
5. **Cerrar la trazabilidad del modelo en producción.** El `model_sha256` que sirve
   `derm.nicolasmoreno.site` (`7fd581af…`) no aparece en el repo. Pedirle a Nicolás el
   `V9_MANIFEST.json` del release, o versionar la identidad del modelo desplegado. Sin eso no se
   puede acreditar en la tesis qué modelo exacto está en producción.
6. **Decidir si la interfaz de la tesis es `demo/phone/` o `Perma`**, y archivar la otra. Son las
   únicas dos que se solapan de verdad. Y si V9 queda en la cadena —lo está—, exigir que su
   tooling entre al repo.
7. **Reescribir `tab:trazabilidad-fases` del anteproyecto.** La fila 2 dice que el plan nuevo no
   se lanzó y la fila 4 que el prototipo no se implementó. Las dos quedaron falsas, y la segunda
   está refutada con una llamada HTTP. Es texto que va al documento evaluable.
8. **Reconciliar D1.** ¿El banco completo de ISIC 2019/2020 satisface la predicción D1 registrada
   en `docs/27` FASE 5, o es un protocolo distinto? Afecta directamente qué se puede afirmar en
   el capítulo de resultados sobre generalización cross-dominio.
9. **Pedirle a Nicolás** acceso de lectura a los releases del repo, el runner/config PAD-DA
   (requisito de la comparación A/B/C), y que acepte el RUA de DDI si se quiere una cohorte
   externa independiente.
10. **Arreglar el `F821` de `scripts/figures/fig3_masks.py`** y decidir si los 58 `F401` se
    limpian con `ruff --fix` o si la configuración del linter se recalibra.
11. **Correr la suite completa** (`pytest -q` con `PYTHONPATH=src`, `CI_REQUIRE_FULL_DEPS=1`) y los
    `--check` de los generadores `build_*`, para tener una lectura verificada de salud del repo.
12. **Actualizar `docs/01-contexto-y-objetivos.md`** a cuatro objetivos y retirar la mención a
    Kaggle T4, o marcarlo como superado en `docs/README.md`.

Baja de prioridad, ya resuelto en parte: la revisión de si `demo/backend/model_service.py` sirve
el CLS768 simplificado aplica solo a nuestra rama, porque ese `demo/` fue eliminado en `main` y
el servicio desplegado reporta `sha256_verified: true` con receta explícita.

---

*Este documento no cambia ningún gate de ejecución. `execution_enabled`, `launch_ready` y
`training_started` siguen en `false`.*
