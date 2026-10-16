# V8 — Cierre de bloqueos de lanzamiento (punch list de docs 46/51)

## Objetivo
Completar los pendientes documentados en docs/46, docs/49 y docs/51 para habilitar
`execution_enabled=true` y lanzar los 4 métodos de segmentación (SLED, USL-Net,
RPI-Net, TriDermNet-HAGO).

## Por qué / contexto
El usuario pidió avanzar el punch list en este orden:
1. Congelar manifiesto elegible con partición paciente/lesión.
2. Resolver SimCLR (acceso TF oficial o despliegue del parche al clúster).
3. Generar y congelar manifiesto de pseudo-máscaras SLED.
4. Cerrar banco de pesos USL-Net.
5. Medir cuotas reales del clúster.
6. Escribir plan serial concreto del controlador.
7. Habilitar `execution_enabled=true` y lanzar.

Acceso al clúster UNAB autorizado explícitamente por el usuario vía Playwright MCP
(URL: https://unabia.unab.edu.co, ver memoria `infra/unab-cluster-access`). El login
lo hizo el usuario manualmente (nunca se tipeó una contraseña por el modelo).

## Avance vigente del paso 1 (2026-10-16 UTC)

### Comprobación del pin y auditoría de geometría en curso

La comparación programática quedó registrada en `/home/jovyan/work/validation_v8/pin-integrity-recheck-20261016-001/pin_integrity_recheck.json`: `MATCH=false`. El pin histórico y el hash real del register reconciliado siguen siendo distintos; `readiness.review.json` conserva su contenido. Esta excepción no modifica la reconciliación independiente de membresía 583 + 1.334 ni autoriza cambiar gates.

Se preparó y revisó `scripts/audit_v8_evaluation_geometry.py`, publicado en `4511ae8d92a4eff8fe9e4771269487d9d3d7ef35`, SHA-256 de fuente `d7e9066e8f2176340facb454984cc8d26522ef53d97629d53368c7475db06580`. El paquete transferido tiene SHA-256 `561e63285841cb5aa83a6fa85cea47dee6aa6866a9056bd8bdb9bbf30992fcc3`. La ejecución audit-only se lanzó desde la copia canónica identificada por `95a8f60`, con hashes esperados de los ZIP leídos programáticamente del archivo de checksums existente. Inspecciona contenido, correspondencia, dimensiones y máscaras binarias sin extraer miembros ni utilizarlas para aprendizaje; las máscaras vacías o completas son hallazgos de revisión, no exclusiones.

**Estado al registrar este avance: en ejecución, sin resultado final verificado.** El recibo final esperado es `/home/jovyan/work/validation_v8/evaluation-geometry-20261016-001/geometry.audit.json`; el log y el recibo de lanzamiento son `/home/jovyan/work/validation_v8/evaluation-geometry-launch-20261016-001/stdout.log` y `launch.json` en ese mismo directorio. La presencia de un log o de un directorio parcial no acredita éxito. El diagnóstico mantiene `training_consumable=false`, `isolation_global_verified=false` e independencia paciente/lesión sin acreditar.

- [ ] Revisar el recibo final de geometría, su estado, conteos, errores, hallazgos y hashes; no volver a aplicar el filtro provisional ya verificado.
- [ ] Completar y revisar los cinco recibos exigidos por `common/protocol.py`: `exclusions_verified`, `duplicates_adjudicated`, `mask_content_validated`, `exposure_review_complete` y `evaluation_masks_isolated`. Cada uno debe vincular la misma revisión y los hashes de las cohortes de evaluación, referencias y aprendizaje. El informe de geometría por sí solo no sustituye esos recibos ni acredita aislamiento global.
- [ ] Resolver la revisión de similitud: excluir conservadoramente 4.195 rutas pHash y 1.917 por ID base no acredita identidad duplicada ni satisface por sí solo `duplicates_adjudicated`; la unión provisional conserva `duplicate_identity_status=unverified`.
- [ ] Preparar el materializador final después de revisar las evidencias. Entre los puntos de entrada inspeccionados, `apply_v8_exclusion_review.py` produce solo un pool provisional no consumible; falta el materializador de protocolo/manifiestos finales, con hash del código/helper y publicación atómica o rechazo explícito de directorios sin recibo final.
- [ ] Documentar exposición de cada checkpoint/encoder y productor de pseudo-máscaras mediante revisión, hashes y cohortes originales, contrastados con la reserva. No reutilizar artefactos aprendidos V7/Camino B sin trazabilidad explícita de exposición y aprobación compatible con el protocolo; V7 continúa excluido según la política vigente. El pool filtrado actual no demuestra ausencia de exposición histórica.

No se habilitaron gates ni se entrenó. La cuota individual continúa sin acreditar: las observaciones del almacenamiento y recursos compartidos sirven únicamente como referencia aceptada de planificación; faltan presupuestos por etapa.

### Bloqueo de integridad de fuentes (2026-10-16 UTC)

La inspección adicional en UNABIA confirmó que `validation_v8/f1dc0d2-20261014/review_register_stage_v7context_20261015_v2/scripts/build_v8_exclusion_review.py` sí está disponible en una carpeta de preparación. Tiene 12.641 bytes y SHA-256 `02a9b52ba10cf802ee72d3cef78c634fa24bf77dcd340c391f1f8180328b4945`, idéntico al archivo versionado y al miembro del ZIP local. No está en el directorio `validation_v8/f1dc0d2-20261014/scripts/` activo, pero no hace falta volver a transferirlo para una ejecución audit-only desde la ruta de preparación. El hash externo del ZIP v2 sigue discrepando del pin histórico; eso no evidencia una diferencia de código y la causa de la diferencia del contenedor permanece desconocida.

La comparación directa de los archivos en UNABIA con sus referencias aclara que no todos los hashes están en conflicto. El registro actual (`isic2018_exclusion_review_register_20261015.json`) tiene SHA-256 `4b9673833559bba375c59b57686b19278301bd6767926fdd3fac215fdd11fecf`, distinto del pin histórico `4b967833559bba375c59b57686b19278301bd6767926fdd3fac215fddd11fecf`. El informe pHash tiene SHA-256 `13561b98b35b5f4e211859d3dfdfe4a3d03feb1e1c8479a6c8df9f3aa7f20df0`, distinto del pin de readiness `13561b98b35b5f4e211859d3dfde4a3d03feb1c1e8497a9c8df9f3aa7f20df0e`. En cambio, la cola nearest sí coincide con su pin: `e64d9aaff64f9dd5e09e3ca0b35bca8409f5a2183f58e1dfb316bff4e3860b7b`; su `source_report_sha256` apunta exactamente al informe pHash actualmente presente. Además, el hash del inventario de pool citado por el registro (`dc8453545319e66f20d0ffd3c21ddf63d9b0828cbefe35ac2107a27d43d1a6d2`) coincide con el archivo fuente, y el checksum del ZIP oficial Task 1 (`80f98572347a2d7a376227fa9eb2e4f7459d317cb619865b8b9910c81446675f`) coincide con el checksum publicado en UNABIA y el citado por el registro.

La comparación programática verificada de nuevo en UNABIA el 2026-10-16 confirma que el pin histórico del register (`4b967833559bba375c59b57686b19278301bd6767926fdd3fac215fddd11fecf`) sigue sin coincidir con el SHA-256 real del archivo reconciliado (`4b9673833559bba375c59b57686b19278301bd6767926fdd3fac215fdd11fecf`): `MATCH=false`. Se conserva la clasificación de excepción de integridad ya registrada; la búsqueda acotada de un contenido cuyo hash fuera el pin terminó sin hallazgo y no se reabre. La afirmación de coincidencia publicada en `367a4a0` fue incorrecta y queda corregida mediante un commit posterior. `readiness.review.json` y todos los gates permanecen intactos. La reconciliación de membresía 583 + 1.334 conserva su validez y su evidencia independiente.

El aplicador candidato se transfirió por ZIP a UNABIA (SHA-256 `1f2cfdeb33870f7af65cd146e717115aee51cba79ffdf902c008b38ca667b10c`) y pasó tres pruebas de contrato allí. El intento real de materializar las 1.917 exclusiones abortó en el preflight por discrepancia del pin SHA-256 del registro frente al readiness histórico; no produjo directorio de salida ni manifiestos. Este aborto es distinto del fallo previo del generador. No se inició adjudicación visual ni se congelaron manifiestos.

El generador había abortado en la fila pHash 2235: `train_id=ISIC_0014591` frente a `image_id=ISIC_0014591_downsampled` del pool. La corrección en `scripts/build_v8_exclusion_review.py` compara el ID base quitando únicamente el sufijo terminal `_downsampled`, conservando los IDs completos y las rutas originales. El rerun documentado a continuación pasó; no aplicó exclusiones ni reconcilió por sí solo el pin histórico.

- [x] Ejecutar en UNABIA el generador audit-only con el pool, ZIP oficial de evaluación y reporte pHash existentes. La salida `validation_v8/f1dc0d2-20261014/isic2018_exclusion_review_register_reconciled_20261016.json` mide 9.271.726 bytes y su SHA-256 es `4b9673833559bba375c59b57686b19278301bd6767926fdd3fac215fdd11fecf`, idéntico al registro actual `isic2018_exclusion_review_register_20261015.json`. El generador informa `consistency.all_checks_passed=true`; los conteos esperados y observados coinciden: pool 40.896 registros, ZIP 2.594 imágenes, 1.917 cruces por ID base (583 stems exactos y 1.334 con sufijo terminal `_downsampled`), 15.702 pares candidatos pHash, 544 IDs de evaluación y 4.195 rutas de pool. Hashes de fuentes registrados: pool `dc8453545319e66f20d0ffd3c21ddf63d9b0828cbefe35ac2107a27d43d1a6d2`, ZIP `80f98572347a2d7a376227fa9eb2e4f7459d317cb619865b8b9910c81446675f` e informe pHash `13561b98b35b5f4e211859d3dfdfe4a3d03feb1e1c8479a6c8df9f3aa7f20df0`. Esta escritura deja evidencia reproducible del generador, pero no aporta un registro distinto ni constituye manifiesto ejecutable; `executable=false`, `manifest_frozen=false`, `execution_enabled=false`, `launch_ready=false`, `training_started=false` e independencia paciente/lesión `unresolved`. Se conservará como copia de auditoría, no como registro canónico.
- [x] Extender el aplicador para consumir el recibo audit-only y materializar la unión conservadora en un pool provisional nuevo, sin editar pins de readiness. UNABIA verificó el recibo de aplicación y los hashes/tamaños de sus dos salidas; el pool pasó de 40.896 a 34.784 filas. La identidad duplicada y los vínculos fiables siguen pendientes antes de congelar manifiestos.
- [x] Reconciliar en UNABIA los IDs de evaluación contra el register reconciliado y el pool provisional. El cruce reprodujo 583 IDs exactos y 1.334 IDs que coinciden solo al retirar el sufijo terminal `_downsampled`; ambos conjuntos coinciden exactamente con sus propuestas y los 1.917 IDs están ausentes del pool provisional. Informe: `/home/jovyan/work/validation_v8/isic2018-pool-reconciliation-20261016-003/isic2018_pool_id_exclusion_reconciliation.json`, SHA-256 `1ef620c362658ae4e364cd4db811a25eb5a411314fa7130670e710282bfc5dce`. La exclusión de los 1.334 candidatos es conservadora y no acredita equivalencia del contenido de imagen.
- [ ] Antes de promover el pool provisional a manifiesto consumible, verificar que el recibo final registre hash del código/helper y que la publicación sea atómica o invalide directorios sin recibo final. La revisión independiente informada no reportó bloqueo de filtrado ni fuga de datos, pero su dictamen no está persistido como informe auditable en el repositorio.

La materialización provisional por ruta está verificada; no equivale a congelar el inventario ni a acreditar identidad. El recibo mantiene en `false`: `manifest_frozen`, `training_consumable`, `executable`, `execution_enabled`, `launch_ready`, `training_started`, `patient_independence_accredited`, `patient_lesion_groups_preserved`, `similarity_adjudication_complete`, `learning_exposure_review_complete`, `images_decoded`, `expert_masks_read`, `phash_recomputed` y `original_inputs_modified`. `patient_lesion_independence=unresolved`; duplicación visual/clínica no acreditada. No hubo entrenamiento. La limpieza sintética permanece justo antes de la primera corrida real.

**Verificación remota de adjudicación pHash (2026-10-16 UTC):** se recuperó la terminal operable de UNABIA. El ZIP `v8-phash-adjudication-green-20261016.zip` (SHA-256 local `5d86f7b6702d3fd1cdcd845901c8d3126d9a003ea47bb25ac25a82d9deeb7064`) coincide byte a byte con `/home/jovyan/work/v8-phash-adjudication-green-20261016.zip`; los dos archivos extraídos coinciden con sus miembros del ZIP. La prueba dirigida `tests/segmentation_v8/test_v8_phash_pair_adjudication.py` pasó en UNABIA: **24 passed**. El generador audit-only produjo recibo en `/home/jovyan/work/validation_v8/phash-adjudication-audit-20261016-001/`, con 40.896 filas de pool, 15.702 decisiones de pares, 544 IDs de evaluación, 4.195 rutas pHash, 1.917 propuestas por ID base, unión conservadora de 6.112 rutas y 34.784 restantes. Fuentes: registro reconciliado `/home/jovyan/work/validation_v8/f1dc0d2-20261014/isic2018_exclusion_review_register_reconciled_20261016.json`; informe pHash SHA-256 `13561b98b35b5f4e211859d3dfdfe4a3d03feb1e1c8479a6c8df9f3aa7f20df0`; pool SHA-256 `dc8453545319e66f20dffd3c21ddf63d9b0828cbefe35ac2107a27d43d1a6d2`. La salida audit-only expresa una decisión conservadora de elegibilidad por ruta; mantiene `duplicate_identity_status=unverified` y no se aplica por sí sola al inventario.

- [x] En UNABIA, verificar hash del paquete transferido, identidad de sus dos archivos extraídos, ejecutar la prueba dirigida y generar/inspeccionar el recibo audit-only en una ruta nueva. Evidencia y fuentes: [docs/46](../../docs/46-pendientes-por-metodo-v8.md#verificación-remota-y-salida-audit-only-2026-10-16-utc).
- [x] Transferir/verificar el aplicador revisado y aplicar el recibo audit-only a un pool provisional nuevo, sin cambiar readiness histórico. El ZIP `v8-phash-exclusion-application-green-20261016-r2.zip` tiene SHA-256 `476cc6e875e6f6670de98e83e6f09b33a8301c9b024bb498f98d7a349c5f997f`; fue verificado y extraído en `/home/jovyan/work/validation_v8/phash-exclusion-application-green-20261016/`. La suite dirigida pasó exclusivamente en UNABIA (**8 passed, 5 subtests passed**). El recibo `/home/jovyan/work/validation_v8/phash-exclusion-application-20261016-001/receipt.json`, schema `v8_provisional_exclusion_application.v2`, registra 40.896→34.784 filas: 1.917 exclusiones por ID base (583 exactas y 1.334 `_downsampled`), 4.195 rutas pHash asociadas a 15.702 pares, cola nearest de 1.490 filas y unión de 6.112. Se verificaron hashes/tamaños de las dos salidas y el conteo de filas del pool; `original_inputs_modified=false`. No se reescribieron los pins históricos ni se congeló el manifiesto.

### Decisión adoptada y entrega candidata

El usuario aprobó `retrospective_public_benchmark`, manteniendo `patient_independence_accredited=false`. Sustituye las declaraciones históricas posteriores sobre reformulación pendiente de aprobación. El alcance es evaluación retrospectiva pública de 2.594 imágenes/máscaras Task 1; no acredita pacientes nuevos ni test ciego original.

- [x] Registrar aceptación del alcance limitado y conservar la limitación clínica.
- [~] Preparar contrato/evaluador candidatos: protocolo común en config/gate/manifest/export; recibos hash de exclusiones, adjudicación, máscaras, exposición y aislamiento; IDs exactos, `>=0.5`, bootstrap por imagen y fallos en denominador original.
- [x] Revisión estática y suite nueva de contratos en copia aislada de UNABIA: 15 passed, 2 warnings. No se ejecutaron localmente; no equivale a validación científica ni a habilitación de la corrida.
- [x] Integrar el recibo audit-only al aplicador y materializar/verificar la unión de 6.112 rutas excluidas en un pool provisional nuevo. Esta aplicación por ruta no acredita identidad duplicada; `similarity_adjudication_complete=false`.
- [ ] Revisar vínculos de similitud y exposición al aprendizaje; validar contenido/geometría de los 2.594 pares y aislar las máscaras expertas. Después materializar y congelar el protocolo/manifiestos/hashes reales. Independencia paciente/lesión continúa sin acreditar.
- [ ] Preparar pseudo-máscaras, planes seriales y presupuestos medidos. No copiar épocas/recursos V7.
- [ ] Limpiar pruebas sintéticas justo antes de la corrida real, conservando recibos.

Se recuperan semilla/config/revisión fijadas, artefactos inmutables, hashes/procedencia y selección de checkpoint independiente de máscaras finales. W&B registra entrega solo con acuse; offline queda pendiente. `execution_enabled=false`, `launch_ready=false`, `training_started=false`.

### Clon canónico en UNABIA (2026-10-16 UTC)

Al verificar un claim de avance (commit local `fb7f806`) directamente en UNABIA se encontró que **no existe un checkout sincronizado del proyecto en el clúster**: `/home/jovyan/work/proyectonp/` tiene más de 30 clones locales acumulados (`repo_s44`, `repo_v7`, `repo_audit_v8`, `repo_four_methods_v8_*`, etc.), ninguno en los commits `fb7f806` ni `ee95504`, y todos clonados unos de otros por filesystem local — nunca desde GitHub (no hay `credential.helper`, `~/.git-credentials` ni `~/.ssh` configurados en la cuenta).

Se resolvió creando `/home/jovyan/work/proyectonp/repo_v8_canonical`:
- Empaquetado local con `git archive HEAD` de `src/ scripts/ tests/ docs/ odd/ openspec/ cluster/ demo/` + configs raíz (código, no datos/resultados pesados: se excluyó deliberadamente `resultados/`, `plan-segmentacion/`, `assets/`, `.git/` histórico — ver `CANONICAL_SNAPSHOT_MANIFEST.txt` dentro del propio clon para el detalle exacto de inclusión/exclusión).
- Transferido por carga binaria real vía el navegador de archivos de JupyterLab (no pegado de texto/base64) — tarball de 5.1 MB.
- Integridad verificada por SHA-256 idéntico entre local y UNABIA: `dcce8dee49caaafc709997f7fb54f2c4d6dee7bc85b044252fcee673323204b7`.
- `git init` + commit único etiquetado `"snapshot of experiment/modular-audit-v8 @ fb7f806"` para que `git log`/`git rev-parse` funcionen como referencia verificable (no es un clon con historial completo).
- Confirmado presentes: `scripts/apply_v8_exclusion_review.py`, `tests/segmentation_v8/test_v8_exclusion_application.py`.

**A partir de ahora, toda verificación o ejecución de scripts v8 en UNABIA debe hacerse en esta ruta**, no en ningún clon viejo. Los clones anteriores se dejaron intactos como sedimento histórico, sin borrar.

### Clasificación del pin y reconciliación actual (2026-10-16 UTC)

La clasificación previa de `readiness_pin_reconciliation` encontró una diferencia benigna en la cola pHash (1.490/1.490 filas idénticas) y ninguna diferencia sustantiva en los conjuntos comparados. El register conserva una excepción de integridad: el archivo `/home/jovyan/work/validation_v8/f1dc0d2-20261014/isic2018_exclusion_review_register_reconciled_20261016.json` tiene SHA-256 real `4b9673833559bba375c59b57686b19278301bd6767926fdd3fac215fdd11fecf`, distinto del pin histórico `4b967833559bba375c59b57686b19278301bd6767926fdd3fac215fddd11fecf` (`MATCH=false`, verificado programáticamente de nuevo el 2026-10-16). La ronda de búsqueda de la causa quedó cerrada sin hallazgo; el registro reconciliado sigue como fuente de trabajo autorizada con esta excepción explícita. La afirmación contraria en `367a4a0` fue falsa. No se editó `readiness.review.json` ni ningún gate.

La reconciliación no congela el manifiesto: el recibo de aplicación sigue con `manifest_frozen=false` y `training_consumable=false`. Continúan sin acreditarse la independencia paciente/lesión, la equivalencia de contenido para los candidatos `_downsampled`, la revisión de exposición al aprendizaje y la adjudicación completa de similitud.

### Script de metadata oficial de agrupación (2026-10-16 UTC)

Se encontró `scripts/audit_v8_official_group_metadata.py` (499 líneas, completo, antes sin ejecutar y sin trackear en git). Descarga metadata oficial de ISIC (`patient_id`, `lesion_id`) para las 4 colecciones relevantes desde `api.isic-archive.com`/S3 con validación estricta de host y calcula cobertura/solapamiento contra el pool. Diseñado para producir evidencia, no para resolver por sí solo: su reporte fija `patient_lesion_independence="unresolved"`; nunca descarga imágenes/máscaras ni aplica exclusiones/entrenamiento. Agregado a `repo_v8_canonical` en commit `95a8f60`, verificado por hash `f1c780d8017e9a46406124918a94d668b94dda1576b74dac7e16b7bf9617293c`. Se ejecutó en UNABIA con el pool derivado del recibo pHash y mapeo explícito `ham10000_p1=212`, `ham10000_p2=212`. Salida: `/home/jovyan/work/validation_v8/official-group-metadata-audit-20261016-002/official_group_metadata_audit.json`; sin `AuditError`. La cobertura de `patient_id` continúa incompleta y el campo de independencia permanece `unresolved`.

**Corrección (2026-10-16 UTC):** Codex reportó que `repo_v8_canonical` no tenía `readiness.review.json` ni el registro histórico/recibo pHash/cola pHash — correcto, y error mío: excluí `resultados/` entero por tamaño sin notar que `readiness.review.json` vive en `resultados/preparacion_v8/readiness_20261014/` (633K, trivial). Se agregó en commit `e44ff10` (verificado por hash `19f86571a5e7e22eca42048d0bb07f573ee7d269c42c633454c83e83761d1d5b` igual entre local y UNABIA), con nota aclaratoria en commit `1c20d87`. El resto de `resultados/` (83M) y `plan-segmentacion/` (82M) siguen excluidos a propósito. El recibo pHash, el registro histórico de exclusión y la cola pHash **nunca estuvieron versionados en git** — son artefactos generados en UNABIA bajo `/home/jovyan/work/validation_v8/...`; sus rutas exactas hay que derivarlas del campo `inputs` del `receipt.json` ya producido ahí, no asumirlas.

La auditoría de geometría/contenido de máscaras permanece como evidencia separada. El audit de metadata oficial y el nuevo informe de reconciliación de membresía documentan los cruces de IDs y la aplicación provisional, pero no certifican equivalencia de imagen ni independencia clínica. Los gates continúan cerrados y `readiness.review.json` no fue modificado.

El candidato incorpora la corrección de revisión sobre linaje de checkpoints: protocolo, manifiesto/gate de entrenamiento y cohorte deben coincidir en pesos, sidecar, configuración y recibo del export. Un checkpoint histórico sin linaje embebido no se admite cambiando el sidecar. Los contratos de mismatch/sustitución se validaron en la suite aislada de UNABIA.

La revisión posterior exige ese recibo también al refinar, ligado a manifiesto/config/gate/export; el evaluador registra revisión del snapshot y hashes propios/de métricas. Los contratos para export histórico, tipo incorrecto de checkpoint, recibo sustituido y procedencia del código se probaron en UNABIA. La evaluación científica y los cierres materiales continúan pendientes.

### Validación aislada de contratos en UNABIA (2026-10-15)

Ruta: `/home/jovyan/work/validation_v8/retrospective-public-benchmark-20261015/`. Paquete ZIP r4: SHA-256 `dd87871b023897160d389e0037216bd74fe2cb85b2945c792aa502b31e78a6de`, 20 rutas seguras. Se corrigió una omisión del bundle: la primera copia carecía del módulo de métricas compartido y sus cuatro dependencias, con resultado 13 passed/2 failed. R4 incorporó esos archivos; la copia base `f1dc0d2-20261014` quedó intacta.

Comando: `PYTHONPATH=src ./.validation_deps/bin/pytest -q tests/segmentation_v8/test_retrospective_protocol.py`. Resultado: **15 passed, 2 warnings en 2,07 s**. Las advertencias corresponden a opciones heredadas `timeout`/`timeout_method` desconocidas por pytest en ese entorno; la suite no acredita su activación. Se validaron contratos con fixtures, sin entrenamiento, evaluación científica, W&B, publicación de métricas ni limpieza sintética.

Al corte de esta actualización (2026-10-15), la independencia por paciente no estaba acreditada y el manifiesto seguía sin congelar. La aplicación provisional posterior del 2026-10-16 está registrada en la sección vigente de materialización por ruta; todavía faltan la revisión de similitud/exposición, contenido/geometría, aislamiento, pseudo-máscaras, planes seriales y presupuestos. La limpieza sintética sigue justo antes de la primera corrida real. `readiness` no se modificó; `execution_enabled=false`, `launch_ready=false` y `training_started=false`.

Se completó en UNABIA el cruce adicional de metadatos oficiales actuales de las
colecciones ISIC 63/65/70/212 y cinco suplementos oficiales con el inventario
histórico. El informe `official_group_metadata_audit.json`, en
`/home/jovyan/work/validation_v8/f1dc0d2-20261014/official-metadata-crosswalk-20261015-final/`,
ocupa 9.748 bytes; SHA-256
`e95774848d3f7c15fdff4d1795ba84746aecef1136e8edd9e4db92383cd69f6`.
Las fuentes y recibos están conservados: no repetir descargas ni el cribado pHash.
El detalle y las fuentes primarias quedan en [docs/46](../../docs/46-pendientes-por-metodo-v8.md#cruce-adicional-de-metadatos-oficiales-isic-en-unabia-2026-10-15-utc)
y en `data.isic2018_independent_evaluation_id_overlap_review` del recibo de preparación.

La colección 63 cubre las 2.594 imágenes reservadas, con `lesion_id` para 1.098 y
`patient_id` para 390; la colección 65 enlaza 2.593. Dentro de la colección 65,
omitiendo coincidencias exactas del ID de imagen, aparecen 3 cadenas de lesión
compartidas (3 filas de evaluación y 3 filas distintas del pool) y 4 cadenas de
paciente (4 y 4). El suplemento oficial ISIC2019 Training también nomina 3 cadenas
de lesión entre imágenes distintas; esos hallazgos no se suman como casos nuevos.
Su identidad sigue sin confirmar. La agrupación oficial ISIC2018 Task 3 no enlaza
esta cohorte de evaluación. Los IDs ausentes y las comparaciones entre espacios
de identidad sin equivalencia acreditada impiden demostrar independencia completa.

En la consulta dirigida separada de los 677 IDs sin coincidencia de ID base,
los sidecars ISIC2019 Full Resolution y Secondary Training enlazan 189 imágenes
cada uno y no aportan `patient_id` ni `lesion_id` no vacíos. ISIC2020 y el test
oficial ISIC2019 enlazan cero. La comparación de lesiones con las 10.991 filas
históricas del pool ISIC2019 da cero cadenas compartidas y cero imágenes
candidatas con lesión compartida. Estos ceros no acreditan independencia ni
ubican dentro de esos 677 los candidatos 3/4 de la cohorte completa.

El cruce dirigido de [HAM10000 en Harvard Dataverse, DOI 10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T), versión 4.0, sí se completó después en UNABIA. El 403 HTML afectaba al listado general por API; los endpoints directos funcionaron al identificar los `fileId` en la página y sus archivos con Playwright, siguiendo la captura del usuario. Solo se descargaron los dos TSV pequeños, sin ZIP de imágenes; sus tamaños y SHA-256, junto al hash del sidecar Task 1 ya disponible, están en [docs/46](../../docs/46-pendientes-por-metodo-v8.md#cruce-dirigido-ham10000--isic2018-task-3-harvard-dataverse-v40-2026-10-15-utc) y el recibo de preparación.

El recibo completo, generado y parseado en el clúster, está en `/home/jovyan/work/validation_v8/f1dc0d2-20261014/official-metadata-crosswalk-20261015-final/harvard_v4_source/ham10000_crosswalk_summary.json`, junto a los TSV fuente. HAM10000: 10.015 filas/IDs únicos, 7.470 lesiones distintas y 1.956 grupos con más de una imagen; Task 3 test: 1.512 filas/IDs únicos, 1.223 lesiones y 226 grupos con más de una imagen. Ambos carecen de `patient_id`. Task 1 colección 63: 2.594 filas/IDs únicos y 390 filas con paciente, correspondientes a 350 IDs de paciente distintos.

Los tres cruces por pares entre Task 1 completa, HAM10000 train y Task 3 test dieron cero IDs de imagen exactos y cero cadenas de lesión compartidas. Los subconjuntos históricos de 2.011 candidatos y de 583 IDs coincidentes con el inventario también dieron cero `image_id` frente a ambos TSV; no sustituyen el filtro vigente de ID base. Dataverse describe una procedencia compartida entre clasificación Task 3 train y validación/test; esa descripción no proporciona un puente de identidad paciente. Task 3 test tiene sus propios IDs de lesión y sus etiquetas de clasificación no se incorporan a segmentación V8.

**Los ceros no acreditan independencia clínica:** falta `patient_id` en HAM10000 y no hay equivalencia de identidades entre sistemas. Tampoco adjudican los candidatos positivos del cruce anterior. Al cierre de esta auditoría histórica las exclusiones seguían sin aplicar; luego se verificó la aplicación provisional de 6.112 rutas en UNABIA. El paso 1 continúa abierto, `manifest_frozen=false`, y la limpieza sintética queda pendiente al final.

La auditoría retrospectiva local del 2026-10-15 distingue dos protocolos: clasificación V7 agrupó HAM10000 por `lesion_id` mediante `StratifiedGroupKFold`, con selección interna y exclusiones del pool SSL; sus JSON finales s42/s43/s44 comparten `n_eval=2004` y `eval_ids_sha256=6ff4c17f04a9eae2f0d058aaa41084ea65b9aff91e4d5e5b05c924b4465a2fc6`. Camino B asignó pseudo-validación por hash de `dataset:image_id` con semilla 42: su cache tiene 58.457 imágenes y 1.169 de validación. El registro S4-A2 real es `plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2/run.json`; declara entrenamiento completo y régimen transductivo sin etiquetas. El análisis 722 dentro/1.872 fuera del pool procede separadamente de `2026-09-21-final/evaluacion_isic2018/summary.json`. Ninguno certifica separación clínica entre fuentes. Rutas, claves y limitaciones quedan en [la auditoría de docs/46](../../docs/46-pendientes-por-metodo-v8.md#auditoría-retrospectiva-de-splits-y-utilidad-de-ham10000-2026-10-15) y `data.step_1_freeze_gate_review.retrospective_split_audit_20261015` del recibo.

Los metadatos HAM sí sirven para agrupar fotografías de una lesión, excluir IDs/grupos conocidos y trazar procedencia. Sus 10.015 filas oficiales tienen lesión y carecen de paciente; no bastan para acreditar independencia paciente/lesión de Task 1 frente a todas las exposiciones de entrenamiento. Falta un enlace de paciente comparable, completo y trazable en ambos lados. La cobertura parcial por fuentes y los ceros de cadenas exactas no lo reemplazan. La reconciliación actual confirma 1.917 exclusiones aplicadas al pool provisional: 583 coincidencias exactas y 1.334 coincidencias solo tras retirar `_downsampled`. Estas últimas no son equivalencias de contenido verificadas; quedan 677 IDs de evaluación sin coincidencia de ID en el pool. Los 2.011 candidatos previos se conservan solo como historial. No se habilitaron gates ni se congeló el manifiesto.

### Contraste de protocolo con fuentes primarias (2026-10-15)

Se corrigió una confusión de tareas: Task 1 tiene 2.594 imágenes de entrenamiento y 2.594 máscaras de lesión; las 12.970 máscaras, cinco atributos por imagen, corresponden a Task 2. Sus etiquetas de entrenamiento eran públicas; los 100 casos de validación y los 1.000 de test del desafío original tenían máscaras ocultas. La cohorte de 2.594 usada aquí es un benchmark público previamente consultado, no el test ciego del desafío. Fuentes: [artículo ISIC2018, §§2.1–2.2](https://arxiv.org/html/1902.03368v1), [aclaración del organizador](https://forum.isic-archive.com/t/number-of-images-and-ground-truths/335) y [Task 1 oficial](https://challenge.isic-archive.com/landing/2018/45/).

La evidencia distingue separación de imágenes, lesiones y pacientes de validación externa por centro/dataset. HAM10000 ayuda a agrupar fotos mediante `lesion_id`, pero no tiene `patient_id`; la lesión adicional previamente hallada pertenece a los 1.917 solapamientos por ID base ya propuestos para exclusión. Los sidecars que enlazan 189 de los 677 IDs restantes no aportan ningún paciente o lesión no vacío. No corresponde repetir la descarga ni atribuir independencia clínica a esos ceros. [CLAIM, ítem 20](https://rsna.github.io/claim/item-20.html) recomienda separar por paciente o nivel superior y justificar niveles inferiores. El hallazgo institucional externo de Task 3 en el artículo del desafío no completa los pacientes de Task 1.

La segmentación transductiva sin máscaras puede ser válida para la cohorte objetivo disponible; exige declarar exposición a sus imágenes y no equivale a generalización inductiva en pacientes nuevos ([Kamnitsas et al.](https://arxiv.org/html/2107.08964v2)). Clasificación V7 agrupó por lesión y Camino B usó hash por imagen y régimen transductivo; no se invalida globalmente ese historial ni se infiere consulta de máscaras adicional. El detalle y las limitaciones están en [docs/46](../../docs/46-pendientes-por-metodo-v8.md#contraste-con-literatura-y-alcance-defendible-de-la-evaluación-2026-10-15).

La recomendación pendiente es fijar protocolo/configuraciones antes de puntuar, conservar las exclusiones por ID base en el pool provisional, revisar similitud/exposición y aplicar grupos conocidos solo donde sean fiables. Los 1.334 IDs con sufijo `_downsampled` se excluyeron de forma conservadora; el cruce de nombres no prueba equivalencia de contenido. Con ese alcance se informaría evaluación retrospectiva de benchmark público e independencia por paciente sin acreditar. Afirmar independencia clínica por paciente/centro necesita cohorte nueva o enlace completo defendible y test ciego adecuado sin consulta previa. No se cambian predicados ni flags; la limpieza sintética sigue al final.

**Tarea 1 abierta:** adjudicar los vínculos positivos del mismo ámbito usando los
identificadores únicamente en UNABIA, conservar las 2.594 imágenes para evaluación,
materializar/verificar las exclusiones del lado de aprendizaje y completar pHash,
grupos y exposición antes de congelar. Sigue vigente la propuesta de excluir
1.917 filas por ID base: 583 exactas y 1.334 derivados terminales `_downsampled`.
Los cero SHA-256 exactos del cribado histórico de 2.011 no revocan la identidad
base de esos derivados reducidos. Solo 677 IDs quedan sin enlace de ID base;
los estratos y cruces anteriores sobre 2.011 se conservan con alcance histórico.
El nuevo informe compara instantáneas actuales y no verifica la pertenencia al ZIP
histórico; la comprobación previa de 2.594 IDs ZIP/API iguales queda separada.
No se aplicaron exclusiones ni se congelaron manifiestos; `execution_enabled=false`,
`launch_ready=false`, `training_started=false`. TriDerm/SLED y los planes seriales
esperan el paso 1; la limpieza sintética se mantiene justo antes de la corrida real.
La cuota agregada continúa como referencia de planificación, sin certificar cuota
individual; no se reabre la búsqueda de herramientas ni se resuelve `cpu.max`.

Los apartados siguientes conservan el historial de tareas y observaciones; las
descargas, forwards y despliegues ya acreditados no se repiten por sus estados antiguos.

## Estado por tarea

- [~] **Tarea 5 — recursos observados; cuota individual sin acreditar.** Observación histórica por terminal
  (2026-10-14): 48 CPUs, 123 GiB RAM (115 GiB libres), 2× RTX PRO 4000 Blackwell
  24 GB cada una (0% uso), cgroup CPU/memoria en `max`/`max 100000` en esa lectura,
  disco compartido 40T (1% usado). Esa lectura no acredita cuota individual ni resuelve
  las discrepancias históricas de `cpu.max`. Los valores quedaron registrados
  con lo ya registrado en `resultados/preparacion_v8/readiness_20261014/readiness.review.json`
  (bloque `resources`). Se aceptan como referencia de planificación, sin afirmar
  disponibilidad actual garantizada. Falta traducirla en límites concretos de
  timeout/RSS/VRAM por etapa para el plan del controlador (tarea 6); no se repiten
  mediciones sin un motivo material nuevo.

- [~] **Tarea 1 — congelar manifiesto elegible.** Descarga y aplicación provisional
  disponibles; revisión de evidencia y materialización final pendientes. El siguiente
  detalle conserva el historial de preparación, superado donde se indica. Ver memoria Engram
  `proyecto-grado-ssl-lesiones/v8-manifiesto-elegible` para el detalle completo:
  - Ubicación real de datasets confirmada: `/home/jovyan/work/datos/{ham10000,isic2019_fullres,isic2020}/`
    con sus CSV de metadata y columnas exactas.
  - Contrato exacto de `src/dermssl/segmentation_methods/common/data.py` leído byte
    a byte (ALLOWED/FORBIDDEN, eligibility block, recomputo de SHA-256 en cada carga).
  - **Bloqueo histórico de adquisición, superado por la descarga**: la "evaluación anatómica independiente" que
    exige docs/39:82 necesita máscaras expertas de referencia (tipo ISIC2017/2018
    Lesion Boundary Segmentation Challenge). En aquella observación faltaban máscaras
    expertas y solo se disponía de máscaras U-Net de QA. Posteriormente se descargaron
    los ZIP oficiales y se verificaron checksums y correspondencia de 2.594 IDs
    imagen/máscara. La validación completa de contenido/geometría sigue en ejecución;
    no se presenta esa correspondencia como aislamiento o independencia acreditados.
  - **Decisión del usuario (2026-10-14): descargar ISIC2018 Task 1 oficial.**
    Verificado con HEAD request (sin login, HTTP 200): `ISIC2018_Task1-2_Training_Input.zip`
    y `ISIC2018_Task1_Training_GroundTruth.zip` (27.402.895 B) en
    `https://isic-challenge-data.s3.amazonaws.com/2018/`. Descarga lanzada en segundo
    plano en el clúster (`nohup`+`disown`, sobrevive aunque se cierre la sesión del
    navegador) hacia `/home/jovyan/work/isic2018_task1/` (NO en `/home/jovyan/work/datos/`,
    que es de solo lectura — dueño `nobody`, permiso denegado al escribir ahí).
    El script de descarga calcula `checksums.sha256.txt` y escribe `.done` al terminar.
    **Corrección (2026-10-14, ~13:52 UTC): el Content-Length real del Training Input es
    11.165.358.566 B (~11,2 GB), no 1,1 GB — error de lectura propio en la primera
    verificación (se perdió un dígito). A los ~45 min llevaba 2,1 GB (~19%); a ese ritmo
    son varias horas, no ~30 minutos.** Ese estado parcial es histórico: la descarga
    y la correspondencia de 2.594 IDs ya están documentadas. La reserva vigente es
    de evaluación bajo `retrospective_public_benchmark`, con máscaras separadas del
    aprendizaje; faltan cierre de geometría, similitud, exposición, aislamiento y
    protocolo/manifiestos finales. No volver a descargar ni repetir la aplicación provisional.

- [~] **Tarea 2 — resolver SimCLR.** Parcialmente resuelto (2026-10-14):
  - Confirmado (3 vías independientes: API JSON del bucket GCS, ruta de objeto
    directa, mirror `simclr-gcs`) que el acceso anónimo al checkpoint TF oficial
    está bloqueado por IAM de Google Cloud (401 listado, 403 objeto), no por una
    URL incorrecta. Requiere credenciales GCP reales — fuera de alcance sin que
    el usuario las provea.
  - **Parche desplegado al clúster con éxito** (autorizado por el usuario):
    `v8_simclr_same_profile_eed86c8.patch.zip` reconstruido desde el checkout
    local (los 3 archivos no cambiaron desde la revisión previa `7a586059`),
    transferido por terminal en bloques base64 (el intento con una sola línea
    de 10KB se corrompió; fragmentado en 6 bloques de 2000 caracteres funcionó),
    extraído en `~/work/public_ssl_v8_preparation/runtime_simclr_same_eed86c8/`,
    verificado hash por hash contra el repo local (match exacto). Ver doc 53.
  - Sigue pendiente: `scientific_execution_authorized=false` — desplegar código
    no es lo mismo que autorizar correr algo sobre él; y la equivalencia TF
    numérica real sigue sin verificar (solo el candidato HF está validado).
- [ ] **Tarea 3 — manifiesto pseudo-máscaras SLED.** Sin avance (depende de tarea 1).
- [x] **Tarea 4 — cerrar banco de pesos USL-Net. Cerrado y verificado end-to-end
  (2026-10-14).** Una sesión paralela (mismo usuario) preparó en paralelo el
  candidato de los tres orígenes (`plan-segmentacion/metodos-v8/source-bank.usl-net.candidate.json`,
  docs/54-banco-uslnet-preparacion-v8.md) pero no pudo verificar en el clúster —
  Jupyter devolvió `502 Bad Gateway`. Esta sesión sí tuvo el clúster accesible y
  completó la verificación real: USL-Net exige exactamente `{simclr, mocov1,
  mocov2}` (`usl_net/model.py`). MoCo v1/v2 ya estaban convertidos. Para SimCLR: se
  completó el árbol de paquete `dermssl` que faltaba en el runtime desplegado
  (checkpoints.py, dino.py, data.py, transferidos por base64 fragmentado y
  verificados hash por hash), se descargó el checkpoint HF al clúster
  (tamaño/SHA-256 exactos), se corrió `prepare_public_ssl_weights_v8.py` de
  verdad — el `checkpoint_sha256` resultante coincide byte a byte con la validación
  local previa y con la de la sesión paralela (conversión determinística). Se
  construyó `source-bank.usl-net.reviewed-candidate.json` y se verificó con
  `source_bank()` real: los 3 encoders cargan con `strict=True`, 23.508.032
  parámetros cada uno. Ver doc 54 (este, el que relata la ejecución real).
  `execution_ready` sigue en `false` — cargar pesos no autoriza entrenar.
- [~] **Tarea 6 — plan serial concreto del controlador.** `scripts/run_segmentation_controller_v8.py`
  leído y entendido (espera plan JSON con `schema_version=1`, `protocol="w3-four-methods-v8"`,
  steps con manifest/gate/config/out, límites `timeout_seconds`/`max_rss_mb` por
  config). Borradores de dependencias, recursos y seis configs deshabilitados preparados en
  `plan-segmentacion/metodos-v8/launch-preparation/` (commit `a983858`). El plan
  ejecutable con rutas/hashes/gates reales sigue dependiendo de tarea 1.
- [ ] **Tarea 7 — habilitar execution_enabled y lanzar.** Bloqueada por todo lo anterior.

## Verificación aplicada
El cierre del paso 4 está registrado en el commit `c55b6b146d78d4ab9e2a7fb6a1743e72c3fb134b`
y en [docs/54-usl-net-banco-pesos-v8.md](../../docs/54-usl-net-banco-pesos-v8.md):
SimCLR HF se transfirió y convirtió en el clúster con el SHA-256 esperado, y
`source_bank()` cargó exactamente `simclr`, `mocov1` y `mocov2` con
`strict=True` (23.508.032 parámetros por encoder). No se entrenó. El recibo
`readiness.review.json` conserva `launch_ready=false` y `training_started=false`.

El recibo versionado no documenta un forward sintético de 224×224 ni la salida
`[1, 9216, 14, 14]`; por ello ese subcheck no se declara aprobado. Esta continuación
no repitió transferencias, conversiones, cargas, auditorías, pruebas ni entrenamientos.
El estado del login y de la descarga ISIC no se inspeccionó en vivo en esta sesión.

## Próximo paso
No repetir la transferencia, conversión ni carga conjunta del banco USL-Net: esa parte
de la tarea 4 ya quedó cerrada en el commit `c55b6b1`. Cuando haya acceso seguro al
clúster, comprobar primero si existe un recibo del forward sintético 224×224; solo si
falta, hacer una única pasada sin imágenes reales ni entrenamiento y registrar si la
salida es `[1, 9216, 14, 14]`.

Las tareas 1, 2, 3 y 6 siguen abiertas: terminar y revisar la reserva anatómica y el
manifiesto elegible (incluida la descarga ISIC2018 ya iniciada), mantener explícita la
falta de equivalencia numérica con el checkpoint TensorFlow original de Google,
congelar el manifiesto de pseudo-máscaras SLED, y concretar el plan serial con sus
límites por etapa. Mantener `execution_enabled=false`; no lanzar corridas hasta que
los bloqueos científicos y operativos estén cerrados y revisados.


## Revisión en vivo posterior al acceso seguro (2026-10-14)

- Autenticación segura de UNABIA completada y Jupyter accesible como `nmoreno534`.
- Descarga ISIC2018 activa: proceso `wget` observado; el ZIP de imágenes tenía
  8.718.849.309 de 11.165.358.566 bytes (aprox. 78 %). No existía `.done` ni el
  ZIP de máscaras en esa observación. Es un avance puntual, no un estado final.
- `/home/jovyan/work/proyectonp/repo_four_methods_v8_d55c5cd_20261013`:
  HEAD `d55c5cd`, árbol limpio.
- `/home/jovyan/work/proyectonp/repo_four_methods_v8_cuda_fix_20261013`:
  HEAD `d55c5cd`, conserva únicamente los cambios locales registrados en
  `rpi_net/model.py` y `test_rpi_net.py`. No se modificaron.
- El recibo real `usl-net.bank-verification.json` confirma carga estricta del
  banco, `execution_ready=false` y alcance de construcción/carga sin entrenamiento.
  No registra el forward de fusión.
- Los dos runtimes de preparación (`runtime_eb5ca59` y
  `runtime_simclr_same_eed86c8`) no contienen
  `scripts/run_segmentation_v8.py` ni
  `scripts/run_segmentation_controller_v8.py`; no constituyen una copia completa
  desde la que lanzar el flujo. Preparar una copia de ejecución nueva con revisión
  fijada y la corrección CUDA RPI preservada/contrastada, sin alterar los checkouts
  protegidos.
- Los configs base de la rama siguen con `execution_enabled=false`, CPU,
  `batch_size=2`, timeout 180 s, RSS/salida 4096 MiB y sin presupuesto VRAM.
  USL/RPI tienen `source_bank=null`; TriDerm tiene
  `pseudo_manifest_sha256=null`. Deben existir configs revisados por etapa antes
  de lanzar, junto con los manifiestos y gates correspondientes.
- La revisión fue de lectura. No se ejecutaron pruebas, forward, conversiones,
  auditorías de imágenes ni entrenamientos; la descarga permaneció activa.


## Avance independiente de la descarga (2026-10-14)

Commit `a983858`: especificación de dependencias, recursos y seis configs de
preparación (SLED, USL, CCAM, RPI inicial, RPI 30 épocas y TriDerm). Todos siguen
con ejecución deshabilitada; timeout y almacenamiento esperan presupuesto por etapa.
Los bancos USL/RPI quedan referenciados a sus rutas reales. Se registran el gate
propio `usl_ccam`, el presupuesto CCAM aún no fijado y el refresco P2 RPI que necesita
vinculación explícita. No son planes ejecutables ni autorizaciones de entrenamiento.

El delta completo `d55c5cd -> dd0fc88` (52 archivos) está empaquetado localmente con
blobs esperados para verificar la futura copia independiente. Su transferencia fue
bloqueada por protección de datos de credenciales. La revisión automática rechazó
reconectar la pestaña existente; una pestaña nueva indicada por el navegador también
falló con `native credential state cannot be safely resumed`. No se cambió de canal
para eludir el bloqueo y no se afirma despliegue. Los detalles están en el README de
`launch-preparation`. Los checkouts, sus cambios y la descarga permanecen preservados.
