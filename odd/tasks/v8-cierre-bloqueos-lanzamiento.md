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

## Avance vigente del paso 1 (2026-10-15 UTC)

### Bloqueo de integridad de fuentes (2026-10-16 UTC)

El aplicador candidato se transfirió por ZIP a UNABIA (SHA-256 `1f2cfdeb33870f7af65cd146e717115aee51cba79ffdf902c008b38ca667b10c`) y pasó tres pruebas de contrato allí. El intento real de materializar las 1.917 exclusiones abortó sin salida: las dos copias en UNABIA de `isic2018_exclusion_review_register_20261015.json` tienen 9.271.726 bytes, pero no coinciden con el pin `4b967833559bba375c59b57686b19278301bd6767926fdd3fac215fddd11fecf`. El ZIP v2 local y remoto tiene SHA-256 `87d32a480c6f2b8a9a50b768f070f53ca954c5ae1cd3d26fca39d7852bcd1e42`, distinto del pin de readiness `87d32a480c6f2b8a9a50b768f070f53ca9545cae1cd3d26fca39d7852bcd1e42`; contiene solo `scripts/build_v8_exclusion_review.py` (12.641 bytes), no el registro JSON. Los hashes observados del informe pHash y la cola nearest también devuelven `pin_match=false`. No se inició adjudicación visual ni se congelaron manifiestos. No se ha demostrado la causa de las discrepancias; no se modifican pins para ajustarlos.

- [ ] Recuperar o reconstruir registro, informe pHash y cola nearest con procedencia y hashes verificables; conservar un recibo comprobable de la reconciliación.
- [ ] Aplicar y verificar las 1.917 exclusiones solo después de reconciliar las fuentes; revisar después todos los pares pHash, no solo la cola nearest.
- [ ] Antes de aplicar exclusiones a datos reales, incorporar al aplicador el hash del código/helper en el recibo y publicación atómica o invalidación explícita de directorios sin recibo final. Una revisión independiente informada en esta sesión no reportó bloqueo de filtrado ni fuga de datos, pero su dictamen no está persistido como informe auditable en el repositorio.

`base_id_exclusions_materialized=false`, `candidate_adjudication_complete=false`, `manifest_frozen=false`, `execution_enabled=false`, `launch_ready=false`, `training_started=false`; independencia por paciente no acreditada. No hubo entrenamiento. La limpieza sintética permanece justo antes de la primera corrida real.

### Decisión adoptada y entrega candidata

El usuario aprobó `retrospective_public_benchmark`, manteniendo `patient_independence_accredited=false`. Sustituye las declaraciones históricas posteriores sobre reformulación pendiente de aprobación. El alcance es evaluación retrospectiva pública de 2.594 imágenes/máscaras Task 1; no acredita pacientes nuevos ni test ciego original.

- [x] Registrar aceptación del alcance limitado y conservar la limitación clínica.
- [~] Preparar contrato/evaluador candidatos: protocolo común en config/gate/manifest/export; recibos hash de exclusiones, adjudicación, máscaras, exposición y aislamiento; IDs exactos, `>=0.5`, bootstrap por imagen y fallos en denominador original.
- [x] Revisión estática y suite nueva de contratos en copia aislada de UNABIA: 15 passed, 2 warnings. No se ejecutaron localmente; no equivale a validación científica ni a habilitación de la corrida.
- [ ] Materializar/verificar 1.917 exclusiones por ID base; adjudicar manualmente 15.702 pares y vínculos fiables. Nearest cubre 1.235/4.195 rutas; no habilita exclusión automática.
- [ ] Validar contenido/geometría de 2.594 pares, completar exposición y aislar máscaras; congelar protocolo/manifiestos/hashes reales.
- [ ] Preparar pseudo-máscaras, planes seriales y presupuestos medidos. No copiar épocas/recursos V7.
- [ ] Limpiar pruebas sintéticas justo antes de la corrida real, conservando recibos.

Se recuperan semilla/config/revisión fijadas, artefactos inmutables, hashes/procedencia y selección de checkpoint independiente de máscaras finales. W&B registra entrega solo con acuse; offline queda pendiente. `execution_enabled=false`, `launch_ready=false`, `training_started=false`.

El candidato incorpora la corrección de revisión sobre linaje de checkpoints: protocolo, manifiesto/gate de entrenamiento y cohorte deben coincidir en pesos, sidecar, configuración y recibo del export. Un checkpoint histórico sin linaje embebido no se admite cambiando el sidecar. Los contratos de mismatch/sustitución se validaron en la suite aislada de UNABIA.

La revisión posterior exige ese recibo también al refinar, ligado a manifiesto/config/gate/export; el evaluador registra revisión del snapshot y hashes propios/de métricas. Los contratos para export histórico, tipo incorrecto de checkpoint, recibo sustituido y procedencia del código se probaron en UNABIA. La evaluación científica y los cierres materiales continúan pendientes.

### Validación aislada de contratos en UNABIA (2026-10-15)

Ruta: `/home/jovyan/work/validation_v8/retrospective-public-benchmark-20261015/`. Paquete ZIP r4: SHA-256 `dd87871b023897160d389e0037216bd74fe2cb85b2945c792aa502b31e78a6de`, 20 rutas seguras. Se corrigió una omisión del bundle: la primera copia carecía del módulo de métricas compartido y sus cuatro dependencias, con resultado 13 passed/2 failed. R4 incorporó esos archivos; la copia base `f1dc0d2-20261014` quedó intacta.

Comando: `PYTHONPATH=src ./.validation_deps/bin/pytest -q tests/segmentation_v8/test_retrospective_protocol.py`. Resultado: **15 passed, 2 warnings en 2,07 s**. Las advertencias corresponden a opciones heredadas `timeout`/`timeout_method` desconocidas por pytest en ese entorno; la suite no acredita su activación. Se validaron contratos con fixtures, sin entrenamiento, evaluación científica, W&B, publicación de métricas ni limpieza sintética.

Independencia por paciente `false/no acreditada`; manifiesto sin congelar. Siguen pendientes exclusiones, adjudicación, contenido/geometría, exposición, aislamiento, pseudo-máscaras, planes seriales y presupuestos; limpieza sintética justo antes de la primera corrida real. Esta actualización modifica solo documentación: no cambia `readiness` ni gates, que siguen `execution_enabled=false`, `launch_ready=false`, `training_started=false`.

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

**Los ceros no acreditan independencia clínica:** falta `patient_id` en HAM10000 y no hay equivalencia de identidades entre sistemas. Tampoco adjudican los candidatos positivos del cruce anterior. El paso 1 sigue abierto, `manifest_frozen=false`, las exclusiones siguen sin aplicar/verificar y la limpieza sintética queda pendiente al final.

La auditoría retrospectiva local del 2026-10-15 distingue dos protocolos: clasificación V7 agrupó HAM10000 por `lesion_id` mediante `StratifiedGroupKFold`, con selección interna y exclusiones del pool SSL; sus JSON finales s42/s43/s44 comparten `n_eval=2004` y `eval_ids_sha256=6ff4c17f04a9eae2f0d058aaa41084ea65b9aff91e4d5e5b05c924b4465a2fc6`. Camino B asignó pseudo-validación por hash de `dataset:image_id` con semilla 42: su cache tiene 58.457 imágenes y 1.169 de validación. El registro S4-A2 real es `plan-segmentacion/camino-b-cnn/resultados/2026-09-28-s4-a2/run.json`; declara entrenamiento completo y régimen transductivo sin etiquetas. El análisis 722 dentro/1.872 fuera del pool procede separadamente de `2026-09-21-final/evaluacion_isic2018/summary.json`. Ninguno certifica separación clínica entre fuentes. Rutas, claves y limitaciones quedan en [la auditoría de docs/46](../../docs/46-pendientes-por-metodo-v8.md#auditoría-retrospectiva-de-splits-y-utilidad-de-ham10000-2026-10-15) y `data.step_1_freeze_gate_review.retrospective_split_audit_20261015` del recibo.

Los metadatos HAM sí sirven para agrupar fotografías de una lesión, excluir IDs/grupos conocidos y trazar procedencia. Sus 10.015 filas oficiales tienen lesión y carecen de paciente; no bastan para acreditar independencia paciente/lesión de Task 1 frente a todas las exposiciones de entrenamiento. Falta un enlace de paciente comparable, completo y trazable en ambos lados. La cobertura parcial por fuentes y los ceros de cadenas exactas no lo reemplazan. Se mantiene el bloqueo o se requiere una revisión explícita del alcance científico; no se adoptó esa revisión ni se habilitaron gates. Esta actualización es documental: no constituye una corrida nueva ni repite búsquedas generales. El recibo actualizado revoca el relato antiguo de `AGENTS.md` §5: los 1.334 `_downsampled` son derivados confirmados por ID base, no falsos positivos; rigen las 1.917 exclusiones pendientes y los 677 IDs sin enlace, con los 2.011 candidatos conservados solo como historial.

### Contraste de protocolo con fuentes primarias (2026-10-15)

Se corrigió una confusión de tareas: Task 1 tiene 2.594 imágenes de entrenamiento y 2.594 máscaras de lesión; las 12.970 máscaras, cinco atributos por imagen, corresponden a Task 2. Sus etiquetas de entrenamiento eran públicas; los 100 casos de validación y los 1.000 de test del desafío original tenían máscaras ocultas. La cohorte de 2.594 usada aquí es un benchmark público previamente consultado, no el test ciego del desafío. Fuentes: [artículo ISIC2018, §§2.1–2.2](https://arxiv.org/html/1902.03368v1), [aclaración del organizador](https://forum.isic-archive.com/t/number-of-images-and-ground-truths/335) y [Task 1 oficial](https://challenge.isic-archive.com/landing/2018/45/).

La evidencia distingue separación de imágenes, lesiones y pacientes de validación externa por centro/dataset. HAM10000 ayuda a agrupar fotos mediante `lesion_id`, pero no tiene `patient_id`; la lesión adicional previamente hallada pertenece a los 1.917 solapamientos por ID base ya propuestos para exclusión. Los sidecars que enlazan 189 de los 677 IDs restantes no aportan ningún paciente o lesión no vacío. No corresponde repetir la descarga ni atribuir independencia clínica a esos ceros. [CLAIM, ítem 20](https://rsna.github.io/claim/item-20.html) recomienda separar por paciente o nivel superior y justificar niveles inferiores. El hallazgo institucional externo de Task 3 en el artículo del desafío no completa los pacientes de Task 1.

La segmentación transductiva sin máscaras puede ser válida para la cohorte objetivo disponible; exige declarar exposición a sus imágenes y no equivale a generalización inductiva en pacientes nuevos ([Kamnitsas et al.](https://arxiv.org/html/2107.08964v2)). Clasificación V7 agrupó por lesión y Camino B usó hash por imagen y régimen transductivo; no se invalida globalmente ese historial ni se infiere consulta de máscaras adicional. El detalle y las limitaciones están en [docs/46](../../docs/46-pendientes-por-metodo-v8.md#contraste-con-literatura-y-alcance-defendible-de-la-evaluación-2026-10-15).

La recomendación pendiente es fijar protocolo/configuraciones antes de puntuar, reservar las 2.594 imágenes, excluir y verificar las 1.917 exposiciones por ID base y derivados/casi duplicados confirmados, aplicar grupos conocidos donde sean fiables y cerrar la adjudicación. Con ese alcance se informaría evaluación retrospectiva de benchmark público e independencia por paciente sin acreditar. Afirmar independencia clínica por paciente/centro necesita cohorte nueva o enlace completo defendible y test ciego adecuado sin consulta previa. Esta recomendación no está adoptada: no se cambian predicados ni flags, no se cierran bloqueos y la limpieza sintética sigue al final.

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

- [x] **Tarea 5 — medir cuotas reales del clúster.** Confirmado en vivo por terminal
  (2026-10-14): 48 CPUs, 123 GiB RAM (115 GiB libres), 2× RTX PRO 4000 Blackwell
  24 GB cada una (0% uso), cgroup CPU/memoria en `max`/`max 100000` (sin límite real
  impuesto por la plataforma), disco compartido 40T (1% usado). Coincide exactamente
  con lo ya registrado en `resultados/preparacion_v8/readiness_20261014/readiness.review.json`
  (bloque `resources`) — no hay deriva, el registro sigue vigente. Pendiente real: NO
  es volver a medir, es traducir estos números en límites concretos de timeout/RSS/VRAM
  por etapa para el plan del controlador (tarea 6).

- [~] **Tarea 1 — congelar manifiesto elegible.** Groundwork completo, construcción
  bloqueada por falta de datos (no de código). Ver memoria Engram
  `proyecto-grado-ssl-lesiones/v8-manifiesto-elegible` para el detalle completo:
  - Ubicación real de datasets confirmada: `/home/jovyan/work/datos/{ham10000,isic2019_fullres,isic2020}/`
    con sus CSV de metadata y columnas exactas.
  - Contrato exacto de `src/dermssl/segmentation_methods/common/data.py` leído byte
    a byte (ALLOWED/FORBIDDEN, eligibility block, recomputo de SHA-256 en cada carga).
  - **Bloqueo raíz real encontrado**: la "evaluación anatómica independiente" que
    exige docs/39:82 necesita máscaras expertas de referencia (tipo ISIC2017/2018
    Lesion Boundary Segmentation Challenge). No existe ningún dataset de máscaras
    expertas descargado en el clúster — solo hay máscaras generadas por U-Net para
    QA de recorte (`mask_manifest.json`, no son expertas). Esto es un problema de
    adquisición de datos externos, no de escritura de script.
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
    son varias horas, no ~30 minutos.** Pendiente: confirmar `.done`, verificar que
    `ISIC2018_Task1_Training_GroundTruth.zip` tenga 2594 máscaras
    `ISIC_<id>_segmentation.png`, y recién ahí diseñar cómo esta reserva se integra al
    manifiesto elegible (como partición `development` separada y fuera del pool `train`,
    per docs/39:82).

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
