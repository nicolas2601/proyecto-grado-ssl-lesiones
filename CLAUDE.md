# CLAUDE.md — Contexto del proyecto

Este archivo describe **este proyecto** (dominio, estado, convenciones). Las reglas
genéricas de orquestación (ODD/SDD, persona, gates de revisión, modelos por fase,
etc.) viven en `~/.claude/CLAUDE.md` del usuario y no se repiten aquí.

## 1. Qué es este proyecto

Proyecto de grado de Ingeniería de Sistemas (UNAB) sobre **clasificación de
lesiones cutáneas con aprendizaje autosupervisado (SSL)**. No es un dispositivo
de diagnóstico; es una herramienta de apoyo al tamizaje. Autores: Nicolás
Santiago Moreno Monroy y María Paula Saavedra; director Andrés Jerez; asesora
Karen Sánchez (KAUST). Fuente: `README.md`.

**Estado actual:** rama de trabajo `experiment/modular-audit-v8` (rama principal
`main`). El resultado científico de clasificación ya cerrado es **v7** (ver
`resultados/clasificacion/v7/SEMILLAS.md` y `docs/28-rediseno-brazo-propuesto-v7.md`).
El trabajo activo en v8 es una **auditoría modular de segmentación sin etiquetas**
(cuatro métodos: SLED, USL-Net, RPI-Net, TriDermNet-HAGO) y su preparación para
correr en el clúster UNABIA — ver la skill `estado-v8` (bloqueos actuales) y
`odd/tasks/v8-cierre-bloqueos-lanzamiento.md`.

Remoto: `https://github.com/nicolas2601/proyecto-grado-ssl-lesiones`. Licencia
de código: Apache-2.0 (`LICENSE`). Los pesos de modelo tienen licencia aparte
(`LICENSE-WEIGHTS`).

## 2. Stack técnico (confirmado por archivos del repo)

- **Lenguaje:** Python. CI corre la suite en 3.11 / 3.12 / 3.13 (`.github/workflows/ci.yml`).
  `cluster/README.md` documenta un perfil UNABIA de preentrenamiento v6 en
  **Python 3.11.x**; no confirma el runtime de otras corridas. Los
  `.pyc` cacheados localmente son `cpython-314` (entorno de desarrollo local en
  Python 3.14) — no confundir con el target de CI/cluster.
- **No hay `pyproject.toml` ni `setup.py`/`setup.cfg`** en el repo — no confirmado
  un empaquetado instalable formal; el proyecto se ejecuta vía `PYTHONPATH=src`
  (ver comando de reproducción en `README.md`) y requirements sueltos.
- **Dependencias** están repartidas en varios archivos `requirements*.txt` según
  propósito, no un único lockfile:
  - `requirements-dev.txt`: suite de tests completa (pytest, pytest-xdist,
    pytest-timeout, numpy, pandas, scikit-learn, scikit-image, ruff, opencv,
    imagehash, fastapi/httpx para el demo, safetensors/transformers/onnx para
    el release).
  - `requirements-pretrain.txt`: runs de preentrenamiento v6 GPU (torch>=2.2,
    torchvision>=0.17, mlflow, Pillow, opencv).
  - `cluster/requirements.txt`: perfil pineado para preentrenamiento v6
    (PyTorch 2.4.1 + torchvision 0.19.1, wheels cu121); no asumir que describe
    el runtime de otros pipelines UNABIA.
  - `demo/requirements.txt` y `demo/requirements-cpu.txt`: demo local.
  - No confirmado un requirements.txt único "de producción"; cada carpeta declara
    el suyo según para qué se usa.
- **Linter:** `ruff` (`ruff.toml` en la raíz). Configuración deliberadamente
  angosta: solo `E9, F63, F7, F82, F401` (errores reales, no estilo), para que el
  lint no se vuelva ruido ignorado. `line-length = 100`, `target-version = "py311"`.
  Excluye `notebooks, data, results, assets, demo`.
- **Testing:** `pytest` con `pytest-xdist` (`--dist loadfile`, porque correr torch +
  scikit-learn + scikit-image + scipy en el mismo proceso segfaltea) y
  `pytest-timeout`. 365 archivos `.py` bajo `tests/` (incluye `tests/segmentation_v8/`).
  Variable `CI_REQUIRE_FULL_DEPS=1` convierte los *skips* por dependencia faltante
  en fallos — así no se puede "pasar en verde" sin instalar todo. No confirmado un
  comando único tipo `make test`; el comando real está en `README.md`:
  `pytest -q` y `ruff check .`.
- **ML/Framework:** PyTorch + torchvision (preentrenamiento SSL propio sobre
  DINOv2 ViT-S/14 y ViT-B/14), `transformers`/`safetensors`/`onnx` para el release
  del modelo, MLflow (file-store) y Weights & Biases (offline en cluster) para
  tracking.
- **CI:** `.github/workflows/ci.yml`, trigger **solo manual** (`workflow_dispatch`)
  desde 2026-09-29 — Actions billing bloqueó las corridas automáticas; para
  reactivar hace falta restaurar los triggers `push`/`pull_request` (comentario en
  el propio workflow). Tres jobs: `lint` (ruff), `tests` (matriz 3.11/3.12/3.13,
  con torch CPU + wandb instalados aparte, timeout 25 min), `generated` (verifica
  que los notebooks generados y las tablas de resultados no diverjan de su fuente).

## 3. Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `src/dermssl/` | Librería del proyecto: preentrenamiento SSL (`pretrain/`), fine-tuning (`finetune/`), evaluación (`isic_eval/`, `final_eval.py`), métricas, bootstrap pareado, procedencia de corridas, `segmentation_methods/` (SLED, USL-Net, RPI-Net, TriDermNet-HAGO — línea v8), `bench/` (generadores de tablas), `release/` (export/conversión de modelos). |
| `scripts/` | Puntos de entrada: preentrenamiento, selección de variantes, evaluación final, generadores `build_*`, utilidades de auditoría v8. |
| `tests/` | Suite pytest (365 archivos), sin GPU; incluye `tests/segmentation_v8/`. |
| `resultados/` | Resultados versionados y vigentes: C1, v7, bancos congelados. Tablas generadas desde JSON (nunca a mano). |
| `results/` | Bancos de versiones anteriores (histórico; compatibilidad con runners viejos). |
| `docs/` | Documentación numerada 00–55 (ver sección 4) + `GUIA.md`, `plan-segmentacion-sin-etiquetas.md`, `figuras_analisis/`, `superpowers/plans/`. |
| `plan-segmentacion/` | Línea de segmentación sin etiquetas (CCAM, pseudo-máscaras, remoción de pelo). |
| `cluster/` | Operación del clúster UNABIA/v6: `setup_env.sh`, `prepare_datasets.sh`, `prepare_offline_assets.sh`, sbatch de preentrenamiento (`pretrain_vits_4gpu.sbatch`, `pretrain_vitb_4gpu.sbatch`), `finetune/` (banco de fine-tuning de baselines), `requirements.txt`, `README.md`. |
| `demo/` | Demo local (FastAPI backend + frontend estático) que sirve el reporte sellado de `resultados/` y, si se configura `MODEL_WEIGHTS_PATH`, corre inferencia real. |
| `assets/` | Figuras publicables (`figuras/`), catalogadas en `assets/README.md`. No borrar una figura sin actualizar el documento que la enlaza. |
| `data/` | Solo `RESULTS_SUMMARY.json` versionado; los datasets reales (HAM10000, ISIC, etc.) no viven en el repo. |
| `odd/tasks/` | Seguimiento ODD (Organic Driven Development) de trabajo en curso — ver sección 5. |
| `openspec/` | Artefactos SDD (OpenSpec): `openspec/config.yaml` (`schema: spec-driven`) y `openspec/changes/<nombre>/{proposal,design,tasks,specs}.md` — ver sección 5. |
| `.github/workflows/` | Único workflow `ci.yml` (disparo manual). |

## 4. Metodología: CRISP-ML(Q)

El proyecto migró de CRISP-DM a **CRISP-ML(Q)** (Studer et al. 2021) por
indicación de la asesora KAUST — ver `docs/03-metodologia-crisp-mlq.md` para el
mapeo completo de las 6 fases del proyecto y sus controles de calidad (Q). Resumen
de las fases y dónde viven:

1. Business & Data Understanding → `docs/01-contexto-y-objetivos.md`, `docs/02-datasets.md`.
2. Data Preparation → split lesión-disjunto, pesos de clase, pool SSL (`docs/02`, `docs/06-problematicas.md`).
3. Modeling → `docs/27-plan-v7-mejoras-ssl.md`, `docs/28-rediseno-brazo-propuesto-v7.md`.
4. Evaluation → KPIs clínicos, calibración, fiabilidad (`docs/07-fiabilidad-y-sesgo.md`, `docs/11-resultados-finales.md` — superado por v7).
5. Deployment → `demo/README.md`.
6. Monitoring & Maintenance → trabajo futuro (no implementado).

El índice vigente de toda la documentación numerada está en `docs/README.md`
(marca qué documento es "Vigente" vs "Superado" por una versión posterior). **Nota:**
la numeración llega al **doc 63** (2026-11-04) y el índice está incompleto en el
tramo alto — verificar con `ls docs/*.md` antes de afirmar que un documento no
existe, y consultar por nombre de archivo lo que el índice no liste.

**Instrumento de tono:** antes de tocar cualquier cosa que use ITA, FST o la
transformación tonal, leer `docs/63-instrumento-ita-alcance-y-limites.md`. Trae
qué mide el instrumento y qué no, el 28 % de abstención con sus tres
alternativas ya descartadas con evidencia, y una tabla de **cifras retiradas**
que no se deben reutilizar (el delta −0,047 del brazo B, las 1.917 coincidencias
de ISIC2018, "el ITA ordena", "la luz mueve más que la biología entera", entre
otras). Esa tabla existe para que una cifra muerta no llegue al informe final.

Punto de entrada recomendado por el propio índice: `docs/27` → `docs/28` → resultado
v7 en `resultados/clasificacion/v7/README.md`.

## 5. Datasets

| Dataset | Uso | Caveats |
|---|---|---|
| **HAM10000** | Núcleo: entrenamiento + evaluación supervisada, 7 clases (`akiec, bcc, bkl, df, mel, nv, vasc`). | Mismo `lesion_id` puede tener varias fotos → split debe ser por `lesion_id`, no por `image_id`, o hay *data leakage*. Desbalance fuerte (~58× entre `nv` y `df`) → métrica principal es macro-F1, no accuracy. |
| **ISIC 2019** | Pool SSL sin etiquetas para preentrenamiento domain-adaptive (~25k imágenes, 224×224). | Si trae `GroundTruth.csv` adjunto se usa para validación externa. |
| **ISIC 2018** (Task1/Task2 — segmentación) | Candidato para evaluación independiente en la línea v8 de segmentación. | Doc `46-pendientes-por-metodo-v8.md` registra un **gate de solapamiento de IDs sin cerrar**: el cruce corregido encontró 583 coincidencias entre 2.594 casos ISIC2018 y los 40.896 IDs del inventario de entrenamiento; quedan 2.011 casos candidatos sin congelar como holdout independiente. Una cifra previa de 1.917 coincidencias fue incorrecta (comparaba solo el sufijo numérico de 7 dígitos, con 1.334 falsos positivos) — **no reutilizar esa cifra**. |
| **BCN20000** | Candidato a validación externa cross-dataset (Objetivo 3). | Usado también en el EDA original. |
| **CO2Wounds-V2** | Contexto regional (heridas), solo EDA. | — |
| **PAD-UFES-20** | Prueba externa zero-shot registrada en `docs/30-pad-ufes-zero-shot.md`. | Registrada pero sin correr. |
| **Fitzpatrick17k** | Validación externa del estimador de fototipo (FST). | `docs/20c`. |

Reglas de higiene de datos (`docs/02`): split lesión-disjunto con `assert` de cero
solapamiento; pesos de clase = raíz del inverso de frecuencia; carga robusta ante
imágenes truncadas (placeholder, nunca tumbar el run).

## 6. Estado v8 (linea activa)

La linea activa es la **auditoria modular v8** (segmentacion sin etiquetas, cuatro
metodos) con despliegue previsto en el cluster UNABIA. Al 2026-10-14 todos los gates
siguen en `execution_enabled=false`, `launch_ready=false`, `training_started=false`.

**El detalle de los bloqueos abiertos, su evidencia y los caveats que no hay que
perder viven en la skill `estado-v8`.** Invocarla antes de retomar trabajo de v8, de
cambiar cualquier gate de ejecucion, o de afirmar que un bloqueo quedo resuelto. No
reabrir esas investigaciones desde cero: cada una tiene evidencia documentada
(`readiness.review.json`, hashes, respuestas HTTP) en `docs/46`, `docs/49`-`docs/55` y
`odd/tasks/v8-cierre-bloqueos-lanzamiento.md`.

## 7. Flujo de trabajo ya establecido (ODD / OpenSpec)

Este repo ya usa activamente los mecanismos genéricos de seguimiento del usuario:

- **`odd/tasks/*.md`** — tareas ODD en curso. Actualmente: `project-claude-config.md`
  (este trabajo de configurar `.claude/`) y `v8-cierre-bloqueos-lanzamiento.md`
  (punch list de cierre de v8, con pasos 1–7 numerados y contexto de autorización
  de acceso al clúster vía Playwright MCP).
- **`openspec/`** (`schema: spec-driven`, ver `openspec/config.yaml`) — artefactos
  SDD bajo `openspec/changes/<nombre>/`. Cambios existentes (algunos archivados,
  algunos activos — revisar `tasks.md` de cada uno para estado real):
  `c4-t1-t2-t4-figures`, `classification-diagnostic-figures`,
  `cuatro-metodos-segmentacion-v8`, `fst-domain-adaptation-circle`,
  `modular-audit-v8`, `s1-t5-qualitative-figures`, `seg-field-roi-eval-wandb`,
  `thesis-c1-c2-figure-system`, `w3-reproducibility-v8`.

No crear un tercer mecanismo de tracking: usar `odd/tasks/` para trabajo orgánico
y `openspec/changes/` cuando el trabajo entra en SDD explícito.

## 8. Archivos sin seguimiento (preguntar antes de tocar)

**Precedente de 2026-11-04, y el más caro que encontró este proyecto:** doce
módulos de `src/dermssl/` **nunca se commitearon** — no en ninguna rama, no
cubiertos por `.gitignore`, solo en el working tree. Entre ellos
`tone_augment_lab.py`, documentado como "congelado por contrato" y citado por
número de línea; `pretrain/paired_factor_gate.py`, el gate que acreditó las
corridas pareadas; y `segmentation_methods/common/hair.py`, que usa la receta
pineada del teacher 026. Un `git clean -fd` los borraba sin aviso. Rescatados en
`f14a63a`. También explica por qué `repo_v8_canonical` en UNABIA no los tenía:
se construyó con `git archive HEAD`, que solo empaqueta archivos rastreados.

**Regla derivada:** correr `git status --short src/ | grep '^??'` periódicamente,
y nunca dar por versionado un archivo porque un documento lo llame "congelado".
El agente `auditor-archivos-sin-seguimiento` existe para esto y no se había
corrido nunca sobre `src/`.

Hay además artefactos sueltos en la raíz que **no se deben borrar ni mover sin
preguntarle primero al usuario**, porque no está confirmado si son desechables:

- `jupyter-terminal-25.png` … `jupyter-terminal-46.png`. **Ya no es un
  problema** (2026-10-14): `.gitignore` línea 12 (`*.png`) ya las cubre,
  confirmado con `git check-ignore -v` — no aparecen como `??`. Se deja la
  entrada solo como ejemplo de este patrón de limpieza.
- `.playwright-mcp/` (directorio sin seguimiento, artefactos de las sesiones de
  automatización de navegador contra UNABIA). **Sigue sin regla de
  `.gitignore`** — a diferencia de los `.png`, este sí sigue apareciendo como
  `??` en `git status`.
- `v8-validation-f1dc0d2.tar` (archivo sin seguimiento). **Ya identificado**
  (2026-10-14): es el snapshot del commit `f1dc0d2` extraído en UNABIA en
  `~/work/validation_v8/f1dc0d2-20261014/` — ver la skill `estado-v8`, "copia candidata
  del lanzador". Sigue sin estar en `.gitignore` ni versionado; decidir si
  conviene ignorarlo explícitamente dado que ya cumplió su propósito de
  validación en el clúster.

El resto (capturas de Jupyter y `.playwright-mcp/`) seguía apareciendo como
`??` en `git status` al momento de escribir esta sección. No se tocaron.

## 9. Convenciones

- **Idioma de la documentación del proyecto: español, registro neutral/profesional**
  — es la convención ya establecida en todo `docs/*.md`, `INFORME.md` e
  `INFORME_ACADEMICO.md`. Toda documentación nueva de este proyecto (docs,
  READMEs, mensajes de estado) debe seguir esa convención. El código, nombres de
  variables/funciones/identificadores y comentarios de código siguen en inglés
  (práctica normal, y es lo que ya usa `src/`).
- **Perfil visual de figuras.** El perfil del Anteproyecto (tipografia, paleta
  cappuccino, paleta Fitzpatrick reservada, criterios de publicacion) vive en
  `.claude/rules/figuras.md`, que carga solo al trabajar sobre figuras. Regla de
  precedencia: si hay una referencia (HTML, imagen, mockup, URL), la referencia manda y
  el objetivo es fidelidad.
- **Commits: Conventional Commits con scope del hito**, por ejemplo
  `docs(v8): record ISIC2018 evaluation overlap`, `feat(v8): deploy verified
  SimCLR patch to cluster runtime via terminal transfer`, `chore(v8): prepare
  disabled execution configs...`. Confirmado leyendo `git log --oneline -20`:
  todos los commits recientes siguen `tipo(scope): descripción en inglés, minúscula,
  imperativo`. Mantener esa forma para nuevos commits de este repo (nótese que los
  *mensajes* de commit están en inglés aunque la documentación del proyecto esté
  en español — así se ha venido haciendo).
- **No copiar cifras a mano.** Las tablas de resultados se generan desde JSON
  (`resultados/*.json`, `results/*.json`) vía scripts `build_*`; un test falla si
  alguien edita un número a mano (`tests/test_readme_tables_fresh.py`,
  `tests/test_resultados_readme_fresh.py`, verificados en el job `generated` de CI).
- **Ningún gate de ejecución se marca `true` sin evidencia real** — ver la skill `estado-v8`.
  Esto es una convención explícita y repetida en los docs v8: no fabricar gates.
- **Pesos públicos bloqueados en su fuente oficial: revisar primero un espejo en
  Hugging Face Hub, no reabrir una investigación larga de acceso.** Lección del
  caso SimCLR (skill `estado-v8`, `docs/52`): la fuente oficial puede estar bloqueada
  por meses sin que eso sea un fallo del clúster; buscar un espejo de terceros
  en Hugging Face, verificar su hash SHA-256 contra el objeto publicado, y
  correr una validación numérica local antes de asumirlo usable — y declarar
  explícitamente si no se verificó equivalencia numérica con el original
  (`upstream_numerical_equivalence_verified`). Esto no es "ya quedó resuelto":
  es el sustituto mejor documentado disponible, nunca presentarlo como el
  checkpoint oficial.


## 10. Agentes y skills del proyecto

Usa el especialista cuyo alcance coincida con la tarea:

- `operador-unabia` — prepara, inspecciona y opera corridas en UNABIA con el procedimiento de `docs/RUN_CATALOG.md`.
- `investigador-literatura-ml` — busca métodos y literatura cuando una línea implementada no rinde lo esperado.
- `arquitecto-metodos-ml` — planifica la implementación de un método ya elegido; no lo selecciona ni lo programa.
- `revisor-pipeline-ml` — revisa cambios de código de ML antes de aceptarlos.
- `auditor-evidencia-pipeline` — verifica afirmaciones de resolución, preparación o cifras con evidencia.
- `auditor-archivos-sin-seguimiento` — identifica archivos sin seguimiento; no los limpia.
- `documentador-corridas` — registra el estado, resultados y procedencia de corridas desde sus artefactos locales.
- `actualizar-tablas-resultados` — regenera tablas desde JSON mediante los generadores del repo.
- `crear-graficas-resultados` — crea o revisa gráficas cuantitativas.
- `crear-figuras-cientificas` — crea o revisa diagramas y esquemas de método.
- `mantener-indice-docs` — sincroniza el índice con los documentos numerados reales.
- `estado-v8` — estado real y bloqueos abiertos de la auditoría v8, con su evidencia.

Los nombres visibles coinciden con el campo `name` y con el nombre base de
cada archivo en `.claude/agents/` y `.claude/skills/`, para que la invocación y
la búsqueda en el repositorio usen términos consistentes.

Además: `.claude/rules/figuras.md` carga automáticamente el perfil visual al trabajar
sobre figuras, y un hook `Stop` (`.claude/settings.json`) corre `ruff` y los tests de
frescura de tablas cuando la sesión tocó código, docs o resultados.

## 11. Orden obligatorio antes de medir o implementar

Fijado el 2026-11-04 tras una sesion que midio en la direccion equivocada medio
dia. Las cuatro fallas fueron de **orden de lectura**, no de capacidad: la regla
que las evitaba ya estaba escrita y no se leyo.

**Antes de medir cualquier cosa nueva, o de implementar, en este orden:**

1. **Memoria local** — `~/.claude/projects/C--Users-mpaul-Documents-ProyectoDeGrado/memory/`.
   Leer los archivos, no solo `MEMORY.md`: **el indice esta incompleto** y los
   enlaces `[[nombre]]` dentro de cada archivo apuntan a cosas que el indice no
   lista. Asi se perdio el puntero a v9 (`[[project_v9-bundles-inferencia]]`,
   enlazado desde `feedback_engram-vps-siempre.md`, ausente del indice).
2. **`docs/` del repo** — en particular `docs/63` antes de tocar ITA/FST/tono, y
   su **tabla de cifras retiradas**.
3. **Engram VPS (`engram-equipo`)** — ahi sube Nicolas. **Barrer por ID con
   `mem_get_observation`, NO con `mem_search` filtrado**: ese servidor devuelve
   "no memories found" con facilidad y muchas observaciones del proyecto estan
   ruteadas bajo el proyecto `claude-web`, no bajo
   `proyecto-grado-ssl-lesiones`. Revisar tambien los proyectos `nicolas`,
   `proyecto-de-grado`, `anteproyecto-grado` y `unab-grado`, no solo el propio.
4. **Declarar contra que corrida se mide**, por nombre y fecha, antes de la
   primera medicion. No asumir que el repo local o GitHub estan al dia: **no lo
   estan**. Nicolas trabaja en maquinas y repos que no se ven desde aqui.

**Si falta un dato: preguntar.** No deducirlo del codigo ni rellenarlo con una
hipotesis plausible. Deducir el diseno de un experimento ajeno costo medio dia.

### Checkpoints: al terminar cada paso, no al final

La skill `checkpoint` exige commit atomico + comando que pasa + linea en
`docs/progress/<feature>.md`, **escrito al terminar cada paso**. Agrupar
dieciseis pasos en una sola escritura al final no es un checkpoint: si la sesion
se corta, se pierde el mapa de lo hecho. Escribir la linea cuando el paso cierra.

### Validar, no rehacer

El trabajo que otra sesion ya entrego se **valida** con sus propias cifras y una
corrida dirigida. Reconstruirlo duplica el costo y pisa a quien lo estaba
haciendo. Antes de asumir que una sesion esta inactiva, mirar su estado real:
`started 4d ago` es cuando arranco, **no** su ultima actividad.

### Contrato de respuesta: la recomendacion va adelante, no al final

Fijado el 2026-11-04 despues de que Maria Paula lo pidiera **cuatro veces** en una
sola sesion. Es la falla mas repetida del proyecto y la mas cara en confianza:
no es un error tecnico, es entregar el trabajo a medias y dejarle la decision a
ella cuando la decision era mia.

**Toda respuesta que presente un hallazgo lleva la recomendacion explicita,
derivada de lo observado, antes de los detalles.** No "hay dos caminos, deci
vos". Si los datos alcanzan para recomendar, se recomienda. Si no alcanzan, se
dice **que dato falta y como se consigue**, y se va a buscar.

**Prohibido partir un paso en varias preguntas.** Si el plan tiene cinco pasos y
ella dijo "hagamoslo", se corren los cinco y se reporta al final. Se para solo
ante algo **irreversible o fuera de lo pedido**. Verificar, calcular, leer
codigo y escribir artefactos nuevos al lado de los originales **no son
irreversibles**: no se piden permiso.

**Una pregunta por respuesta, como maximo, y solo si bloquea de verdad.** Si se
puede avanzar en otras cuatro cosas mientras, se avanza y se pregunta al final.

**Cuando una recomendacion propia se cae, decirlo de frente y con la evidencia
que la tumbo**, en la misma respuesta, no en la siguiente. Ya paso dos veces:
"re-correr el control sobre `source_fit`" quedo refutada por la lista blanca del
envoltorio (`clinical_batch_quota not in (1, 4)` lo rechaza por codigo), y
"re-materializar el espejo" quedo refutada por la modalidad de `pad_adapt`.

**Nunca recomendar rendirse.** Paso el 2026-11-04 y la respuesta de ella es la
regla: no se baja la cabeza ante un resultado negativo **y menos sin haber
validado la fuente**. Un brazo que nunca se corrio no es un brazo que fallo.
Antes de dar algo por perdido, verificar que lo que fallo comparte propiedad con
lo que se propone.

### Que el repositorio no es la fuente de verdad

Confirmado el 2026-11-04 por diferencia de modulos, no por suposicion. El repo y
el codigo congelado del cluster **divergieron en las dos direcciones**:

| | modulos |
|---|---|
| el congelado tiene, el repo **no** | `coverage_sampling.py`, `sampling_contract.py`, `image_input.py`, `torch_monitor.py` |
| el repo tiene, el congelado **no** | `paired_factor_gate.py`, `tone_augment_pairs.py`, `torch_view_audit.py` |

Por eso buscar `clinical_batch_quota` en `src/` da **cero** y aun asi el control
corrio con cuota 4: la cuota vive en modulos que el repo nunca tuvo. **Un grep
local negativo no prueba que una funcionalidad no exista.** Se confirma en el
cluster antes de afirmarlo.

Las corridas reales se lanzan con envoltorios del cluster
(`overlay_ssl_runtime.py` y sus hermanos en
`~/work/clf1_clinical_reference_20261024/scripts/`) que **no estan versionados**
y que imponen listas blancas de configuracion propias. Leer el envoltorio antes
de disenar cualquier corrida pareada.

### Leer el campo, no su nombre

`launch-seal.json` guarda `wrapper_sha256`, y **no es el hash del envoltorio**:
la linea 99 hace `seal = config_seal(config)`, un sello de la configuracion. El
archivo en disco hashea distinto y eso es correcto. Se estuvo a punto de
reportar una procedencia rota por confiar en el nombre del campo.

Igual que `ita_delta_applied_abs_mean` en el manifiesto del espejo, que es una
media **con signo** pese a decir `abs`. Antes de citar un campo, confirmar que
lo que calcula es lo que su nombre promete.

### Verificar el insumo antes de lanzar N

Correr sobre 5 filas y confirmar que los campos son por-item antes de lanzar
sobre miles. Un campo que parecia por-imagen (`source_revision`) era una
constante del dataset y obligo a repetir una corrida de 1.500 imagenes.
