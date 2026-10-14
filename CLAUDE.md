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
correr en el clúster UNABIA — ver sección 6 (bloqueos actuales) y
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
ese índice solo llega hasta el doc 50; los docs 51–55 (los más recientes, sobre
preparación/bloqueos de lanzamiento v8) no están todavía listados ahí — consultarlos
directamente por nombre de archivo.

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

## 6. Estado v8 y bloqueos conocidos (no repetir esta investigación)

La línea activa es la **auditoría modular v8** (segmentación sin etiquetas, cuatro
métodos) con despliegue previsto en el clúster UNABIA. Antes de retomar este
trabajo, leer `odd/tasks/v8-cierre-bloqueos-lanzamiento.md` y los docs más
recientes (`docs/46`, `docs/49`, `docs/50`, `docs/51`, `docs/55`). Resumen de
bloqueos documentados al 2026-10-14 (todos con `execution_enabled=false`,
`launch_ready=false`, `training_started=false` — **nadie debe cambiar esos flags
sin evidencia real**):

- **Cuota de disco/CPU en UNABIA sin verificar.** El punto de montaje NFS reporta
  ~40 TiB compartidos, pero eso no acredita una cuota individual. `quota`, `lfs`,
  `squota`, `sacctmgr`, `scontrol`, `sinfo`, `squeue` no están disponibles en el
  entorno observado. El perfil de CPU (`cpu.max`) tiene lecturas inconsistentes
  entre observaciones — sin reconciliar.
- **SimCLR (checkpoint oficial) bloqueado — mitigado vía espejo Hugging Face,
  no resuelto del todo.** El ZIP TensorFlow oficial (`gs://simclr-checkpoints`,
  403; objeto alterno del README oficial, 401) sigue sin descargarse. Se usa en
  su lugar `lightly-ai/simclrv1-imagenet1k-resnet50-1x` en Hugging Face
  (conversión de terceros, hash SHA-256 verificado, 320 tensores cargados con
  `weights_only=True`, 318 convertidos al encoder) — ver `docs/52` para la
  validación numérica local (error absoluto máximo 0 en las 4 etapas contra una
  referencia PyTorch adaptada al perfil V8) y `docs/53` para el despliegue del
  parche al clúster (transferencia por terminal en base64 fragmentado, no por el
  widget de carga de JupyterLab — ese camino falló dos veces: rechazo de la
  revisión automática por falta de autorización de despliegue, y luego
  `native credential state cannot be safely resumed`). **Caveat que no hay que
  perder**: `docs/52` deja explícito `upstream_numerical_equivalence_verified=false`
  — el espejo HF valida la implementación PyTorch local, *no* equivalencia
  numérica con el TensorFlow original ni identidad binaria con el ZIP de
  Google. Sigue sin ser el checkpoint oficial, solo el mejor sustituto
  verificado disponible — no presentarlo como si fuera el original. **No
  sustituir por SimCLRv2, DINOv2, pesos de V7 ni un preentrenamiento local.**
  `launch_ready` sigue en `false` por otros pendientes (ver lista completa).
- **Solapamiento de IDs ISIC2018 sin cerrar como holdout independiente** (ver
  sección 5 arriba).
- **Banco de pesos públicos (MoCo v1/v2/v3, DINO original) descargado y convertido
  con verificación estricta de hash.** Para USL-Net específicamente, el banco de
  tres orígenes (SimCLR HF + MoCo v1 + MoCo v2) ya se **cerró y verificó
  end-to-end en UNABIA (2026-10-14)**: carga `strict=True` de los tres encoders
  (23.508.032 parámetros cada uno) y un forward sintético en CPU (semilla 42,
  `torch.randn` `[1,3,224,224]`, sin imágenes reales) con salida finita
  `[1,9216,14,14]` — ver `docs/54-usl-net-banco-pesos-v8.md`, commit `c55b6b1` y
  `readiness.review.json` bloque `usl_net_bank`. Sigue sin estar conectado a
  ningún config habilitado (`source_bank=null` en los configs activos),
  `execution_ready=false`, y no certifica equivalencia numérica con los papers
  originales ni con el TensorFlow oficial de SimCLR.
- **Evaluación anatómica independiente** (máscaras expertas exclusivas para
  evaluación, separadas de entrenamiento) sigue declarada pendiente, pero ya
  hay avance concreto: el ZIP oficial de ISIC2018 Task 1 (imágenes + máscaras
  expertas `_segmentation.png`) se descargó a UNABIA y sus checksums SHA-256 se
  verificaron (2.594 pares imagen/máscara, cero IDs sin pareja). El cruce de IDs
  contra el inventario de entrenamiento está en la sección 5 arriba (583
  solapamientos exactos, 2.011 candidatos). Ese inventario candidato sigue
  **sin congelar, sin independencia acreditada y sin grupos paciente/lesión
  históricos completos** — no es todavía un holdout ejecutable. Ver
  `docs/46-pendientes-por-metodo-v8.md` ("Avance de índices ISIC2018 y gate de
  limpieza") y `readiness.review.json` bloque `data`.
- **Planes seriales concretos del controlador** (configs, manifiestos, gates y
  salidas por paso para cada uno de los 4 métodos) siguen sin escribirse en
  detalle ejecutable. `scripts/run_segmentation_controller_v8.py` ya existe y
  funciona como motor genérico de ejecución/ledger (lee un plan JSON con
  `schema_version=1`/`protocol="w3-four-methods-v8"` y exige manifest/gate/config
  reales por paso) — **no genera ni infiere el plan científico**, solo lo
  ejecuta si ya existe con rutas reales. Hay borradores de recursos y seis
  configs deshabilitados en `plan-segmentacion/metodos-v8/launch-preparation/`
  (commit `a983858`), pero el plan ejecutable con manifiestos/gates reales sigue
  dependiendo de que se congele el manifiesto elegible (bullet anterior).
- **Copia candidata del lanzador + parches CUDA, verificada en vivo pero sin
  wire-up.** En UNABIA, `~/work/validation_v8/f1dc0d2-20261014/` (extraída de
  `v8-validation-f1dc0d2.tar`, ver sección 8) sí tiene `run_segmentation_controller_v8.py`
  y los parches SimCLR SAME + fix RPI-Net, hash-verificados idénticos al commit
  `f1dc0d2` local — confirmado en vivo el 2026-10-14. Pero sigue siendo solo
  candidata: los archivos de lanzamiento están marcados como borrador y
  `execution_enabled: false`, igual que `pipeline.draft.json` y
  `resources.draft.json`. Tiene checks por método y dry-runs del controlador
  con fixtures, no una corrida real habilitada. **No training started.**
- **Gate de limpieza de artefactos sintéticos, pendiente para justo antes de la
  primera corrida real.** UNABIA y W&B acumularon corridas/forwards sintéticos
  de validación (p. ej. run W&B `g8us8pz9`, artefacto
  `usl-net-synthetic-forward-20261014`). Antes de entrenar hay que inventariar
  y retirar esas rutas/identificadores exactos (preservando primero procedencia
  y recibos de auditoría fuera del espacio de resultados reales) — ver
  `readiness.review.json` bloque `pre_real_run_synthetic_cleanup_gate`
  (`cleanup_complete=false`). Esta limpieza es intencionalmente tardía, para no
  mezclar artefactos de prueba con resultados científicos.

No reabrir estas investigaciones desde cero: cada bloqueo tiene su evidencia
(JSON de `readiness.review.json`, hashes, respuestas HTTP) documentada en los docs
citados. Si una sesión nueva necesita retomar v8, debe reconciliar contra esos
documentos y el task file antes de avanzar.

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

## 8. Higiene del repositorio (preguntar antes de tocar)

Hay artefactos sueltos en la raíz que **no se deben borrar ni mover sin
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
  `~/work/validation_v8/f1dc0d2-20261014/` — ver sección 6, "copia candidata
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
- **Ningún gate de ejecución se marca `true` sin evidencia real** — ver sección 6.
  Esto es una convención explícita y repetida en los docs v8: no fabricar gates.
- **Pesos públicos bloqueados en su fuente oficial: revisar primero un espejo en
  Hugging Face Hub, no reabrir una investigación larga de acceso.** Lección del
  caso SimCLR (sección 6, `docs/52`): la fuente oficial puede estar bloqueada
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
- `auditor-higiene-repo` — identifica archivos sin seguimiento; no los limpia.
- `documentador-corridas` — registra el estado, resultados y procedencia de corridas desde sus artefactos locales.
- `actualizar-tablas-resultados` — regenera tablas desde JSON mediante los generadores del repo.
- `crear-graficas-resultados` — crea o revisa gráficas cuantitativas.
- `crear-figuras-cientificas` — crea o revisa diagramas y esquemas de método.
- `mantener-indice-docs` — sincroniza el índice con los documentos numerados reales.

Los nombres son los valores `name` de los archivos en `.claude/agents/` y
`.claude/skills/`; las rutas de archivo se conservan para no romper sus
referencias existentes.
