# Flujo completo de experimento Kaggle Implementation Plan

> **Estado al 2026-09-18:** solo las Tasks 1 y 2 están implementadas. Las Tasks 3-7 son plan pendiente: los archivos que mencionan (por ejemplo `stage1_bundle.py`, `kaggle_stage1_features.py`, `kaggle_complete_push.py` y los tests `test_stage1_*` y `test_kaggle_complete_*`) todavía no existen en el repositorio.

> **Para agentes de implementación:** SUB-SKILL OBLIGATORIO: usar `subagent-driven-development` o `executing-plans` y ejecutar cada tarea con sus pasos RED/GREEN. Los pasos usan casillas (`- [ ]`) para seguimiento.

**Objetivo:** preparar un único kernel Kaggle reproducible que produzca evidencia JSON desde el manifiesto FST hasta CIRCLe condicionado, sin incluir UNAB ni afirmar resultados que no hayan corrido en Kaggle.

**Arquitectura:** se separará el contrato puro de artefactos del código GPU. Un generador construirá un notebook autocontenido: comprimirá las fuentes Python necesarias, las extraerá bajo `/kaggle/working/ssl_flow`, añadirá esa raíz a `sys.path` y ejecutará un orquestador por etapas. Cada etapa escribirá un JSON con `run_id`, huellas de insumos y estado; Stage 2 sólo se invoca cuando la puerta G3 de Stage 1 sea `FAIL`.

**Tecnologías:** Python 3.11, argparse, JSON/NPZ, NumPy, Pillow, PyTorch/Torchvision, scikit-learn, Kaggle CLI y pytest.

**Especificación:** `openspec/changes/fst-domain-adaptation-circle/{proposal.md,design.md,specs/adaptacion-dominio-ssl/spec.md,specs/evaluacion-equidad-tono/spec.md,specs/mitigacion-circle/spec.md,specs/evidencia-fst/spec.md}`.

## Restricciones globales

- No tocar ni revertir cambios ya presentes en el checkout; limitar cada cambio a los archivos enumerados en esta planificación.
- UNAB queda explícitamente fuera: los únicos datos del kernel serán los handles Kaggle declarados en `kernel-metadata.json`.
- Todo texto visible de `--help`, mensajes de consola y errores de los CLI del flujo estará en español profesional; flags, claves JSON, identificadores Python y contratos existentes permanecen estables.
- El lanzamiento remoto siempre usa `--via kaggle`; no se permite selección automática ni Composio en el nuevo ejecutable integral.
- `SMOKE=True` o `--smoke` produce `valid_for_thesis=false`; un smoke nunca habilita Stage 2 ni resultados de tesis.
- No hacer commit, push ni PR como parte de esta implementación.
- El preentrenamiento SSL no recibe etiquetas diagnósticas ni FST; las etiquetas aparecen únicamente en extracción/evaluación de Stage 1 y en los probes de Stage 2.
- No declarar una mejora: la promoción exige reducción de brecha y caídas globales de macro-F1 y AUC de como máximo 0,02 frente a la base comparable.

## Estructura de archivos objetivo

| Archivo | Responsabilidad |
|---|---|
| `src/dermssl/cli.py` | Creador de parsers con ayuda de argparse completamente localizada. |
| `src/dermssl/experiment_contract.py` | Esquema puro, huellas y validación del ledger de etapas. |
| `src/dermssl/stage1_bundle.py` | Contrato puro del NPZ de features y de la salida Stage 1. |
| `scripts/kaggle_stage1_features.py` | Extracción GPU de features, probe Stage 1, G3 y escritura de bundle/JSON. |
| `scripts/kaggle_complete_experiment.py` | Orquestador que encadena FST, transformaciones, SSL, Stage 1 y CIRCLe. |
| `scripts/build_kaggle_complete_notebook.py` | Generador del notebook autocontenido y del archivo fuente comprimido. |
| `scripts/kaggle_complete_push.py` | Staging y publicación del kernel integral exclusivamente mediante Kaggle CLI. |
| `notebooks/kaggle_complete_experiment.ipynb` | Artefacto generado que Kaggle ejecuta; no editar manualmente. |
| `scripts/kaggle_run_bank.py` | Conserva bancos heredados y corrige frescura con token y descarga aislada. |
| `scripts/kaggle_push.py` y `scripts/notebook_code_v5.py` | Inyectan y persisten el token de una ejecución de banco heredada. |
| `docs/22-flujo-completo-kaggle.md` | Runbook operativo, artefactos esperados y límites local/Kaggle. |

---

### Tarea 1: Fijar contratos puros y localización compartida

**Archivos:**
- Crear: `src/dermssl/cli.py`
- Crear: `src/dermssl/experiment_contract.py`
- Crear: `tests/test_experiment_contract.py`
- Modificar: `tests/test_pipeline_localization.py`
- Modificar: `scripts/kaggle_fst_manifest.py`
- Modificar: `scripts/generate_tone_transforms.py`
- Modificar: `scripts/pretrain.py`
- Modificar: `scripts/kaggle_circle_stage2.py`
- Modificar: `scripts/kaggle_run_bank.py`

**Interfaces:**
- Produce `build_spanish_parser(*, description: str) -> argparse.ArgumentParser`.
- Produce `new_ledger(run_id: str, code_sha256: str) -> dict`, `record_stage(ledger: dict, *, name: str, status: str, artifacts: dict[str, Path | str]) -> dict` y `validate_ledger(ledger: dict) -> None`.
- Cada etapa válida tendrá `schema_version == "1.0"`, `run_id`, `input_sha256`, `output_sha256`, `status` y `valid_for_thesis`.

- [ ] **Paso 1: Escribir los tests RED de parser y ledger.**

```python
def test_spanish_parser_replaces_argparse_default_help():
    parser = build_spanish_parser(description="Flujo de prueba")
    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["--help"])
    assert "opciones" in capsys.readouterr().out
    assert "muestra esta ayuda y termina" in capsys.readouterr().out

def test_ledger_rejects_stage2_without_failed_stage1(tmp_path):
    ledger = new_ledger("run-1", "a" * 64)
    record_stage(ledger, name="stage1", status="PASS", artifacts={})
    record_stage(ledger, name="stage2", status="COMPLETE", artifacts={})
    with pytest.raises(ValueError, match="Stage 2"):
        validate_ledger(ledger)
```

- [ ] **Paso 2: Ejecutar RED.**

Ejecutar: `pytest tests/test_experiment_contract.py tests/test_pipeline_localization.py -q`

Esperado: falla porque no existen `dermssl.cli` ni `dermssl.experiment_contract`, y las ayudas aún contienen los valores predeterminados de argparse.

- [ ] **Paso 3: Implementar el mínimo contrato y parser.**

```python
def build_spanish_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument("-h", "--help", action="help", help="muestra esta ayuda y termina")
    parser._optionals.title = "opciones"
    return parser

def validate_ledger(ledger: dict) -> None:
    stages = {stage["name"]: stage for stage in ledger["stages"]}
    if "stage2" in stages and stages["stage1"]["status"] != "FAIL":
        raise ValueError("Stage 2 solo puede ejecutarse tras Stage 1 con estado FAIL")
```

Usar `build_spanish_parser` en los cinco CLI existentes de esta tarea. Traducir también el `help` faltante de cada flag y cualquier mensaje de validación visible, sin renombrar flags ni JSON.

- [ ] **Paso 4: Ejecutar GREEN.**

Ejecutar: `pytest tests/test_experiment_contract.py tests/test_pipeline_localization.py -q`

Esperado: pasa; `python scripts/<cli>.py --help` presenta encabezados y ayudas en español para los cinco CLI.

### Tarea 2: Corregir la frescura de los resultados de bancos Kaggle

**Archivos:**
- Modificar: `scripts/kaggle_push.py`
- Modificar: `scripts/kaggle_run_bank.py`
- Modificar: `scripts/notebook_code_v5.py`
- Modificar: `scripts/build_notebook_v5.py`
- Modificar: `tests/test_kaggle_push.py`
- Modificar: `tests/test_kaggle_run_bank.py`
- Modificar: `tests/test_notebook_build.py`

**Interfaces:**
- `stage(..., run_token: str = "") -> Path` escribe exactamente un `KAGGLE_RUN_TOKEN` en el notebook staged.
- `verify_result(path: Path, *, real: bool, expected_run_token: str) -> dict` exige el modo y el token exactos.
- El estado pasa de `model -> "running"` a `model -> {"state", "run_token", "mode"}`; `load_state` acepta el formato heredado y lo normaliza.

- [ ] **Paso 1: Escribir RED para JSON real obsoleto y descarga aislada.**

```python
def test_real_json_with_another_run_token_is_stale(tmp_path):
    path = tmp_path / "results_alexnet.json"
    path.write_text(json.dumps({"smoke": False, "kaggle_run_token": "old"}))
    with pytest.raises(krb.StaleResult, match="token"):
        krb.verify_result(path, real=True, expected_run_token="new")

def test_collect_never_accepts_a_json_left_in_the_final_directory(tmp_path, monkeypatch):
    final = tmp_path / "results"
    final.mkdir()
    (final / "results_alexnet.json").write_text('{"smoke": false, "kaggle_run_token": "old"}')
    monkeypatch.setattr(krb, "download_with_retry", lambda *args, **kwargs: True)
    assert krb.collect("baselines", "alexnet", "user", True, final, "new") == "failed"
```

- [ ] **Paso 2: Ejecutar RED.**

Ejecutar: `pytest tests/test_kaggle_run_bank.py tests/test_kaggle_push.py tests/test_notebook_build.py -q`

Esperado: falla porque el resultado actual sólo compara `smoke` y descarga directamente en `results/<bank>/`.

- [ ] **Paso 3: Implementar token, staging aislado y publicación explícita.**

1. Añadir `KAGGLE_RUN_TOKEN = ""` a la celda de configuración y `"kaggle_run_token": KAGGLE_RUN_TOKEN` a `RESULTS_SUMMARY_v5.json`.
2. Hacer que `stage` reemplace una sola línea `KAGGLE_RUN_TOKEN = ...`, igual que `SMOKE`.
3. Generar el token antes de cada push con `uuid.uuid4().hex`, persistirlo antes de consultar estado y exigirlo en `verify_result`.
4. Descargar en `TemporaryDirectory(dir=out_dir / ".downloads")`; verificar allí el único JSON esperado y copiarlo atómicamente a `out_dir` sólo después de validar modo y token.
5. Añadir `--via` obligatorio con `choices=("kaggle",)` a `kaggle_run_bank.py`; rechazar la omisión para que las ejecuciones sean literalmente `--via kaggle`.

```python
with tempfile.TemporaryDirectory(dir=out_dir / ".downloads") as raw:
    download_dir = Path(raw)
    if not download_with_retry(slug, download_dir):
        return "failed"
    data = verify_result(download_dir / expected_name, real=real,
                         expected_run_token=run_token)
    shutil.copy2(download_dir / expected_name, out_dir / expected_name)
    return "done"
```

- [ ] **Paso 4: Ejecutar GREEN y comprobar el notebook generado.**

Ejecutar: `pytest tests/test_kaggle_run_bank.py tests/test_kaggle_push.py tests/test_notebook_build.py -q`

Esperado: pasa; cubre JSON smoke obsoleto, JSON real con token anterior, JSON con token correcto, estado reanudable y descarga transitoria.

### Tarea 3: Definir el bundle de features y la puerta Stage 1

**Archivos:**
- Crear: `src/dermssl/stage1_bundle.py`
- Crear: `scripts/kaggle_stage1_features.py`
- Crear: `tests/test_stage1_bundle.py`
- Crear: `tests/test_stage1_runner_static.py`
- Modificar: `src/dermssl/metrics.py`
- Modificar: `src/dermssl/gates.py`

**Interfaces:**
- `validate_feature_bundle(bundle: Mapping[str, ndarray]) -> Stage1BundleEvidence` exige las siete claves actuales de CIRCLe, mismos tamaños, labels enteros, buckets `1..6|None` y `backbone_frozen=True`.
- `build_stage1_result(*, stratified: dict, gate: Gate, bundle: Stage1BundleEvidence, run_id: str, smoke: bool) -> dict` escribe G3, huellas y `valid_for_thesis`.
- El script escribe `stage1_features.npz` y `stage1_results.json`; nunca entrega labels ni buckets al CLI de preentrenamiento.

- [ ] **Paso 1: Escribir RED para la forma del bundle y el gate.**

```python
def test_bundle_rejects_unfrozen_backbone():
    bundle = valid_bundle(backbone_frozen=False)
    with pytest.raises(ValueError, match="congelado"):
        validate_feature_bundle(bundle)

def test_stage1_result_records_g3_and_is_not_final_for_smoke():
    result = build_stage1_result(
        stratified={"max_gap": 0.20}, gate=Gate("G3 equidad por fototipo", "gap <= 0.15", 0.20, "FAIL"),
        bundle=valid_evidence(), run_id="r1", smoke=True,
    )
    assert result["gates"][0]["status"] == "FAIL"
    assert result["valid_for_thesis"] is False
```

- [ ] **Paso 2: Ejecutar RED.**

Ejecutar: `pytest tests/test_stage1_bundle.py tests/test_stage1_runner_static.py -q`

Esperado: falla porque no existe el contrato ni el ejecutor de features.

- [ ] **Paso 3: Implementar el extractor y el JSON Stage 1.**

El script recibirá `--checkpoint`, `--train-image-root`, `--val-image-root`, `--labels-json`, `--fst-manifest`, `--output-bundle`, `--output-json`, `--run-id` y `--smoke`. Validará rutas y JSON antes de importar Torch, cargará el backbone SSL exportado en modo `eval()`, extraerá features originales y desplazadas con el mismo orden de IDs, ejecutará únicamente el probe lineal balanceado y llamará `metrics.stratified_summary` y `gates.fairness_gate`.

```python
np.savez_compressed(
    output_bundle,
    x_train_original=x_train_original,
    x_train_shifted=x_train_shifted,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    fst_buckets_val=fst_buckets_val,
    backbone_frozen=np.array(True),
)
```

- [ ] **Paso 4: Ejecutar GREEN.**

Ejecutar: `pytest tests/test_stage1_bundle.py tests/test_stage1_runner_static.py tests/test_gates.py tests/test_metrics.py -q`

Esperado: pasa; el test estático confirma importación diferida de Torch, y los tests puros verifican JSON, hashes, tamaños y los casos `FAIL`, `PASS` y `SKIP`.

### Tarea 4: Orquestar las cinco etapas y el resultado final

**Archivos:**
- Crear: `scripts/kaggle_complete_experiment.py`
- Crear: `tests/test_kaggle_complete_experiment.py`
- Modificar: `scripts/kaggle_fst_manifest.py`
- Modificar: `scripts/generate_tone_transforms.py`
- Modificar: `scripts/pretrain.py`
- Modificar: `scripts/kaggle_circle_stage2.py`
- Modificar: `src/dermssl/experiment_contract.py`

**Interfaces:**
- `run_flow(args) -> dict` emite `complete_experiment_results.json` y el ledger.
- Salidas canónicas bajo `--work-dir`: `fst_manifest.json`, `mask_manifest.json`, `tone_transform_manifest.json`, `ssl/run.json`, `stage1_features.npz`, `stage1_results.json`, `stage2_results.json` y `complete_experiment_results.json`.
- `stage2_results.json` siempre existe: con `status="SKIPPED"` para G3 `PASS|SKIP`, o con el reporte de las tres variantes para G3 `FAIL`.

- [ ] **Paso 1: Escribir RED de orden, corto circuito y ledger.**

```python
def test_flow_skips_stage2_when_stage1_passes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(flow, "run_stage1", lambda *_: result_with_gate("PASS"))
    monkeypatch.setattr(flow, "run_stage2", lambda *_: calls.append("stage2"))
    result = flow.run_flow(valid_args(tmp_path))
    assert calls == []
    assert result["stages"]["stage2"]["status"] == "SKIPPED"

def test_flow_rejects_an_ssl_run_without_paired_evidence(tmp_path):
    with pytest.raises(ValueError, match="transformado"):
        flow.validate_upstream_artifacts(incomplete_ssl_run(tmp_path))
```

- [ ] **Paso 2: Ejecutar RED.**

Ejecutar: `pytest tests/test_kaggle_complete_experiment.py -q`

Esperado: falla porque el orquestador y su ledger aún no existen.

- [ ] **Paso 3: Implementar llamadas por API, no por scraping de consola.**

`run_flow` debe construir los `argparse.Namespace` de cada script, invocar sus funciones públicas `run`/`train`, y registrar hashes tras cada escritura. No capturar ni interpretar texto de consola. El orden es obligatorio:

```text
FST + mask_manifest
  -> transformaciones tonales
  -> SSL original-transformada
  -> exportación de features + G3
  -> CIRCLe si G3 == FAIL; de otro modo JSON SKIPPED
  -> complete_experiment_results.json
```

Antes de Stage 1, exigir que `ssl/run.json` declare `adaptation_mode == "original_transformed_consistency"`, la revisión/hash del manifiesto transformado y evidencia de pares. Antes de Stage 2, pasar los paths del bundle y Stage 1 JSON existente a `kaggle_circle_stage2.main`; no recalcular G3 ni sintetizar datos.

- [ ] **Paso 4: Ejecutar GREEN.**

Ejecutar: `pytest tests/test_kaggle_complete_experiment.py tests/test_pretrain_transformed_smoke.py tests/test_ssl_run_contract.py tests/test_stage2_runner_static.py -q`

Esperado: pasa; los doubles usados en el test prueban el orden y el gate sin requerir GPU ni datasets.

### Tarea 5: Construir un kernel Kaggle autocontenido y publicarlo explícitamente

**Archivos:**
- Crear: `scripts/build_kaggle_complete_notebook.py`
- Crear: `scripts/kaggle_complete_push.py`
- Crear: `notebooks/kaggle_complete_experiment.ipynb`
- Crear: `tests/test_kaggle_complete_notebook.py`
- Crear: `tests/test_kaggle_complete_push.py`
- Crear: `requirements-kaggle.txt`

**Interfaces:**
- `build_notebook() -> dict` incluye una celda bootstrap que restaura `ssl_flow_sources.zip` en `/kaggle/working/ssl_flow` y hace `sys.path.insert(0, str(root / "src"))`.
- `stage_complete_kernel(user: str, staging: Path, *, real: bool) -> Path` deposita notebook y `kernel-metadata.json` con T4, Internet y los datasets Kaggle requeridos.
- `kaggle_complete_push.py` exige `--via kaggle` y usa exactamente `kaggle kernels push -p <staging>`.

- [ ] **Paso 1: Escribir RED de autocontención y transporte.**

```python
def test_generated_kernel_restores_sources_before_importing_dermssl():
    source = notebook_code(builder.build_notebook())
    assert "ssl_flow_sources.zip" in source
    assert "extractall(WORK_ROOT)" in source
    assert source.index("extractall(WORK_ROOT)") < source.index("import dermssl")

def test_complete_push_requires_explicit_kaggle_transport():
    with pytest.raises(SystemExit):
        push.main(["--user", "u"])
```

- [ ] **Paso 2: Ejecutar RED.**

Ejecutar: `pytest tests/test_kaggle_complete_notebook.py tests/test_kaggle_complete_push.py -q`

Esperado: falla porque no existe el generador ni el publicador integral.

- [ ] **Paso 3: Implementar empaquetado determinista.**

El generador debe incluir sólo `src/dermssl/**`, los scripts invocados por el flujo y `requirements-kaggle.txt`; excluir `.git`, `tests`, notebooks heredados, resultados, checkpoints y credenciales. Ordenar entradas del ZIP, fijar timestamps ZIP y calcular `source_sha256` para que la construcción sea reproducible. La primera celda comprobará el hash del ZIP antes de extraerlo, validará imports de `dermssl`, NumPy, Torch, Torchvision, sklearn y Pillow, y escribirá versiones en el ledger.

El metadata declarará únicamente los datasets Kaggle de ISIC2018 Task 1 y del pool objetivo, `machine_shape: "NvidiaTeslaT4"`, GPU e Internet. El notebook llamará:

```bash
python scripts/kaggle_complete_experiment.py \
  --work-dir /kaggle/working/experiment \
  --run-id "$KAGGLE_KERNEL_RUN_TYPE-$KAGGLE_KERNEL_RUN_ID" \
  --real
```

El publicador mostrará y documentará el comando exacto:

```bash
python scripts/kaggle_complete_push.py --user <usuario> --via kaggle --real
```

- [ ] **Paso 4: Ejecutar GREEN y control de regeneración.**

Ejecutar: `pytest tests/test_kaggle_complete_notebook.py tests/test_kaggle_complete_push.py -q; python scripts/build_kaggle_complete_notebook.py --check`

Esperado: pasa; el notebook versionado es idéntico al generado, no contiene credenciales, no depende del checkout local y el metadata no permite P100.

### Tarea 6: Completar ayudas visibles y documentar la matriz de verificación

**Archivos:**
- Crear: `docs/22-flujo-completo-kaggle.md`
- Modificar: `tests/test_pipeline_localization.py`
- Modificar: `README.md`
- Modificar: `scripts/kaggle_complete_experiment.py`
- Modificar: `scripts/kaggle_complete_push.py`
- Modificar: `scripts/build_kaggle_complete_notebook.py`

**Interfaces:**
- La documentación presenta los artefactos de entrada/salida y los comandos sin secretos.
- `tests/test_pipeline_localization.py` cubre cada CLI de la lista de alcance, incluido `-h` y `--help`.

- [ ] **Paso 1: Escribir RED para la totalidad de ayudas del flujo.**

```python
@pytest.mark.parametrize("main", [
    kaggle_fst_manifest.parse_args, generate_tone_transforms.parse_args,
    pretrain.parse_args, kaggle_stage1_features.parse_args,
    kaggle_circle_stage2.parse_args, kaggle_complete_experiment.parse_args,
    kaggle_complete_push.main, kaggle_run_bank.main,
])
def test_flow_cli_help_is_fully_localized(main, capsys):
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])
    help_text = capsys.readouterr().out
    assert "opciones" in help_text
    assert "muestra esta ayuda y termina" in help_text
```

- [ ] **Paso 2: Ejecutar RED.**

Ejecutar: `pytest tests/test_pipeline_localization.py -q`

Esperado: falla hasta que todos los parsers del alcance usen el creador localizado y las nuevas utilidades existan.

- [ ] **Paso 3: Escribir el runbook y actualizar el README con un enlace.**

El runbook debe contener una tabla que no mezcle afirmaciones:

| Comprobación | Local | Kaggle T4 |
|---|---:|---:|
| Contratos JSON/NPZ, hashes, gate y corto circuito | Sí | Sí |
| Construcción del notebook, ZIP, imports sintácticos y `--help` | Sí | Sí |
| U-Net FST sobre ISIC2018 y pool real | No | Sí |
| Transformaciones del pool real y consistencia SSL | Smoke con fixtures | Sí |
| Checkpoint SSL, extracción de features y G3 real | No | Sí |
| Tres variantes CIRCLe y bootstrap con datos reales | No | Sólo si G3 es `FAIL` |

Incluir los comandos locales de la Tarea 7, el comando de publicación con `--via kaggle`, las rutas de descarga y el criterio de que `PASS`/`SKIP` produce `stage2_results.json` omitido, no un fallo.

- [ ] **Paso 4: Ejecutar GREEN.**

Ejecutar: `pytest tests/test_pipeline_localization.py -q`

Esperado: pasa para todos los CLI del alcance; README enlaza a `docs/22-flujo-completo-kaggle.md` sin introducir resultados nuevos.

### Tarea 7: Verificar localmente y ejecutar la validación remota separada

**Archivos:**
- Modificar: `docs/22-flujo-completo-kaggle.md`
- Modificar: `README.md`

**Interfaces:**
- Ningún resultado real se copia a README hasta que el JSON final tenga `valid_for_thesis=true`, hashes completos y evidencia del gate.

- [ ] **Paso 1: Ejecutar el conjunto local completo antes de una GPU.**

Ejecutar:

```bash
CI_REQUIRE_FULL_DEPS=1 pytest -q -rs
ruff check src scripts tests
python scripts/build_notebook_v5.py --check
python scripts/build_kaggle_complete_notebook.py --check
```

Esperado: todos pasan; los tests GPU-only son estáticos o usan fixtures, no intentan contactar Kaggle.

- [ ] **Paso 2: Ejecutar smoke remoto controlado.**

Ejecutar:

```bash
python scripts/kaggle_complete_push.py --user <usuario> --via kaggle
kaggle kernels status <usuario>/kaggle-complete-experiment
kaggle kernels output <usuario>/kaggle-complete-experiment -p results/kaggle_complete_smoke
```

Esperado: `complete_experiment_results.json` existe con `valid_for_thesis=false`; el ledger registra versiones, fuentes, hashes y una salida Stage 2 `SKIPPED` o smoke, sin publicar una conclusión científica.

- [ ] **Paso 3: Ejecutar la corrida real remota sólo después del smoke.**

Ejecutar:

```bash
python scripts/kaggle_complete_push.py --user <usuario> --via kaggle --real
kaggle kernels output <usuario>/kaggle-complete-experiment -p results/kaggle_complete_real
```

Esperado: los JSON de cada etapa y `complete_experiment_results.json` comparten `run_id`; Stage 2 tiene tres variantes únicamente si G3 es `FAIL`. Si G3 es `PASS` o `SKIP`, el JSON final conserva la omisión explícita y la corrida se considera correcta, no incompleta.

- [ ] **Paso 4: Validar resultados descargados sin reentrenar.**

Ejecutar:

```bash
python -c "from pathlib import Path; from dermssl.experiment_contract import validate_downloaded_run; validate_downloaded_run(Path('results/kaggle_complete_real')); print('JSON válido')"
```

Esperado: valida esquema, `run_id`, hashes encadenados, gate y consistencia entre outputs. Documentar las métricas observadas sin modificar tablas históricas ni crear afirmaciones de UNAB.

## Revisión de cobertura

- FST manifest y máscaras trazables: Tareas 1 y 4.
- Transformaciones tonales preservadoras: Tareas 3 y 4.
- SSL pareado sin etiquetas y evidencia de pares: Tareas 3 y 4.
- Features, Stage 1 y G3: Tarea 3.
- CIRCLe condicionado y resultados JSON: Tarea 4.
- Frescura real de `kaggle_run_bank.py`: Tarea 2.
- `--via kaggle`, empaquetado e imports del kernel: Tarea 5.
- Traducción completa de ayudas visibles: Tareas 1 y 6.
- Separación explícita de prueba local y ejecución Kaggle: Tareas 6 y 7.
- UNAB excluida: restricciones globales, metadata y runbook.

## Control de consistencia y ausencia de placeholders

La implementación debe verificar antes de finalizar que `Stage1BundleEvidence`, `run_id`, `valid_for_thesis`, `stage1_results.json` y `stage2_results.json` conservan exactamente esos nombres en scripts, tests y documentación. Revisar que ninguna tarea deje marcadores pendientes antes de entregar.
