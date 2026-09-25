# Demo local DermSSL

Demo local de la rama `proyecto-de-grado`. Sirve el reporte sellado de
`resultados/` y, si se configura el checkpoint externo, ejecuta inferencia con
el backbone congelado y el probe compatible.

## Qué está versionado

| Artefacto | Ubicación | Estado |
|---|---|---|
| API FastAPI | [`backend/app.py`](backend/app.py) | Versionada. |
| Servicio de modelo | [`backend/model_service.py`](backend/model_service.py) | Versionado; carga pesos de forma perezosa. |
| Reporte | [`backend/report.py`](backend/report.py) | Lee `resultados/site_data.json` y los JSON del banco. |
| Frontend | [`frontend/`](frontend/) | Versionado. |
| Probe y manifiestos | [`models/`](models/) | Versionados; permiten validar el contrato. |
| Imágenes golden | [`models/golden_images/`](models/golden_images/) | Versionadas para pruebas y presentación. |
| Backbone `.pt` | Externo | 331 MB; no se sube a Git. |

El checkpoint se configura sin modificar el repositorio:

```bash
export MODEL_WEIGHTS_PATH=/ruta/al/ssl_proposed_vitb14_backbone.pt
```

También se pueden definir `MODEL_MANIFEST_PATH`, `MODEL_PROBE_PATH`,
`MODEL_CLASSIFIER_MANIFEST_PATH` y `DEMO_DEVICE`.

## Ejecutar

Desde la raíz del repositorio:

```bash
python -m venv .venv-demo
source .venv-demo/bin/activate
pip install -r demo/requirements-cpu.txt
uvicorn demo.backend.app:app --host 127.0.0.1 --port 8765
```

Abrir <http://127.0.0.1:8765>.

## Verificar

```bash
pytest -q tests/test_demo_api.py tests/test_demo_model_service.py tests/test_demo_report.py
```

La suite verifica 13 casos, incluyendo el reporte final, el probe, la carga de
fixtures golden y el modo TTA de tres vistas.

## Endpoints

| Endpoint | Propósito |
|---|---|
| `GET /api/status` | Estado del backbone, probe y manifiestos. |
| `GET /api/report` | Reporte consolidado desde artefactos versionados. |
| `POST /api/features` | Extrae la representación congelada. |
| `POST /api/predict` | Clasifica una imagen; acepta `?mode=base` o `?mode=tta`. |

La API limita cargas a 10 MB, acepta JPEG/PNG/WebP y añade headers de seguridad.

## Límite científico

La interfaz distingue la predicción del modelo de la clase real de una imagen
golden mediante hash SHA-256. La etiqueta de benchmark no se infiere del nombre
del archivo y no se usa para cambiar la predicción. El demo es una herramienta
de presentación y reproducibilidad local, no un dispositivo de diagnóstico.

La lectura científica está en [`../resultados/README.md`](../resultados/README.md)
y [`../docs/24-analisis-final-resultados-y-escenarios-ada.md`](../docs/24-analisis-final-resultados-y-escenarios-ada.md).
