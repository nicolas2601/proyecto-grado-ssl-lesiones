# Auditoría modular v8

Estado: exportador implementado; validación local con datos sintéticos.
**Las 100 ternas reales y los nuevos experimentos de segmentación/ITA/dominio no se han ejecutado.**
Plan registrado: [docs/35](../../docs/35-auditoria-modular-v8.md).

## W0: exportar las vistas reales

En la rama `experiment/modular-audit-v8`, con el entorno existente del clúster.
No descargar pesos ni reservar GPU: esta auditoría corre en CPU y un solo proceso.

```bash
PYTHONPATH=src python scripts/audit_ssl_views.py \
  --image-root ~/work/proyectonp/salidas/tone/inputs/originals \
  --transformed-image-root ~/work/proyectonp/salidas/tone/shift2 \
  --pseudo-mask-root /ruta/a/pseudomascaras_png \
  --mask-encoding binary255 \
  --exclude-ids-file ~/work/proyectonp/salidas/bench/pretrain_exclude_ids_clean.txt \
  --recipe plan-segmentacion/comun/configs/audit-v8-paired.json \
  --count 100 \
  --out ~/work/proyectonp/salidas/auditoria_v8/w0_paired_s42
python scripts/build_view_audit_report.py \
  ~/work/proyectonp/salidas/auditoria_v8/w0_paired_s42
python scripts/build_view_audit_report.py \
  ~/work/proyectonp/salidas/auditoria_v8/w0_paired_s42 --check
```

La ruta de pseudomáscaras es una entrada local pendiente, no una ubicación verificada.
Requiere PNG con el MISMO stem que la imagen y la misma rejilla. No redimensionar
silenciosamente. Si las máscaras están en un caché .npy, exportarlas previamente
con el orden/identificadores de su manifiesto; el exportador no adivina correspondencias.
`binary255`: 0=fondo, 255=lesión; `binary01`: 0=fondo, 1=lesión;
`pseudo01255`: 0=fondo, 1=lesión, 255=ignorado.
No pasar máscaras expertas a esta auditoría del pool.

La receta inicial refleja resolution=224, hue_jitter=0.02, seed=42 y las probabilidades
de color_constancy por defecto de PretrainConfig: identity=0.4, gray_world=0.3,
shades_of_gray=0.3. Confirmarlas contra el run.json específico antes de lanzar;
si difieren, crear OTRA receta y registrar la diferencia. Con tone_view=true,
se observan v1 original, v2 transformada y tone alineada. No corrige todavía B2.

Se exige una pareja y pseudomáscara por ID elegible; faltantes y stems duplicados
abortan. No se reducen silenciosamente los 100 casos. El muestreo no estratifica por
dataset: repetir por raíces separadas si se necesita un panel por dataset.
La lista de exclusiones debe ser la de esa corrida/pool; no abrir target_test PAD.

## Artefactos

- `run.json`: receta, versiones, IDs, exclusiones, revisión y hashes del código.
- `cases.jsonl`: hashes de entradas y salidas, recortes y retención por vista.
- `<id>/views.png`: ternas con máscara superpuesta; archivos de vistas/máscaras aparte.
- `status.json`: se escribe complete SOLO si termina la exportación entera.
- `README.md`: tabla descriptiva y enlaces a cada rejilla, generados desde JSONL.

Un fallo deja una carpeta incompleta; corregir la entrada y usar una NUEVA carpeta.
No mezclar salidas de dos recetas. Los archivos grandes siguen en salidas; subir
solo JSON/tabla y ejemplos seleccionados cuando corresponda, con su procedencia.

## Verificación antes de datos reales

```bash
PYTHONPATH=src python -m pytest tests/test_view_audit.py tests/test_torch_view_audit.py -q
python -m ruff check scripts/audit_ssl_views.py scripts/build_view_audit_report.py \
  src/dermssl/view_audit.py src/dermssl/pretrain/torch_view_audit.py \
  tests/test_view_audit.py tests/test_torch_view_audit.py
```

Torch/torchvision son reales en las pruebas locales; no se simula el dataloader.
El entorno local CPU difiere del clúster GPU: los metadatos registran las versiones
de cada ejecución. La prueba sintética no estima rendimiento científico.

