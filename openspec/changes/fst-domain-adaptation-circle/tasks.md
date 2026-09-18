# Tareas: FST/CIRCLe

## Review Workload Forecast

| Campo | Valor |
|---|---|
| Líneas estimadas | 2.500–3.600 |
| Riesgo de 400 líneas | Alto |
| Partición sugerida | 5 unidades verificables |
| Estrategia de entrega | ask-on-risk |
| Estrategia de cadena | pendiente |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: pending
400-line budget risk: High

> Estado al 2026-09-18: el código de `circle.py`, `kaggle_circle_stage2.py`, `kaggle_circle_bundle.py` y sus pruebas ya está en la rama, pero las casillas 4.1-4.3 y 5.1-5.2 se mantienen abiertas hasta verificarlas contra esos criterios de aceptación, y 5.2 requiere GPU en Kaggle. La estrategia de cadena sigue pendiente.

| Unidad | Límite autónomo | Dependencia |
|---|---|---|
| U1 | Datos/splits | — |
| U2 | Máscaras/FST | U1 |
| U3 | SSL | U1 |
| U4 | CIRCLe/métricas | U2, U3 |
| U5 | Pruebas/GPU | U4 |

Paralelismo: tras CP1, U2 y U3 pueden ejecutarse en paralelo; U4 y U5 son secuenciales.

## Fase 1: Datos/splits

- [x] 1.1 Crear `pool_manifest.py` y extender `data_resolver.py` con inventario de 68.472 imágenes, SHA-256 y cobertura. [Req.: Ingesta y cobertura] Aceptación: conteos 10.015/25.331/33.126 y JSON válido.
- [x] 1.2 Implementar splits por paciente/lesión y duplicados exactos/perceptuales. [Req.: Ingesta y cobertura] Dep.: 1.1. Aceptación: ningún grupo cruza splits; limitación ISIC2019 registrada.
- [x] 1.3 Probar unicidad compuesta, cobertura y ausencia de leakage. Dep.: 1.1–1.2. CP1: `pool.json` y `split_hash` reproducibles.

## Fase 2: Máscaras/FST

- [x] 2.1 Implementar `mask_manifest.json`, QA, cuarentena y máscara `true=lesión` en `fitzpatrick.py`; limitar ISIC2018 a segmentador auxiliar. [Req.: FST y máscaras trazables] Dep.: CP1. Aceptación: faltantes/degeneradas no son elegibles.
- [x] 2.2 Actualizar `diagnostico_fst_comparativa.py` y `kaggle_fst_manifest.py` para completo/enmascarado, `unavailable` y JSON-first. [Req.: FST y máscaras trazables] Dep.: 2.1. Aceptación: CSV solo exportación.
- [x] 2.3 RED/GREEN/REFACTOR: probar fallback, trazabilidad, calidad y preservación píxel a píxel. Checkpoint CP2: cobertura y calidad aceptadas o fallo documentado.

## Fase 3: SSL

- [x] 3.1 Extender `pretrain/` y `scripts/pretrain.py` con DINOv2 base/adaptado, pesos, reanudación y presupuesto GPU. [Req.: Adaptación SSL] Dep.: CP1. Aceptación: cargador excluye diagnóstico, tono e IDs clínicos.
- [x] 3.2 Añadir ResNet50 ImageNet y matriz pareada con igual `split_hash`, semillas y probe. [Req.: Adaptación SSL] Dep.: 3.1. Aceptación: variantes incompatibles se rechazan.
- [x] 3.3 Comparar la rama `origin/feature/v6-gpu-ssl-pretraining`; integrar cambios selectivos probados. CP3: checkpoint/objetivo acordados antes de GPU.

## Fase 4: CIRCLe/métricas

- [ ] 4.1 Implementar en `circle.py` y `circle_torch.py` transformaciones con máscara y variantes referencia/transformación/CIRCLe. [Req.: Transformación y CIRCLe] Dep.: CP2–CP3. Aceptación: no preservadoras se excluyen.
- [ ] 4.2 Extender `metrics.py` y `gates.py` con globales, tono, EOD, NAR, bootstrap, `null` y puerta ±0,02. [Req.: Métricas y puerta] Dep.: 4.1. Aceptación: grupos escasos quedan `SKIP`.
- [ ] 4.3 Integrar `kaggle_circle_stage2.py` para `run.json` y `evaluation.json`. CP4: promoción bloqueada sin reducir brecha.

## Fase 5: Pruebas GPU

- [ ] 5.1 Probar contratos JSON, split, etiquetas SSL, reanudación, métricas y puertas. Aceptación: `CI_REQUIRE_FULL_DEPS=1 pytest -q -rs --strict-markers` y `ruff check .` pasan.
- [ ] 5.2 Ejecutar smoke y GPU: `python scripts/diagnostico_fst_comparativa.py`, `python scripts/pretrain.py` y `python scripts/kaggle_circle_stage2.py`. Dep.: CP4. Aceptación: registrar RNG, CUDA, AMP y checkpoints; smoke no habilita tesis.

## Corrección transversal de localización

- [x] L1 Localizar al español la ayuda argparse, mensajes de consola, errores, comentarios y documentación visible de los utilitarios Python, incluidos los flujos Kaggle heredados, sin cambiar flags, claves JSON, identificadores ni workflows. Aceptación: contratos focalizados y `ruff` pasan sin ejecutar Kaggle ni entrenamiento.

## Límites de alcance

- ISIC2018 no es principal ni clasificador FST; CNN solo como fallback documentado.
- No promover CIRCLe ni integrar modelo sin puerta favorable.
- No afirmar validez clínica ni mejora desde probes.
