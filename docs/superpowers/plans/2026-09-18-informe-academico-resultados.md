# Informe académico y resultados — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidar la documentación académica del proyecto en español, con resultados verificables, tablas, gráficas, decisiones técnicas y trabajos futuros.

**Architecture:** `INFORME.md` será el documento principal para presentación académica. Los documentos de `docs/` conservarán el detalle por tema y enlazarán al informe. Las figuras se derivarán únicamente de JSON/CSV existentes y cada resultado se clasificará como real, smoke o no ejecutado.

**Tech Stack:** Markdown, Python para generación reproducible de tablas/figuras, JSON/CSV de resultados FST y Kaggle, documentación OpenSpec existente.

**Spec:** `openspec/changes/fst-domain-adaptation-circle/` y `.superpowers/sdd/kaggle-complete-experiment-flow/`.

## Global Constraints

- Redactar el contenido nuevo en español académico claro.
- No inventar métricas ni presentar smoke como resultado de tesis.
- No ejecutar Kaggle, UNAB, entrenamiento GPU, commits ni push.
- Preservar el código y los artefactos locales existentes.
- Referenciar explícitamente las limitaciones, máscaras degeneradas y tareas no ejecutadas.

---

### Task 1: Inventario y clasificación de evidencia

**Files:**
- Read: `resultados_fst_v3_*/diagnostico_fst_*.json`
- Read: `results/kaggle_v3_download/RESULTS_SUMMARY_v5.json`
- Read: `docs/19-resultados-bancos-congelados.md`
- Read: `docs/20-validacion-externa-fst-fitzpatrick17k.md`
- Read: `.superpowers/sdd/kaggle-complete-experiment-flow/*.md`

- [ ] Extraer dataset, tamaño, comparables, cambios de bucket, máscaras degeneradas y estado `smoke` de cada JSON FST.
- [ ] Separar resultados reales, smoke, componentes implementados sin ejecución y pendientes GPU.
- [ ] Registrar qué afirmaciones están respaldadas por artefactos y cuáles deben presentarse como trabajo futuro.

### Task 2: Informe principal

**Files:**
- Modify: `INFORME.md`

- [ ] Redactar resumen, problema, objetivos y contribución.
- [ ] Explicar el pipeline: datos, splits, máscaras, estimación FST, SSL, fairness y CIRCLe.
- [ ] Documentar decisiones y descartes: Otsu, constancia de color, máscaras degeneradas, DINOv2, SimSiam, validación externa y restricciones de GPU.
- [ ] Incorporar tablas de resultados FST, resultados SSL/Kaggle y estado de validación.
- [ ] Incorporar conclusiones y trabajos futuros separados por prioridad y dependencia.

### Task 3: Tablas y gráficas reproducibles

**Files:**
- Create or modify: `results/generar_figuras_pipeline.py`
- Create or modify: `docs/figuras_analisis/`
- Modify: `docs/11-resultados-finales.md`

- [ ] Generar una gráfica comparativa de cambios de bucket FST por dataset.
- [ ] Generar una gráfica de cobertura y máscaras degeneradas.
- [ ] Preparar tablas Markdown con fuente, condición experimental, métrica y validez.
- [ ] Añadir títulos, leyendas y notas metodológicas en español.
- [ ] Verificar que cada figura se puede rastrear a un JSON/CSV concreto.

### Task 4: Estado, límites y trabajo futuro

**Files:**
- Modify: `docs/08-estado-actual-y-next-steps.md`
- Modify: `docs/21-checkpoint-manual-circle-stage2.md`
- Modify: `openspec/changes/fst-domain-adaptation-circle/tasks.md`

- [ ] Actualizar el estado real de Task 1, Task 2 y Task 3.
- [ ] Explicar que Stage 2 CIRCLe no tiene resultados reales.
- [ ] Documentar las precondiciones para Stage 1 y Stage 2.
- [ ] Separar trabajos futuros inmediatos, de laboratorio y de validación académica.

### Task 5: Revisión documental

**Files:**
- Review: `INFORME.md`, `docs/11-resultados-finales.md`, `docs/08-estado-actual-y-next-steps.md`, `docs/19-resultados-bancos-congelados.md`, `docs/20-validacion-externa-fst-fitzpatrick17k.md`, `docs/21-checkpoint-manual-circle-stage2.md`

- [ ] Comprobar consistencia de cifras entre documentos.
- [ ] Buscar afirmaciones que confundan smoke con resultado real.
- [ ] Verificar enlaces, nombres de archivos, tablas y figuras.
- [ ] Ejecutar validación Markdown y revisar el diff documental sin modificar código de producción.

