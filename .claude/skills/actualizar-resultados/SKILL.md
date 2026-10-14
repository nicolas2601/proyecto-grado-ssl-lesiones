---
name: actualizar-resultados
description: Usar después de que termine una corrida de entrenamiento o evaluación de este proyecto, para regenerar las tablas/figuras de resultados afectadas desde su fuente JSON y verificar que nada quedó editado a mano. Invocar con frases como "actualiza los resultados", "regenera las tablas", "corrí una evaluación nueva, actualiza el README de resultados".
---

Flujo determinista, no un juicio abierto: regenerar tablas/figuras desde JSON
y verificar frescura. No decidís qué significan los números — solo los
regenerás y confirmás que nadie los tocó a mano.

## Por qué existe este flujo

Convención explícita del proyecto (`CLAUDE.md` §9): "No copiar cifras a
mano. Las tablas de resultados se generan desde JSON (`resultados/*.json`,
`results/*.json`) vía scripts `build_*`; un test falla si alguien edita un
número a mano." El job `generated` de CI (`.github/workflows/ci.yml`) aplica
exactamente esta misma secuencia.

## Generadores reales en este repo

`scripts/build_*.py` (confirmado con `Glob` al escribir esta skill — más de
30 archivos; esta es una muestra representativa, no la lista completa):

- `scripts/build_results_tables.py`, `scripts/build_results_readme.py` —
  tablas y README de resultados generales.
- `scripts/build_bench_tables.py`, `scripts/build_bench_all.py`,
  `scripts/build_bench_ablation.py`, `scripts/build_bench_contrasts.py` —
  banco de benchmarks.
- `scripts/build_model_card.py`, `scripts/build_release_notes.py`,
  `scripts/build_release_package.py` — artefactos de release del modelo.
- `scripts/build_final_eval_table.py` — tabla de evaluación final (la que usa
  el flujo de reproducción v7 del `README.md` raíz).
- `scripts/build_classifier_table.py`, `scripts/build_segmentation_table.py`
  — tablas específicas de clasificador y de segmentación.
- `scripts/build_baseline_tests.py`, `scripts/build_finetune_tests.py` —
  regeneran los notebooks de Kaggle (no son tablas de resultados, pero
  también son "generado desde fuente", y CI los corre en el mismo job).
- Hay además generadores de propósito más puntual: `build_c1_readme.py`,
  `build_pad_ufes_table.py`, `build_pad_ufes_fst_table.py`,
  `build_pad_ufes_wiseft_table.py`, `build_pad_ufes_addendum_table.py`,
  `build_fitzpatrick_table.py`, `build_ranking_table.py`,
  `build_t4_table.py`, `build_v71_table.py`, `build_seed_summary.py`,
  `build_closing_table.py`, `build_ft_contrasts.py`, `build_site_data.py`,
  `build_split_attestation.py`, `build_view_audit_report.py`,
  `build_audit_progress.py`, `build_modular_report.py`,
  `build_cluster_env.py`, `build_kaggle_diagnostico_fst.py`,
  `build_blas_limitation_note.py`, `build_notebook_v5.py`.

**No asumas que esta lista sigue completa**: antes de ejecutar nada, corré
`Glob("scripts/build_*.py")` (o el equivalente de tu entorno) para confirmar
cuáles existen al momento de correr esta skill, y cuál corresponde al tipo de
resultado que cambió.

## Pasos

1. **Identificá qué JSON cambió.** La corrida que acaba de terminar escribió
   o actualizó un JSON bajo `resultados/*.json`, `resultados/*/*.json`, o
   (para bancos históricos) `results/*.json`. Ese archivo es la fuente de
   verdad; no edites ningún `.md` directamente.
2. **Corré el/los generador(es) que corresponden a ese JSON.** Ejemplo real,
   tomado del comando de reproducción documentado en el `README.md` raíz
   (sección "Cómo reproducir la evaluación v7"):

   ```bash
   PYTHONPATH=src python scripts/build_final_eval_table.py \
       resultados/clasificacion/v7/final_eval.json \
       --out resultados/clasificacion/v7/README.md
   ```

   Para los generadores que CI corre siempre juntos (porque comparten una
   celda base y arreglar uno sin regenerar el otro deja el otro desfasado —
   ver comentario en `.github/workflows/ci.yml` job `generated`, basado en el
   incidente real del 2026-09-03 con la celda de extracción vgg19):

   ```bash
   python scripts/build_baseline_tests.py
   python scripts/build_finetune_tests.py
   ```

   Si no estás seguro de los argumentos exactos de un generador puntual,
   leé su `--help` o las primeras líneas del script (`argparse`) en vez de
   adivinar — no todos los `build_*.py` comparten la misma firma de CLI.

3. **Verificá que nada quedó sin regenerar o editado a mano.** Corré los
   tests de frescura reales del repo:

   ```bash
   pytest -q tests/test_readme_tables_fresh.py -rs
   pytest -q tests/test_resultados_readme_fresh.py -rs
   pytest -q tests/test_generated_readmes_fresh.py -rs
   ```

   (Confirmados con `Glob` al escribir esta skill: los tres existen bajo
   `tests/`.) Si alguno falla, el fix es correr de nuevo el generador
   correspondiente y volver a verificar — nunca editar el `.md` a mano para
   que el test pase.

4. **Para los notebooks generados**, CI además compara contra `git status`
   porque `git diff` es ciego a directorios nuevos sin seguimiento (comentario
   real en `ci.yml`): si el paso anterior tocó notebooks, corré

   ```bash
   git status --porcelain -- notebooks/
   ```

   y confirmá que coincide con lo que acabás de regenerar (no debería quedar
   nada sin commitear que el generador no haya producido él mismo).

5. **Lint.** Corré `ruff check .` (configuración en `ruff.toml`, solo reglas
   de error real: `E9, F63, F7, F82, F401`) sobre cualquier script que hayas
   tocado manualmente (no sobre el `.md` generado, que `ruff.toml` excluye
   junto con `results`, `data`, `assets`, `demo`, `notebooks`).

6. **No commiteés automáticamente.** Esta skill regenera y verifica; no decide
   si el resultado nuevo es aceptable ni crea el commit. Dejá los archivos
   regenerados listos para que el usuario (o el flujo ODD/SDD en curso)
   decida el commit, siguiendo la convención de Conventional Commits con
   scope de hito ya usada en este repo (`CLAUDE.md` §9).

## Qué NO hacer

- No edites un `.md` de `resultados/` a mano para "arreglar" un test de
  frescura que falla — el fix siempre es correr el generador de nuevo.
- No inventes un comando `make test` único: no existe en este repo
  (confirmado — no hay `Makefile` ni comando unificado); los comandos reales
  son los de `README.md` (`pytest -q`, `ruff check .`) más los generadores
  puntuales de arriba.
- No asumas que un generador nuevo (no listado arriba) existe sin confirmarlo
  con `Glob` primero.
