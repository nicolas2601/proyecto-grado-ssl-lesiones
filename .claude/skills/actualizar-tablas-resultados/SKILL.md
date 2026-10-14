---
name: actualizar-tablas-resultados
description: "Usar cuando una corrida o evaluación haya actualizado JSON y haya que regenerar tablas o resúmenes tabulares del proyecto. No usar para registrar la corrida ni para diseñar gráficas o diagramas."
---

# Actualizar tablas de resultados

Regenera artefactos tabulares desde sus fuentes autorizadas. Los valores no se
corrigen a mano. Esta skill no documenta la procedencia completa de la corrida
ni diseña figuras.

## Procedimiento

1. Lee `CLAUDE.md` §9 y `docs/RUN_CATALOG.md`. Identifica la familia,
   versión, JSON autoritativo modificado y tabla/README afectado. Comprueba
   `git status` para conservar cambios ajenos.
2. Busca los generadores actuales con `Glob("scripts/build_*.py")`. Abre el
   script pertinente, su documentación y el job `generated` de
   `.github/workflows/ci.yml`. El nombre `build_*` por sí solo no demuestra
   que un script sea el generador correcto.
3. Confirma la entrada, salida y argumentos. Usa `--help` o inspecciona
   `argparse`; nunca adivines la interfaz. Como puntos de partida (no como
   lista exhaustiva), el repo ha usado `build_results_tables.py`,
   `build_bench_tables.py`, `build_final_eval_table.py`,
   `build_classifier_table.py` y `build_segmentation_table.py`. Ejecuta solo
   el generador que corresponda al JSON y a la tabla cambiados. Si CI exige
   regenerar artefactos relacionados juntos, sigue ese flujo y explica por
   qué.
4. Ejecuta los tests de frescura que cubren el artefacto. Comprueba primero
   cuáles existen en `tests/`; en este repositorio se han usado
   `test_readme_tables_fresh.py`, `test_resultados_readme_fresh.py` y
   `test_generated_readmes_fresh.py`. Corre el conjunto más pequeño que
   valide los archivos afectados.
5. Para la evaluación final V7 existe este ejemplo documentado; úsalo solo si
   la familia, entrada y salida solicitadas son exactamente las mismas. No
   copies el comando para otra versión:

   ```bash
   PYTHONPATH=src python scripts/build_final_eval_table.py \
       resultados/clasificacion/v7/final_eval.json \
       --out resultados/clasificacion/v7/README.md
   ```

6. Inspecciona el diff: todas las cifras deben corresponder a los JSON fuente
   y el generador debe producir los cambios. Ejecuta `git diff --check`.
   Reporta cualquier fuente ausente o diferencia no explicada.

## Reglas

- No escribas ni ajustes a mano números en tablas o README generados.
- No edites el JSON para que coincida con una tabla; si la fuente parece
  incorrecta, detente y explica el desacuerdo.
- No ejecutes entrenamiento ni evaluación. La corrida debe haber producido
  ya sus artefactos.
- No incluyas métricas de particiones distintas en la misma comparación sin
  etiquetarlas, y conserva unidades, denominadores, semillas y protocolo.
- No incluyas gráficos, diagramas, notas narrativas de corrida, model cards,
  paquetes de release o notebooks en esta skill salvo que la documentación de
  un generador muestre que son parte inseparable de la tabla afectada.
- No hagas commit automáticamente.

## Entrega

Resume el JSON fuente, el generador ejecutado, las tablas regeneradas y los
tests/verificaciones. Si no puedes vincular la tabla con una fuente y un
generador concretos, no la modifiques.
