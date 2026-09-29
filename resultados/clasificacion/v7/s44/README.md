# Evaluación única v7 (val)

Probe `logreg`, n = 2004, `code_revision` 1d1109eaf2c8382fab1b0e5ed7aae9f5de356a17. Generado desde `final_eval.json`. No editar a mano: `--check` lo compara en CI.

| Brazo | Macro-F1 | Bal-acc | AUC | ECE |
|---|---|---|---|---|
| A* | 0,6843 [0,6398; 0,7248] | 0,6445 [0,5995; 0,6927] | 0,9639 [0,9562; 0,9709] | 0,0239 [0,0189; 0,0401] |
| B* | 0,6725 [0,6267; 0,7139] | 0,6405 [0,5946; 0,6865] | 0,9643 [0,9587; 0,9698] | 0,0204 [0,0166; 0,0381] |
| DINOv2 público | 0,6158 [0,5690; 0,6550] | 0,5924 [0,5459; 0,6376] | 0,9257 [0,9123; 0,9370] | 0,1307 [0,1159; 0,1476] |

| Contraste | Δ Macro-F1 | Δ Bal-acc | Δ AUC | Δ ECE |
|---|---|---|---|---|
| A* − DINOv2 público | 0,0684 [0,0245; 0,1105] | 0,0521 [0,0096; 0,0943] | 0,0382 [0,0286; 0,0482] | -0,1069 [-0,1215; -0,0832] |
| B* − DINOv2 público | 0,0567 [0,0142; 0,0998] | 0,0481 [0,0017; 0,0953] | 0,0385 [0,0292; 0,0495] | -0,1103 [-0,1218; -0,0879] |
| B* − A* | -0,0117 [-0,0476; 0,0271] | -0,0040 [-0,0409; 0,0385] | 0,0004 [-0,0049; 0,0065] | -0,0034 [-0,0180; 0,0136] |

| Predicción (docs/28 §5) | Criterio | Veredicto |
|---|---|---|
| P1 | macro-F1 A* - público >= -0.01 | cumplida |
| P2 | ECE B* - A* < 0 con IC que excluye el cero y macro-F1 >= -0.01 | no cumplida |
| P3 | macro-F1 B* - público >= 0 y ECE B* - público < 0 con IC que excluye el cero | cumplida |
| P1-NI | límite inferior IC macro-F1 A* - público >= -0.01 | cumplida |
| P3-IC | límite inferior IC macro-F1 B* - público > 0 (permite decir que le gana) | cumplida |

## ECE con IC basic (lectura secundaria)

IC basic (reverse-percentile) de ECE, lectura secundaria. El ECE remuestreado tiene sesgo hacia arriba, así que el IC percentil puede dejar el punto afuera (A* en val v7). El basic refleja los cuantiles alrededor del punto y puede salir de [0, 1]. Los veredictos primarios usan el IC percentil registrado y no cambian.

| Brazo o contraste | ECE / Δ ECE | IC basic | ¿Punto fuera del IC percentil? |
|---|---|---|---|
| A* | 0,0239 | [0,0076; 0,0289] | no |
| B* | 0,0204 | [0,0028; 0,0243] | no |
| DINOv2 público | 0,1307 | [0,1138; 0,1456] | no |
| A* − DINOv2 público | -0,1069 | [-0,1306; -0,0922] | no |
| B* − DINOv2 público | -0,1103 | [-0,1328; -0,0988] | no |
| B* − A* | -0,0034 | [-0,0205; 0,0112] | no |

P2 leída con el IC basic: no cumplida (el veredicto primario no cambia).

## Procedencia

Generado desde `eval_provenance.json`.

- **Evaluación (semilla 44).** `scripts/c1_final_eval.py --split val --single-evaluation` desde el clon `repo_s44` en 1d1109eaf2c8382fab1b0e5ed7aae9f5de356a17 (limpio), `CUDA_VISIBLE_DEVICES=0`, `OMP_NUM_THREADS` sin fijar, `OPENBLAS_NUM_THREADS` sin fijar, `MKL_NUM_THREADS` sin fijar; 2026-09-29 08:01:15 a 08:05:56 UTC. Log: `logs/eval.log`.
- **Entrenamiento.** Las finales de s44 se entrenaron en 1d1109e; las de s42, s43 en f1c2053. `tests/test_pretrain_finals_characterization.py::test_head_trains_the_finals_like_f1c2053` exige que el entrenamiento sea el mismo.
- **Control (semilla 42), antes de la evaluación.** El mismo comando y el mismo entorno, 2026-09-29 07:56 a 08:01 UTC. Contra `resultados/clasificacion/v7/final_eval.json`: las probabilidades `plain`, `proposed`, `public`, `y_true` son idénticas bit a bit (dtype incluido), y `results`, `paired` del JSON son idénticos. Log: `logs/control_s42_eval.log`.
- **Arranque abortado.** 2026-09-29 07:53:35 a 07:55:46 UTC: arranque abortado a los 2 min 11 s, antes de puntuar (carga del hub y extracción): sin probas ni métricas, ningún JSON; no cuenta como lectura de val. Carpeta conservada como evidencia en el pod: `salidas/v7/eval_final_s44_aborted_0755`. Log: `logs/aborted_0755_eval.log`.
- **Entorno.** `final_eval.environment.json` se capturó en un proceso hermano con las mismas variables de hilos (`threads_env.json`), porque el código que evaluó (1d1109e) es anterior al contrato de sidecars (3fa9292); no registra versiones de paquetes ni el estado de torch.
