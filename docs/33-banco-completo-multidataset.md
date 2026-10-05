# 33. Banco completo multi-dataset: re-preentrenamiento limpio y lo que faltaba

**Registrado el 2026-10-02, antes de escribir el código que lo corre y antes de ver cualquier número nuevo.** El bloque entre marcadores no se edita después del commit de registro. Las erratas, si hacen falta, van después del bloque. Todo número publicado sale de un script con `--check` y test de frescura; nada se transcribe.

<!-- registro-banco-completo:inicio -->

## 1. Propósito y alcance

Cerrar el banco de clasificación en HAM10000, ISIC 2019 e ISIC 2020 corriendo solo lo que falta, sin repetir lo que ya está commiteado.

- **Se reutiliza sin volver a correr, cada uno con su split actual (no se unifican):** los baselines congelados (`results/baselines`, `results/baselines_isic2019`, `results/baselines_isic2020`), el fine-tuning de HAM10000 e ISIC 2019 (`results/finetune/*`) y v7.1 en HAM10000 (`resultados/clasificacion/v71`).
- **Fuera de alcance (se declara "no ejecutado"):** split unificado, re-correr los baselines con el protocolo de B71, el fine-tuning de A\*/B\* (el propuesto es SOLO auto-supervisado; el fine-tuning supervisado aparece únicamente en los baselines de comparación), la integración C2 (adaptador residual SimSiam) de Pau y las 3 semillas de D2/D3. Si sobra GPU, primero las semillas 43 de D2 y D3.
- **Segmentación:** entra solo como bloque exploratorio (E3), con selección únicamente en el holdout de HAM. C1-v7 ya mostró que la máscara no mejora (0 de 15 celdas), y Mahbod et al. 2020 (CMPB 197:105725, doi 10.1016/j.cmpb.2020.105725) solo respalda que enmascarar el fondo degrada y que el recorte dilatado no difiere significativamente del baseline; las variantes de fusión son hipótesis propia, no un resultado de esa cita.

## 2. Transductividad del pool SSL y su corrección (opción b)

- El pool de A\*/B\* v7 son las 55.034 imágenes de `salidas/tone/inputs/originals` (HAM10000, ISIC 2019 \ HAM e ISIC 2020) menos `pretrain_exclude_ids_finals.txt` (sha256 `95d21013…`: 3.851 ids, la val y el holdout de HAM y sus casi-duplicados), es decir 51.256 imágenes. **Verificado en el preflight (2026-10-02, `transductividad.json`)** contra los splits existentes reproducidos en el pod:
  - ISIC 2019 (split de los baselines y del fine-tuning, `StratifiedKFold`, val n = 5.067, sha256 `8cde8a71…`, soportes por clase idénticos a los commiteados): **4.039 de 5.067 imágenes de val (79,7 %) estuvieron en el pool efectivo.**
  - ISIC 2020 (`StratifiedGroupKFold` por `patient_id`, val n = 6.660, sha256 `ec3f5408…`, soportes 6.538 / 122 idénticos a los commiteados): **6.311 de 6.660 (94,8 %) estuvieron en el pool efectivo.**
  - HAM10000: 0 (su val y holdout ya estaban excluidos).
- **Hallazgo de reproducibilidad (declarado):** `StratifiedGroupKFold(shuffle=True)` cambió entre scikit-learn 1.7.2 y 1.8.0. Con 1.5.2-1.7.2 se reproduce exactamente el split commiteado de ISIC 2020; con 1.8.0-1.9.0 (el del pod) sale otro (val 6.627). Todo lo que reconstruya un split existente en este banco usa scikit-learn 1.7.2 aislado (`~/work/skl_try/1.7.2`), y el JSON registra la versión. ISIC 2019 usa `StratifiedKFold`, que no cambió.
- **Corrección:** se re-preentrenan A\* y B\* semillas 42 y 43 con la MISMA receta y revisión (`pretrain/v7_finals.py`; presupuesto A 8 h-GPU, B 12 h-GPU), cambiando solo el archivo de exclusión: la exclusión v7 ∪ los ids de val de los splits EXISTENTES de ISIC 2019 (5.067) e ISIC 2020 (6.660); esos splits no tienen holdout. Archivo `salidas/bench/pretrain_exclude_ids_clean.txt`: 14.842 ids, sha256 `3b7e8833…`; pool efectivo limpio 40.906 imágenes (−20 % respecto de 51.256). **Declarado:** el presupuesto de cómputo de la receta no cambia, así que cada imagen del pool limpio se ve ~1,25 veces más; B71-limpio difiere de B71 en los datos, no solo en la exclusión. Las copias byte-idénticas se quitan igual que en v7. El `run.json` nuevo guarda el sha del archivo de exclusión.
- Con esos pesos se re-arman, con su configuración fija y sin re-seleccionar: **B71-limpio** = `B_s42@224|cls4|balanced|tta` y **A71-limpio** = `A_soup(s42+s43)@224|cls_attn|la0.5|tta`. Se reportan al lado del B71/A71 original y no lo reemplazan en silencio. B71 sigue siendo el propuesto en HAM10000.
- Límite que queda: una lesión re-fotografiada con otro id y otros bytes no se detecta en ISIC (no hay `lesion_id`). Los B71/A71 originales en ISIC se reportan con la marca "transductivo (imágenes de val en el pool SSL, sin etiquetas)".

## 3. Pipeline (dos colas, una por GPU; un JSON por celda, reanudable)

**Cola GPU0:** G0.1 re-pretrain A\* s42 limpio; G0.2 re-pretrain A\* s43 limpio; G0.3 D3 (SimSiam continuado desde A\* s42 limpio con HAM `source_ssl_pool` + PAD `target_adapt`, solo imágenes; N = 2.000 pasos o lo que dé ~2,5 h, mismos N, vistas e init que D2).
**Cola GPU1:** G1.1 re-pretrain B\* s42 limpio; G1.2 D2 (igual que D3 pero solo HAM `source_ssl_pool`; control de cómputo); G1.3 fine-tuning de los 8 baselines en ISIC 2020 (receta existente de `dermssl.finetune`: 8 épocas, batch 32, LR backbone 5e-5, LR cabeza 5e-4, AMP, semilla 42, split por `patient_id` reconstruido con scikit-learn 1.7.2 y verificado contra los soportes commiteados 6.538 / 122). **Época elegida en el holdout interno** (el 20 % del train); las filas de ISIC 2019 y HAM10000 ya commiteadas eligieron la época en val y llevan la nota "época elegida en val (sesgo optimista)". Misma receta y revisión que G0.1 para los re-pretrains (f1c2053).
**Huecos de GPU/CPU:**
- **E1** B71-limpio (B s42 limpio, config fija) y A71-limpio (sopa A s42+s43 limpios) en HAM, ISIC 2019 e ISIC 2020; en paralelo B71/A71 originales en ISIC marcados "transductivo".
- **E2** D0 público, D1 (A\* s42 v7), D2 y D3: features → probe logreg ajustado en HAM train (fuente) → aplicado CONGELADO a PAD `target_test` (BCC/MEL/NV), bootstrap agrupado por paciente. Primario D3 − D2. Se declara que el probe se ajusta con etiquetas de la fuente; el backbone nunca ve etiquetas, y el plan de Pau prohibía cualquier refit.
- **E3 (exploratorio)** variantes con segmentación, todas con `student_r3` (sha fijado; las de ISIC que falten se predicen con el mismo checkpoint): F1 concat(features imagen completa, features recorte con margen 10 %); F2 = F1 con recorte desde la resolución original; F3 TTA guiada por máscara (completa + 4 recortes); F4 = F1 + descriptores de forma y color de la máscara; F5 = F1 + anillo perilesional. Selección SOLO en el holdout de HAM; val se lee una vez. **Regla de adopción:** macro-F1 ≥ (mejor variante sin máscara − 0,01) en el holdout Y Δ > 0 en recall de melanoma o en macro-F1 del tercil L\* más oscuro. Si la cumple: "propuesto + segmentación", con contrastes pareados contra B71 y contra cada baseline; si no, se reporta igual y el propuesto sigue siendo B71.
- **E3b (condicional)** si alguna F1-F5 cumple la regla, esa MISMA variante (configuración y máscaras congeladas) se aplica a los 8 baselines congelados en los 3 datasets, con el mismo probe, y se contrasta cada baseline con y sin segmentación y el propuesto + segmentación contra cada baseline + segmentación. En los baselines con fine-tuning va como secundario (variante en inferencia + probe) y solo si sobra GPU. Si ninguna cumple, E3b no se corre y se reporta así.
- **E4** MCC (baselines: exacto desde `matriz_confusion` de los JSON existentes; nuestros: desde las probabilidades), matrices de confusión solo de los auto-supervisados (DINOv2 S y B y SimSiam, congelados y con fine-tuning; A71 y B71 originales y limpios; cuentas y normalizadas, PNG + JSON) y AUPRC en ISIC 2020 y PAD.
- **E5** Fitzpatrick descriptivo de HAM, ISIC 2019 e ISIC 2020 (estimador ITA, **NO validado**: 22,9 % en PAD-UFES-20, PMC13459198; docs/20) con terciles de luminancia L\* (no son tono de piel; relativos a cada dataset), conteos clase × bucket y el ROI perilesional de Pau (anillo fuera de la máscara, excluyendo pelo/reflejos/saturación, con abstención) como estimador alternativo, también descriptivo.
- **E6** `docs/34-entorno-cluster.md` + `resultados/entorno/cluster_env.json` (hardware, versiones, pip freeze aparte, hilos, git rev, por dataset número de imágenes, clases y sha256 de manifiestos y splits; sin hostnames, IPs ni tokens).

## 4. Reglas heredadas del plan de Pau

- PAD se divide por `patient_id` en `target_adapt` / `target_test` (clave de lesión = (patient_id, lesion_id); dedup por id y por hash de contenido); `target_test` queda bloqueado.
- Firewall de etiquetas: los loaders SSL no leen diagnóstico, FST ni máscaras expertas; un test lo verifica.
- W&B fail-closed: se valida `run_id` y URL antes de entrenar y se escriben en `run.json`; si W&B no registra, la corrida aborta. Proyecto `nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-*`. Ninguna URL se reporta sin comprobarla.
- Las predicciones se congelan con su hash ANTES de abrir etiquetas de test o val.
- Rutas nuevas: no se sobrescribe V7, `student_r3`, las máscaras de C1 ni las evaluaciones zero-shot.
- Robustez: cada celda se salta si su JSON existe con el sha correcto; 2 reintentos; lo que falla se lista al final y el resto sigue.

## 5. Métricas, contrastes y lectura

- Por fila: accuracy, balanced accuracy, precisión / recall / F1 macro, MCC, AUC macro OvR, ECE, y por clase precisión, recall, F1 y soporte. IC 95 % bootstrap: 1.000 remuestras, semilla 20260923, agrupado por `lesion_id` (HAM10000), `patient_id` (ISIC 2020, PAD) o por imagen (ISIC 2019, sin grupo real). Para los baselines ya commiteados se usan sus IC; el MCC de los baselines va sin IC si no hay probabilidades.
- **Contrastes pareados** solo donde los dos modelos comparten split e imágenes: B71 contra DINOv2 S y B y SimSiam (congelados y con fine-tuning), B71-limpio contra B71, A71-limpio contra A71, B71-limpio contra A71-limpio, D3 contra D2 en PAD. Entre splits distintos la comparación es de intervalos y se dice así.
- "Mejora" solo si el IC pareado excluye 0; nunca "gana" si el IC toca 0. Todo lo no listado es descriptivo; los bloques exploratorios (E3, E3b, E5) no cambian el primario.
- **Predicción registrada:** en ISIC 2019 e ISIC 2020, B71-limpio queda por debajo de B71 original o igual dentro del IC (|Δ macro-F1| con IC que incluye 0). Si B71-limpio cae con IC que excluye 0, la ventaja de B71 en ISIC dependía en parte de haber visto las imágenes de val.
- **Predicción D3 − D2:** Δ de macro-F1 en PAD `target_test` > 0 con IC pareado que excluye 0; si el IC incluye 0, la adaptación con imágenes de PAD sin etiquetas no se distingue del control de cómputo.

## 6. Entregables

`resultados/banco_completo/` (generados por script con `--check` y test): TABLA_HAM10000.md, TABLA_ISIC2019.md, TABLA_ISIC2020.md (propuesto + los 8 baselines congelados y con fine-tuning, todas las métricas + MCC, columnas de preentrenamiento / fine-tuning / segmentación / transductivo, una fila por condición, ordenadas por F1 macro), TABLA_ABLACION.md (todos nuestros métodos: v7, v7.1, limpios, F1-F5, D0-D3), TABLA_PAD_DA.md, `confusion/`, FITZPATRICK.md, `latex/*.tex` (booktabs, coma decimal) y glosario en español sobre cada .md. Hitos de aviso a Paula: lanzamiento de las colas (con ETA medido y los links de W&B de cada run), fin de los 3 re-pretrains, y todo listo con tests verdes.

<!-- registro-banco-completo:fin -->

## Erratas (fuera del bloque registrado)

- **2026-10-05, lanzador de los re-pretrains.** El "W&B fail-closed" heredado del plan de Pau (sección 4 del bloque) no está implementado en el lanzador `v7up/finals_clean.sh`; solo rige `require_wandb` de `scripts/pretrain.py`, que ya existía. El `run.json` no guarda el `run_id` ni la URL de W&B (solo `wandb_attempt.txt`). Los links de W&B de A\* s42 limpio (`b139bd2dfd0d`) y B\* s42 limpio (`f90cd2e3f142`) se tomaron del log del cliente de W&B y no se verificaron en la web. No se corrige a mitad de corrida: la receta registrada se ejecuta tal cual.
- **2026-10-05, fine-tuning de ISIC 2020 (G1.3).** El entorno del clúster no tiene `timm` instalado y `dermssl.finetune.torch_run` lo importa para cargar los 6 backbones de `timm`; además, la receta existente elige la época por balanced accuracy en val, no en el holdout interno. Ninguna de las dos cosas se resolvió todavía; se registran aquí antes de tocarlas.
- **2026-10-05, resolución del fine-tuning de ISIC 2020 (G1.3).** El plan decía "época elegida en el holdout interno"; se mantuvo la receta existente (época elegida por balanced accuracy en val) por consistencia entre datasets: las filas de fine-tuning de HAM10000 e ISIC 2019 ya commiteadas la usan. Por eso TODAS las filas de fine-tuning de los tres datasets se marcan "optimista: época elegida en val". El propuesto no se ve afectado: no tiene fine-tuning. `timm` 1.0.30 se instaló con `--no-deps --target` en un directorio aislado (ver docs/34) y no se modificó el venv.
