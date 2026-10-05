# Diseño: FST, adaptación de dominio y CIRCLe

## Enfoque técnico

Pipeline por etapas con JSON versionado y puertas: FST provisional, adaptación SSL sin etiquetas, transformación y CIRCLe. Las especificaciones en `specs/` rigen cobertura, comparación, equidad y promoción. Cada variante comparte partición y evaluación.

## Decisiones de arquitectura

| Decisión | Alternativa descartada | Justificación |
|---|---|---|
| Registro por `(dataset, image_id, revisión)` y huellas de contenido; una fila por cada una de las 68.472 entradas | `data_resolver.pool_images` como manifiesto | Ese índice sobrescribe colisiones por `image_id` y excluye HAM del pool; no acredita cobertura ni duplicados. |
| Partición congelada por paciente o lesión, más componentes de duplicados exactos y perceptuales entre fuentes | División aleatoria por imagen | HAM10000 está contenido en ISIC2019; se impide compartir imágenes o grupos conocidos entre entrenamiento y evaluación. ISIC2019 sin agrupación clínica fiable conserva una limitación declarada. |
| Máscaras automáticas versionadas, con Otsu marcado como fallback y cuarentena de calidad; ISIC2018 solo ante insuficiencia documentada | Entrenar ahora una CNN o llamar válidas a todas las pseudo-máscaras | La validez FST depende de la máscara; ISIC2018 solo podría entrenar un segmentador auxiliar. |
| DINOv2 preentrenado con continuación SSL sobre partición de entrenamiento; ResNet50 ImageNet como control separado | Mezclar arquitecturas y objetivos como una única mejora causal | Comparaciones pareadas dentro de cada backbone; sin etiquetas de diagnóstico, tono ni IDs clínicos en el cargador SSL. |
| CIRCLe condicionado a brecha medible: base, solo transformación y transformación más consistencia | Equiparar `regularization` y `synthetic_blend` actuales con esas variantes | El runner actual prueba otras ablaciones; requiere una adaptación explícita y conserva sus resultados previos. |

## Flujo de datos y límites

`data_resolver` -> manifiesto completo y huellas -> divisiones congeladas -> máscaras/QA -> FST completo y enmascarado -> puerta FST -> adaptación DINOv2/ResNet50 -> puerta de equidad -> transformación protegida -> CIRCLe -> JSON comparativo.

El manifiesto completo sirve al diagnóstico FST. El preentrenamiento usa solo imágenes de entrenamiento, sin metadatos en el cargador. Validación y prueba se congelan antes de ajustar umbrales o checkpoints. Fitzpatrick17k aporta contraste externo provisional, con solapamientos y discordancias registrados; no valida clínicamente. Se informa cobertura y concordancia por método, calidad y fuente, incluidos casos `unavailable`.

## Cambios previstos en archivos

| Ruta | Diseño de integración |
|---|---|
| `src/dermssl/data_resolver.py`; nuevo `src/dermssl/pool_manifest.py` | Conservar resolución existente; añadir inventario sin colapsar fuentes, huellas y particiones. |
| `src/dermssl/fitzpatrick.py`; `scripts/diagnostico_fst_comparativa.py`; `scripts/kaggle_fst_manifest.py` | Consumir máscara trazable, separar estimación completa/enmascarada y emitir JSON primario; CSV heredado como exportación. |
| `src/dermssl/pretrain/{config.py,torch_data.py,torch_loop.py}`; `scripts/pretrain.py` | Adaptación sin etiquetas, origen de checkpoint, semillas, presupuesto GPU y reanudación reproducible. |
| `src/dermssl/circle.py`; `src/dermssl/finetune/circle_torch.py`; `scripts/kaggle_circle_stage2.py` | Máscara obligatoria, transformaciones auditables y variantes de ablación; mantener separado el probe del modelo completo. |
| `src/dermssl/metrics.py`; `src/dermssl/gates.py` | Métricas por tono, intervalos y puerta de éxito, sin alterar puertas históricas. |

La rama `origin/feature/v6-gpu-ssl-pretraining` de Nico se compara por interfaz y revisión antes de incorporar cambios selectivos. JSON desacopla evaluación de PyTorch en Kaggle o clúster.

## Contratos

`pool.json`: `schema_version`, `dataset`, `image_id`, `source_revision`, `sha256`, `duplicate_group`, `patient_or_lesion_group`, `split`, `coverage_status`. `mask_manifest.json`: clave de imagen, `producer`, `version`, `mask_sha256`, `quality`, `area_fraction`, `eligible`, `fallback_used`. `fst.json`: ITA, bucket o `unavailable`, estado, confianza, método y procedencia de máscara por imagen. Una máscara booleana `H×W` usa `true=lesión`; faltante o degenerada prohíbe transformar.

`run.json`: `schema_version`, revisiones de datos/código, `split_hash`, backbone, pesos iniciales y adaptados, objetivo, variante, probe, semilla, GPU, precisión, épocas, presupuesto y checkpoints. `evaluation.json`: predicciones vinculadas al mismo `split_hash`, métricas globales y por tono, tamaños/omisiones, matriz, EOD, NAR, brecha máxima, bootstrap pareado por grupo y veredictos. EOD usa diferencia máxima de sensibilidad por clase entre grupos elegibles; la definición operativa de NAR se fija antes del análisis. Valores no estimables son `null` con motivo, nunca cero.

## Verificación y despliegue experimental

Pruebas de contrato: conteos por fuente, unicidad compuesta, duplicados y grupos sin cruce, ausencia de etiquetas en SSL, máscaras y lesión preservada píxel a píxel, igualdad de `split_hash`, reanudación y JSON válido. En GPU, registrar semillas de Python/NumPy/Torch/CUDA, estado RNG, optimizador, scaler, época y huellas en checkpoints; usar AMP según hardware y `eval()` en validación. Smoke no habilita tesis. La puerta exige brecha menor y caídas absolutas de macro-F1 y AUC globales de como máximo 0,02 frente a base comparable; grupos insuficientes producen `SKIP`. Solo evidencia favorable permite proponer integración CIRCLe al entrenamiento completo.

## Preguntas abiertas

- Umbrales de cobertura/calidad de máscara y muestra experta; mínimo por tono (valor actual: 30).
- Fórmula y dirección de NAR, brecha primaria definitiva y checkpoint/objetivo de adaptación acordados antes de correr GPU.
- Alcance de la promoción al modelo completo tras evidencia positiva del probe.
