# Inventario de experimentos clf1 (Fase B), versión 2

Actualizado 2026-11-10 con la tabla maestra de referencia (`reference-table.jpeg`). Solo lectura. Detalle en `inventory.json`, sección `reference_table_mapping` (44 filas, cada una con su carpeta).

## Corrección importante
- La versión anterior decía que todos los readouts eran comparables con el protocolo actual. **Eso es falso para el pooling.** Verificado por shape de features: los readouts viejos (SimSiam, VICReg, Barlow, DINO+iBOT, MoCo v3/v4, ablaciones) usan CLS de 768 dims. V10 usa CLS4 de 3072 dims (cls4patch: 6144).
- Mismo reader en todos: balancedLR, nested grouped 5x3, CV seeds 20261025/26/27.
- Consecuencia: la diferencia V10 (0.621/0.675) contra MoCo v3 (0.602/0.644) mezcla encoder, recipe y pooling. No se puede atribuir a la recipe sin medir CLS vs CLS4 sobre el mismo encoder.

## Historia SimSiam → MoCo
1. Campaña v3 (2026-10-27): baseline SimSiam PAD 0.597. VICReg (0.638) y Barlow (0.633) suben PAD. Ganadores provisionales (3 vs 3, permutación min p=0.10).
2. Cola del 2026-10-29: MoCo-v3-style sin ancla da PAD 0.644 y HAM 0.602. a colleague detectó el 2026-11-05 que el ancla estaba apagada (confound; Engram #2344). Con ancla: PAD 0.617, HAM 0.549. El ancla perjudica MoCo.
3. MoCo v4 (10 variantes) y v5: barrido plano, SD ~0.01.
4. MoCo v10 (CLS4, 224/448+D4, cola 4096): candidatos A (V10+448+D4) y B (queue4096+448+D4). Decisión de project owner pendiente.
- El motivo explícito del cambio a MoCo no está en Engram. Se infiere.

## Tabla resumida (medias 3 semillas x 3 CV, salvo 1 semilla donde se indica)
| Método | HAM F1 | PAD F1 | Nota |
|---|---|---|---|
| SimSiam (propuesto) | 0.518 | 0.597 | Referencia |
| VICReg | 0.546 | 0.638 | Ganador PAD provisional |
| Barlow Twins | 0.519 | 0.632 | Ganador PAD provisional |
| Barlow+VICReg | 0.521 | 0.637 | Sin ganancia sobre VICReg |
| DINO+iBOT | 0.527 | 0.607 | Casi atribuible al objetivo |
| MoCo-v3-style (sin ancla) | 0.602 | 0.644 | Confundido; la tabla de referencia no lo marca |
| V10 CLS4 224 (referencia actual) | 0.621 | 0.675 | CLS4, no comparable directo con filas CLS |
| V10 + 448 + D4 | 0.614 | 0.678 | Candidato A (reportado) |
| V10 queue4096 + 448 + D4 | 0.631 | 0.677 | Candidato B (reportado) |

Filas de 1 semilla (todas CLS 768, verificadas en JSON): contraste HAM+PAD 0.639/0.559, SIGReg* 0.622/0.549, MuLan 0.602/0.529, BYOL 0.594/0.507, OOB 0.590/0.531, lr x4 0.609/0.518, long dose 0.602/0.517, freeze4 0.602/0.519, res336 0.598/0.513, balanceo SimSiam† 0.597/0.513, balanceo DINO+iBOT† 0.604/0.536, y el resto de negativos. Las filas con * son aproximaciones "estilo X", no reproducciones del paper. Las filas con † usan etiquetas para elegir imágenes (no SSL puro).

## Huecos (no probados)
- **Pooling CLS vs CLS4 sobre el mismo encoder.** Prioritario, barato (solo readout).
- EN CURSO hoy: debiased contrastive (`debiased-s42`) y cola 16384 (`queue16384-s42`), ambos GPU0.
- Sin resultado verificado: post-procesamiento de features (`moco-v10-phase-b-postproc-20261110`).
- No encontrados: MoCHi, false-negative elimination (no hay lesion_id en mixed-fit), hard-negative reweighting, objetivos patch/local-global, registers, schedule de temperatura coseno.
- No entrenado: SSL sobre DINOv2 ViT-L y ViT-g. Solo congelados públicos (L PAD 0.592; g PAD 0.613).
- No ejecutado: preprocesamiento derm (Shades-of-Gray, hair removal, crops de lesión). `preprocessing-variants-v1` solo cubre squash 224/336/448 con TTA.
- No ejecutada: auditoría SSCD de solapamiento en el pool.
- Pendiente de verificar: PanDerm_base. Mi versión anterior lo marcó "no factible". phase-b-research-derm dice que el repo oficial lo publica, con licencia no declarada. Verificar antes de descartar. Solo como referencia, no como encoder.

## Duplicados
- `moco-v4-m0` es idéntico a moco-v3 sin ancla.
- M10 (DINOv2 nativo) ya corrido como dino-ibot s42/43/44.
- Cola 4096, 448 y TTA D4 ya medidos.
- Color jitter / d4_color: negativo en Fase A. Retirado como propuesta nueva.

## Conflictos
- Track B PAD: 0.763 (brief), 0.737 (email), 0.703 (tabla maestra SigLIP2+meta).
- Sin cifra en JSON: lr x4 (tomada de la carpeta, verificada), MLP y ensamble sobre SimSiam (mapeo probable a `supervised-readers-v1`, no verificado).
- El p=0.010 de MoCo no es p calibrado (Engram #2366).
