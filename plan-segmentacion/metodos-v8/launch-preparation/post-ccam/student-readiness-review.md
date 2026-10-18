# Estudiante USL-Net: recuperación, observabilidad y abstenciones

Fecha: 2026-10-17. Alcance: revisión estática y propuesta de preparación.
No se ejecutó diagnóstico GPU, no se desplegó código ni se modificaron configs,
gates o la copia de CCAM en ejecución. La propuesta 3×20 sigue sin aprobación.

## Contratos comprobados por lectura de código

| Área | Implementación actual | Límite pendiente |
|---|---|---|
| Entrenamiento | `common/training.py:fit`, AdamW y BCE en píxeles utilizables; pérdida y gradientes no finitos detienen | No hay certificación GPU propia del estudiante |
| Registro por época | `epoch`, `step`, `mean_loss`, `elapsed_seconds`, `sample_count`, en JSONL y sink | No registra cobertura de píxeles, casos sin supervisión útil ni estabilidad de predicciones |
| Recuperación | `resume.pt` atómico: modelo, optimizador, época, paso, sampler/permutación/cursor, pérdidas parciales, resultados y RNG Python/NumPy/torch/loader | Implementación presente; equivalencia real de recuperación pendiente |
| Frecuencia | Inicial, cada `checkpoint_every_steps=100`, al terminar época y ante parada solicitada atendida en frontera de batch | Un fallo abrupto puede perder hasta los pasos posteriores al último checkpoint; no se promete pérdida cero |
| Identidad | Config, manifiesto, gate, revisión, inicialización, arquitectura, tracking, fuentes y protocolo ligados al checkpoint | Cambiar épocas/config/gate/código no es reanudar la misma corrida |
| Salida final | `final.pt` y procedencia con linaje del entrenamiento | No existe selección del mejor checkpoint ni parada temprana del estudiante |
| Geometría | Imagen, pseudo y valid reciben flips emparejados deterministas por semilla/época/índice, `num_workers=0` | La comprobación futura debe incluir recuperación dentro de una época |

Las fuentes están bajo `src/dermssl/segmentation_methods/`: `common/training.py`,
`common/data.py`, `common/losses.py`, `common/inference.py`,
`common/refinement.py` y `usl_net/model.py`. La política de pseudomáscaras está
en `src/dermssl/w3_components.py:USLTextPolicy/usl_text_pseudolabel`.

### Riesgos concretos que debe cubrir el único diagnóstico futuro

1. `fit` carga `resume.pt` con `map_location=device`. `restore_rng` convierte a
   CPU las claves NumPy y los estados CPU/loader, pero pasa la lista
   `torch_cuda` directamente a `torch.cuda.set_rng_state_all`. La diferencia
   respecto a la carga CPU aplicada a CCAM requiere comprobar restauración en
   CUDA. Es un riesgo identificado estáticamente, no un fallo reproducido del
   estudiante ni evidencia de que necesariamente fallará.
2. Al cerrar época se guarda primero `resume.pt` y luego `metrics.jsonl`.
   Una interrupción entre ambas escrituras puede dejar JSONL atrasado respecto
   a `results` del checkpoint. `fit` no reconstruye ese JSONL al entrar en
   recuperación; lo vuelve a escribir al cerrar otra época. Si el checkpoint
   ya agotó el presupuesto, el bucle no lo reconstruye. Esta ventana de
   consistencia se deriva del orden de escritura; no se ejecutó su reproducción.
3. El tiempo registrado de una época reanudada comienza al nuevo proceso,
   aunque `epoch_losses` y `step` conservan progreso previo. No debe usarse ese
   `elapsed_seconds` como duración completa de la época; el supervisor debe
   conservar tiempos de todas las sesiones.

No se afirma que la verificación anterior de CCAM cubra ninguno de estos puntos.
Las correcciones que resulten necesarias se harán en preparación aislada y con
revisión antes del diagnóstico; no se parchea la copia activa de CCAM.

## Política concreta de incertidumbre y abstenciones

### Lo que ya hace el código y se mantiene

- Pseudomáscaras `0 / 0.5 / 1`: `0.5` significa incertidumbre, nunca fondo.
  La BCE ignora estos píxeles y los que tienen `valid=0`.
- La política actual usa saliencia `<0.35` para fondo y `>0.65` para regiones
  candidatas; los empates quedan inciertos. Área mínima 500, conectividad 4,
  sin corte de distancia al centro inventado (`max_centroid_distance=null`).
- Sin región elegible, `DegenerateCase` produce abstención por imagen en
  `cases.jsonl`. Se conserva ID, hash y razón; errores de geometría/hash/runtime
  detienen y no se convierten en abstenciones.
- Generación escribe conteo original, éxitos y abstenciones. Aunque escriba
  `successful_pseudos_manifest.json` y `export_complete=true`, cualquier
  abstención termina con error y bloquea la promoción automática del parcial.
- Un batch completamente incierto/no válido lanza `All-uncertain batch must
  abstain`: actualmente detiene `fit`. No se omite ni se cuenta como actualización.
  En un batch mixto, una imagen sin píxeles útiles puede no contribuir a BCE;
  ese caso requiere contabilidad explícita y no se deduce de `sample_count`.
- En refinamiento, un `DegenerateCase` interrumpe antes de materializar el
  nuevo directorio; no existe una política de exclusión parcial implementada.

### Decisión propuesta, sin habilitación ni cambio de cohorte

Mantener **parada estricta y revisión** ante cualquier abstención por caso o
batch completamente incierto. Es una condición de promoción, no una exigencia
de fabricar 34.784 máscaras: se permite que el generador abstenga y se conserva
el fracaso informado. No modificar umbrales mirando máscaras expertas ni
convertir casos problemáticos en fondo para satisfacer cobertura.

La referencia del método permite descartar casos sin foreground confiable
([USL-Net, sección 3.4](https://arxiv.org/html/2309.13289v3#S3.SS4)). Una adaptación
que excluyera casos podría proponerse después de conocer el ledger real; no
está implementada ni aprobada aquí. Exigiría manifiesto hijo inmutable con
IDs retenidos/excluidos, razones, hashes y recibo contra los 34.784 originales,
vinculación compatible con el protocolo y gates propios revisados. Un subconjunto
arbitrario no pasa los contratos actuales de cohorte congelada. El manifiesto
padre no se modifica; la evaluación reservada mantiene sus 2.594 casos.

## Observabilidad que falta implementar antes del estudiante real

Registrar por época y en recuperación contadores acumulables con denominadores:

- imágenes vistas, imágenes con cero píxeles utilizables y actualizaciones;
- píxeles totales, válidos, inciertos y utilizables; positivos/negativos entre
  utilizables, evitando contar inválidos como incertidumbre semántica;
- tiempo total de sesión y tiempo acumulado, RSS y VRAM pico medidos por el
  supervisor; tamaño de checkpoint y salida/tracking/logs;
- estabilidad de probabilidades sobre un subconjunto fijo de **aprendizaje**,
  seleccionado antes de ejecutar y ligado por IDs/hashes, solo observacional.

No se fija un umbral nuevo de calidad o convergencia con esas métricas. Cambios
de cobertura o estabilidad motivan inspección, no selección contra referencias
expertas ni ampliación automática del presupuesto. Las pérdidas actuales contra
pseudos no son métricas anatómicas. Estos registros son requisitos propuestos,
no campos que el código ya emita.

## Un solo diagnóstico representativo, futuro y solo en UNABIA

Depende de CCAM aceptado, pseudos reales aceptados, resolución/config revisados y
preparación aislada del código. No se ejecutó ni se fija duración arbitraria aquí.
Debe agrupar en una sesión supervisada:

1. IDs de aprendizaje con pseudos/valid reales y geometría de producción;
   resolución 224 y batch 2; suficiente para interrumpir dentro de época y
   cruzar su frontera. Sin máscaras expertas, nuevos splits ni promoción del
   subconjunto diagnóstico a manifiesto científico.
2. Referencia continua y recuperación con identidad idéntica, comparando modelo,
   AdamW, sampler/cursor, RNG, pasos y pérdidas; reconstrucción de métricas sin
   pérdidas/duplicados incluso al recuperar un presupuesto ya completo.
3. Contabilidad de incertidumbre y comportamiento de parada explícito; errores
   conservados en recibo, sin esconderlos como éxito parcial.
4. Tiempo por actualización, RSS/VRAM pico, tamaño de checkpoints/salida y
   extrapolación declarada al pool completo. El mismo recibo fundamentará el
   presupuesto por ciclo y el margen operativo; no acredita convergencia.

Se preservan recibos diagnósticos fuera de resultados científicos y la limpieza
tardía se registra antes de lanzar el estudiante. No repetir auditorías cerradas
de pesos/datos/cuotas ni volver a ejecutar la comprobación CCAM.

## Orden de cierre

1. Revisar/aprobar los cambios mínimos de observabilidad y consistencia de
   recuperación identificados aquí; implementarlos en copia aislada.
2. Al finalizar CCAM, aceptar o rechazar candidato y ligar hashes/linaje reales.
3. Generar pseudos, contabilizar todos los casos y aplicar la parada estricta;
   cualquier propuesta de subconjunto requiere decisión y contratos nuevos.
4. Ejecutar el único diagnóstico del estudiante en UNABIA y cerrar presupuesto
   con mediciones. 3×20, reinicialización por ciclo y selección final siguen
   siendo una propuesta, no autorización para entrenar.
5. Materializar gates/configs por transición y solicitar la habilitación real.
