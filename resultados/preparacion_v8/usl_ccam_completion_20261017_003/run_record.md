# Cierre de CCAM V8 `usl_ccam-20261017-003`

## Identidad y evidencia

Familia: segmentación V8; método USL-Net; etapa CCAM; semilla 42. Estado
**completada operacionalmente**, con 30 épocas y código de salida 0. Fuente local
autoritativa de esta revisión: [metrics.snapshot.json](metrics.snapshot.json),
observada en UNABIA a las `2026-10-18T02:47:48.012207+00:00`. El snapshot conserva
los contenidos observados de `selection.json`, `exit.json`, `status.json` y el
historial, sin editar los originales del clúster. La UI de W&B fue observada por
el operador como `Finished`; esa observación complementa la salida del supervisor.

Origen remoto del snapshot:
`/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/completion-metrics-run003-001.log`.
Salida real: `/home/jovyan/work/proyectonp/runs_v8/usl_ccam-20261017-003`.
Inicio: `2026-10-17T23:16:02.889029+00:00`; fin:
`2026-10-18T01:56:53.700321+00:00` (ambos UTC; 18:16:02–20:56:53 del 17 de
octubre en UTC−05:00). Duración: `exit.json.elapsed_seconds=9650.812952130102`,
aproximadamente 2 h 40 min 51 s. `stop_reason=null`, `training_started=true` en
el estado final observado; no se modifica ningún gate por este registro.

W&B: [2761bbcd828a](https://wandb.ai/nmoreno534-universidad-autonoma-de-bucaramanga/dermssl-segmentacion/runs/2761bbcd828a).
Rama: `experiment/modular-audit-v8`; fuente ejecutada: despliegue ZIP basado en
`4a6832c`, inventariado por 57 archivos efectivos; el HEAD Git del checkout
UNABIA no sustituye ese inventario. Configuración, manifiesto, autorización,
comando y estado del árbol al lanzamiento se conservan en el
[registro inicial](../usl_ccam_launch_20261017_003/run_record.md). El estado del
árbol remoto al cierre no está registrado en este snapshot.

## Configuración, datos y protocolo

Config real:
`/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/ccam.run.approved.json`,
SHA-256 `30fd4ad9e870f86009e1b591fab05cb84226f66621297912d8349c43bed7756c`.
Presupuesto fijo de 30 épocas, batch 2, resolución 224, inicio nuevo y selección
predeclarada del primer candidato con estabilidad Pearson >= 0,98; sin parada
temprana. No se eligió una época retrospectivamente por pérdida o máscaras expertas.

Manifiesto:
`/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/learning.manifest.json`,
SHA-256 `7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff`.
Las 30 filas registran `sample_count=34784`; todos los valores numéricos son
finitos según `metrics_summary`. Estos son casos de aprendizaje, sin evaluación
anatómica sobre las 2.594 referencias expertas reservadas. Persisten las
limitaciones clínicas y del banco SimCLR aceptadas en el protocolo congelado.

## Historial y selección

La pérdida `mean_ccam_loss` comienza en 11,906631771180344, alcanza el mínimo
observado de 1,7674453332814581 en la época 6 y termina en 8,923746465639363.
La trayectoria posterior oscila: no demuestra convergencia sostenida.
Los únicos cruces registrados de Pearson >= 0,98 son las épocas 6, 14 y 30;
el monitor corresponde a ocho imágenes fijas de aprendizaje, no al conjunto
reservado ni a una métrica anatómica. Tres cruces aislados tampoco acreditan
estabilidad sostenida.

`selection.json`: `rule=first_stability_threshold`, `selected_epoch=6`,
`status=stability_candidate`, `stopping_reason=fixed_budget_exhausted`,
`early_stopping=false`, `scientific_convergence_verified=false` y
`anatomical_quality_verified=false`. La época 6 es el candidato correcto según
la regla previa; la inspección posterior documentada abajo impide promoverlo
bajo la política vigente de cero abstenciones.

Artefactos declarados en `selection.json`, relativos a la salida real:

- `selected.ccam.pt`: cabeza candidata de la época 6, SHA-256 declarado
  `83bcf068ef91e6b989a5b71b6ac4231368e91a842d6874397c4555f6619edfec`.
- `ccam.pt`: cabeza final de la época 30, SHA-256 declarado
  `443e4c170451f68811d5dab533f57b511754acb446045200d2e6666c38639802`.
- Procedencia declarada por `exit.json.provenance_sha256`:
  `8b7f1f210cc5e1d88eee425128c92cc7ca6079087ad00a942883af420e59e273`.

El [inventario observado](artifacts.snapshot.json) incorpora SHA-256 calculados
por lectura de bytes en UNABIA: ambos checkpoints coinciden con `selection.json`
y el final con `exit.json`. Sidecar seleccionado: `selected.ccam.pt.json`,
SHA-256 `c6a112fecb36116c1b1d513d06465ba2701c3716b66c28462be99f57a7c20b1f`;
sidecar final: `ccam.pt.json`, SHA-256
`8b7f1f210cc5e1d88eee425128c92cc7ca6079087ad00a942883af420e59e273`.
El inventario confirma además la existencia de `resume.pt` (1.786.975 bytes,
SHA-256 `f61e94b63c68a7002696f0f23d0bb38528e40c9d07d99a2e9eaa597afec01c2a`)
y `metrics.jsonl` (SHA-256
`4992b25ed5de3e86acb040f152144f518bc5fee7ab586cfa59bfaab34564c68a`).
Config, gate y `launch.json` mantienen los hashes del lanzamiento. El inventario
de bytes no sustituye una prueba de recuperación del checkpoint de esta corrida.

El [contenido observado de los sidecars y log](lineage.snapshot.json) confirma
época 6 con `selection_status=stability_candidate` y época 30 con
`selection_status=budget_final_not_selected`. Ambos vinculan el manifiesto y
gate anteriores, el protocolo SHA-256
`f4836405a7cbb8309f1757a3488e34ab4fa4f4f02f785b4f72bae79b9d349b03`
y la cohorte SHA-256 `c9b179aef6325b2cf97027ed2214ddd77c71c86ee9178af5b63f4b55b1f708ae`.
Declaran `uses_expert_masks=false`, `uses_diagnoses=false` e independencia por
paciente no acreditada. `producer_revision=95a8f602b2fb49304dbbb9a78a18edc43cc3ebba`
es el HEAD del checkout complementado por ZIP; no identifica por sí solo el
código efectivo ni debe reemplazarse por `4a6832c` en el relato de ese campo.
El log registra `completed=1`, época 30 y tiempo del worker 9.630,16702 s,
distinto del tiempo total del supervisor. Su estabilidad redondeada a 1,0 no
reemplaza el valor exacto de las métricas.

## Recursos y límites de interpretación

Tiempo total observado: el del supervisor indicado arriba. RSS pico, VRAM pico,
tiempos por época, versiones de entorno y huella final de disco: no registrados
en este snapshot; el inventario no contiene un recibo final de picos de recursos.
La información de entorno requiere el sidecar observado por el operador. Los caps
del config y el uso puntual inicial de GPU no se presentan como picos medidos.
No hubo evaluación anatómica, comparación con otros métodos ni validación
experta; terminar el presupuesto no demuestra calidad científica suficiente.

## Inspección de saliencias y bloqueo de promoción

La [inspección GPU observada](saliency.snapshot.json), a las
`2026-10-18T02:55:10.490802+00:00`, usó los primeros ocho originales del
aprendizaje, sin entrenamiento, máscaras expertas, cambios de gates ni
reselección de checkpoint. Con política `low=0.35`, `high=0.65`,
`min_area=500`, conectividad 4, el candidato de época 6 produjo abstención
en **8/8 sondas**: ningún componente elegible, componente mayor de 179 píxeles,
foreground de aproximadamente 0,3567% y fondo de 99,4121% en cada caso.
Las estadísticas son muy similares entre imágenes; este análisis no demuestra
identidad de las matrices píxel a píxel. La media ronda 0,005102.

El checkpoint final de época 30 también produce abstención en **8/8 sondas**:
seis mapas completamente nulos y dos con componente mayor de 179 píxeles.
Los ocho casos no representan estadísticamente los 34.784 originales: no se
extrapola una tasa de abstención global. Sí aportan contraejemplos suficientes
para impedir la promoción bajo el contrato actual de cero abstenciones.
Pearson alto no garantiza una región elegible ni calidad anatómica; el estado
operativo completado coexiste con candidatos no utilizables en estas sondas.

Fuente remota:
`/home/jovyan/work/validation_v8/usl-ccam-launch-20261017-003/saliency-review-run003-002.log`,
SHA-256 `167bd7b550853e3d881c7eb595a29e574729a5de473deb7caa07d134e9f4275f`.
Script de análisis SHA-256
`c95a98122e8ee79fbbed39a4bef2c767e68003fce8817a183ebc8fc8d5345422`.
Su intento inicial abortó antes de inferencia por reiniciar estadísticas CUDA
sin inicializar el dispositivo; el helper corregido no cambió el runtime
científico. La inspección completada duró 1,3958822302520275 s, con CUDA pico
asignado/reservado 464,2099609375/608 MiB. Son recursos del análisis acotado,
no de la corrida de 30 épocas.

## Próximo paso

No promover ni el candidato seleccionado ni el checkpoint final a generación
completa o estudiante con la política actual. Revisar causalmente la cabeza,
objetivo/pérdida y dinámica con batch 2, preservando este resultado y sin
relanzar entrenamiento, cambiar umbrales para forzar elegibilidad, adoptar
5×300 ni seleccionar otra época retrospectivamente. No usar máscaras expertas
para resolver el ajuste. La generación completa y el estudiante siguen sin
ejecutarse; ninguna bandera de ejecución se modifica por este diagnóstico.

## Revisión causal posterior (17 de octubre local)

La [revisión causal acotada](../usl_ccam_causal_review_20261017_003/run_record.md)
ya documenta tres snapshots reales de UNABIA, con cero actualizaciones de pesos.
Confirma saturación de la cabeza y supervivencia de una esquina en el candidato;
los RGB y features conservan diversidad. Los gradientes observados son diminutos.
Esto explica el fallo próximo, pero no aísla el disparador inicial ni acredita
que aumentar batch o modificar escalas lo resuelva. La revisión no cambia el
estado operativo completado ni desbloquea pseudo-máscaras o estudiante.
