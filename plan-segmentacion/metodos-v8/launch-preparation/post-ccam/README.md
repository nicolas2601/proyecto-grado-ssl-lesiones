# USL-Net después de CCAM: preparación deshabilitada

Fecha: 2026-10-17. Estado: propuesta preparada, sin ejecución ni despliegue.
Los cinco configs de esta carpeta tienen `execution_enabled=false`. No se
modifica CCAM 003, readiness, gates históricos ni los otros métodos.

## Presupuesto científico propuesto para revisar

Se propone **tres estudiantes nuevos de 20 épocas cada uno**, inicialización
aleatoria con semilla 42 en cada ciclo, batch 2, resolución 224, AdamW LR 0,001
y weight decay 0,00001. La referencia es el antecedente ejecutado de tres rondas
de 20 épocas descrito en `../training-protocol-review.md`; los hiperparámetros
son los del borrador actual. Esto es una adaptación propuesta de 60 épocas en
total; no reproduce las cinco iteraciones de 300 épocas del artículo y todavía
no está aprobada para ejecutar. Se entrega únicamente el config del ciclo 1:
los siguientes se materializan con el manifiesto refinado realmente aceptado.

No se hereda la selección histórica por validación: el código actual `fit`
entrena todo el manifiesto y solo exporta `final.pt`. La regla propuesta es
presupuesto fijo completo y checkpoint final por ciclo; no hay parada temprana
ni búsqueda del mejor checkpoint usando máscaras expertas. El modelo final
propuesto es el final del ciclo 3, decidido antes de la evaluación. Las pérdidas
contra pseudomáscaras describen optimización, no convergencia anatómica.
La misma semilla y arquitectura dan una inicialización comparable entre ciclos;
es reinicialización, no continuación de pesos ni recuperación del optimizador.

La recuperación `--resume` es exclusivamente dentro del mismo ciclo/config/gate/
código/W&B. Cambiar el total de épocas rompe esa identidad. Los checkpoints del
estudiante tienen recuperación implementada, pero su comportamiento real en la
GPU y presupuesto propios todavía no se ha medido: la prueba CCAM no lo acredita.

## Artefactos y dependencias

Raíz de salidas propuesta en UNABIA:
`/home/jovyan/work/proyectonp/runs_v8/usl_net-post-ccam-001/`.
Estos directorios no se han creado ni se afirma que existan.

| Etapa/config | Entrada real requerida | Salida relativa propuesta | Dependencia pendiente |
|---|---|---|---|
| `generate.draft.json` | `learning.manifest.json` congelado | `generate/` | Selección CCAM 003 aceptada, checkpoint y sidecar, hashes y linaje |
| `fit-cycle01.draft.json` | `generate/successful_pseudos_manifest.json` | `fit-cycle01/` | 34.784 éxitos, cero abstenciones y manifiesto aceptado |
| `predict-learning-cycle01.draft.json` | El mismo manifiesto exacto usado en fit | `predict-learning-cycle01/` | `fit-cycle01/final.pt`, sidecar, hashes y linaje |
| `refine-cycle01.draft.json` | El mismo manifiesto exacto usado en predict | `refine-cycle01/manifest.json` | Export anterior completo y verificado |
| Ciclos 2 y 3 | `refine-cycle01/manifest.json`, luego refinamiento del ciclo 2 | Directorios separados por ciclo | Aprobación del presupuesto; configs/gates propios; no iniciar ciclo siguiente automáticamente |
| `predict-evaluation.draft.json` | Manifiesto reservado de imágenes del recibo de freeze | `predict-evaluation/` | Selección final previamente fijada, checkpoint/sidecar ligados |
| Evaluador separado, sin config inventado | Manifiesto reservado, referencias expertas del freeze y export final | `evaluation/` | Predicción completa de 2.594 imágenes y revisión de procedencia |

Manifiesto conocido de aprendizaje:
`/home/jovyan/work/validation_v8/accepted-manifests-20261016-002/learning.manifest.json`,
SHA-256 `7749471410aa98a76dd978cca2ef58f9642d14404530c98e534484ba40a7a6ff`.
Banco: `/home/jovyan/work/public_ssl_v8_preparation/source-bank.usl-net.reviewed-candidate.json`,
SHA-256 `306136b3abd29ae32ac8be2a1456cf411c3f1422f93b39f05d46636141cce6f6`.
El protocolo congelado está ligado en todos los configs; esos campos no reemplazan
el argumento CLI `--manifest`. Las rutas de imágenes/referencias reservadas se
toman del recibo de freeze, sin adivinarlas por su nombre.

Los campos nulos de checkpoint son bloqueos deliberados, no valores consumibles:
`ccam_checkpoint`, `ccam_provenance`, `ccam_checkpoint_sha256`,
`ccam_checkpoint_provenance_sha256`, `ccam_checkpoint_lineage` para generar;
`checkpoint`, `checkpoint_provenance`, `checkpoint_sha256`,
`checkpoint_provenance_sha256`, `checkpoint_lineage` para predecir.
El linaje contiene exactamente `protocol_sha256`, `training_manifest_sha256`,
`training_gate_sha256`, `training_cohort_sha256`, copiados del archivo y sidecar
reales compatibles. No se inventan hashes de archivos futuros.

Cada etapa requiere un gate nuevo ligado al método, al manifiesto y al hash
canónico de su config revisado. Los campos obligatorios reales están en
`common/training.py:gate_report`: datos, presupuesto medido, procedencia de pesos,
smoke, revisión retrospectiva, revisión de código y
`patient_independence_accredited=false`. Ninguno se declara verdadero aquí.
Aceptar limitaciones clínicas no convierte la independencia por paciente en
acreditada. Se conserva además la limitación del espejo SimCLR.

Una abstención en generación impide promover silenciosamente un manifiesto
parcial: `successful_count=34784`, `abstentions=0` y el denominador original son
condiciones de la propuesta. Si falla esa condición se necesita una decisión
explícita de política, sin cambiar umbrales mirando las referencias expertas.
`export_complete=true` por sí solo no acredita esos requisitos.

## Recursos: límites propuestos, no mediciones

| Etapa | Tiempo máximo | RSS MiB | CUDA MiB + reserva | Salida MiB |
|---|---:|---:|---:|---:|
| Generación | 86.400 s | 65.536 | 18.432 + 4.096 | 65.536 |
| Fit por ciclo | 259.200 s | 65.536 | 18.432 + 4.096 | 8.192 |
| Predicción aprendizaje | 86.400 s | 65.536 | 18.432 + 4.096 | 65.536 |
| Refinamiento | 21.600 s | 65.536 | 18.432 + 4.096 | 16.384 |
| Predicción reservada | 21.600 s | 65.536 | 18.432 + 4.096 | 8.192 |

Los tiempos son techos operativos iniciales (24 h para recorrer aprendizaje,
72 h por estudiante y 6 h para CPU/refinado o evaluación), no extrapolaciones
del CCAM ni una duración medida. El cap CUDA deja 4 GiB explícitos y margen en
una GPU de aproximadamente 24 GiB; tampoco acredita que el estudiante quepa.
La cuota agregada sigue siendo referencia de planificación, no cuota individual.

Con N=34.784 y P=224², un mapa float32 ocupa N×P×4 = **6,5018 GiB**.
`inference.py` guarda generación: pseudo float32 + valid float32, dos PNG RGB
(original y overlay), leyenda y manifiestos; predicción agrega probability.npy.
La cuenta conservadora exacta del precheck del CLI es
N×(P×24+16384)/2²⁰ = **40.490,75 MiB** para ambas etapas; 65.536 MiB deja margen
para tracking, PNG, JSON y logs. No se supone compresión favorable. Para 2.594
reservadas la misma fórmula da **3.019,5781 MiB**, cubiertos por 8.192 MiB.
NPY agrega cabeceras; el margen de metadatos y del cap evita tratarlas como cero.

Refinado escribe un mapa por imagen: el precheck da
N×(P×4+4096)/2²⁰ = **6.793,75 MiB**. Se proponen 16 GiB; el código además acumula
mapas en RAM antes de escribir, al menos 6,5 GiB más estructuras/manifest/listas.
No duplica los originales ni valid.npy: los manifiestos refinados los referencian.
Por ello no se puede borrar la generación inicial mientras existan descendientes.

Fit conserva `resume.pt`, su temporal atómico transitorio, `final.pt` y sidecar,
config, métricas y tracking. AdamW necesita estados adicionales al modelo; el
cap de 8 GiB es reserva operativa, pendiente medir tamaño real, no una cifra de
parámetros inventada. `checkpoint_every_steps=100` mantiene el contrato actual.

Retención conservadora sin borrar etapas: generación 64 GiB + tres fits 24 GiB
+ dos predicciones de aprendizaje 128 GiB + dos refinados 32 GiB + predicción
reservada 8 GiB = **256 GiB**; reservar **320 GiB libres** para evaluación,
tracking, logs, temporales y margen compartido. No es consumo observado.
El evaluador requiere supervisión propia de tiempo/RSS/disco: no acepta flags
`--config`, `--gate` ni `--resources`. Propuesta: 6 h y 8 GiB adicionales, fuera
de los configs del runner. Mantener originales, exports, checkpoints y recibos
hasta cerrar evaluación; no comprimir encima de las entradas referenciadas.
El CLI hace un precheck de espacio y controla tiempo/RSS; el cap de salida no
debe presentarse como un monitor continuo de disco. El supervisor debe contar
salida + tracking + logs y vigilar espacio real durante cada etapa.

## Secuencia CLI condicionada, no plan ejecutable

No se entrega JSON al controlador: este exige todas las rutas reales y no admite
la etapa `evaluate`. Tras materializar y revisar cada transición, los comandos
tienen las siguientes formas reales. Las variables son insumos obligatorios
del futuro recibo de lanzamiento; no se inicializan a rutas supuestas aquí.

```bash
python scripts/run_segmentation_v8.py --method usl_net --stage generate --config "$GENERATE_CONFIG" --manifest "$LEARNING_MANIFEST" --gate "$GENERATE_GATE" --out "$GENERATE_OUT" --wandb-project dermssl-segmentacion
python scripts/run_segmentation_v8.py --method usl_net --stage fit --config "$FIT_CONFIG" --manifest "$PSEUDO_MANIFEST" --gate "$FIT_GATE" --out "$FIT_OUT" --wandb-project dermssl-segmentacion
python scripts/run_segmentation_v8.py --method usl_net --stage predict --config "$PREDICT_LEARNING_CONFIG" --manifest "$PSEUDO_MANIFEST" --gate "$PREDICT_LEARNING_GATE" --out "$PREDICT_LEARNING_OUT" --wandb-project dermssl-segmentacion
python scripts/run_segmentation_v8.py --method usl_net --stage refine --config "$REFINE_CONFIG" --manifest "$PSEUDO_MANIFEST" --gate "$REFINE_GATE" --predictions "$PREDICT_LEARNING_OUT" --out "$REFINE_OUT" --wandb-project dermssl-segmentacion
python scripts/run_segmentation_v8.py --method usl_net --stage predict --config "$PREDICT_EVALUATION_CONFIG" --manifest "$EVALUATION_MANIFEST" --gate "$PREDICT_EVALUATION_GATE" --out "$PREDICT_EVALUATION_OUT" --wandb-project dermssl-segmentacion
python scripts/evaluate_segmentation_v8.py --manifest "$EVALUATION_MANIFEST" --references "$EXPERT_REFERENCES" --predictions "$PREDICT_EVALUATION_OUT" --out "$EVALUATION_OUT" --wandb-project dermssl-segmentacion --code-revision "$DEPLOYED_FULL_REVISION"
```

Son comandos documentales, no un script para ejecutar en bloque. Los ciclos 2/3
repiten fit/predict/refine con entradas y salidas nuevas y revisadas; no hace falta
refinar después del ciclo final. W&B recibe métricas/procedencia por etapa, nombres
de salida únicos, sin imágenes expertas ni overlays (`save_review` es local al
clúster). Conservar también métricas y recibos locales; W&B no sustituye evidencia.

## Qué falta para ejecutar

1. Resultado CCAM 003 y candidato aceptado; si no alcanza 0,98 no promover el final automáticamente.
2. Aprobar o ajustar la propuesta 3×20, reinicialización y selección final fija.
3. Medir una pasada GPU representativa del estudiante en UNABIA, una vez preparado
   el presupuesto coherente; verificar recuperación con el contrato de código real.
4. Enlazar hashes/linajes y gates reales por tramo, inventariar espacio disponible,
   preparar supervisión de salida/tracking y transportar por ZIP a una copia que
   preserve el runtime activo. No ejecutar ninguna etapa desde estos borradores.

No se ejecutaron pruebas locales, GPU, descargas ni operaciones de clúster para
crear estos archivos; la revisión realizada aquí es lectura estática del contrato.
