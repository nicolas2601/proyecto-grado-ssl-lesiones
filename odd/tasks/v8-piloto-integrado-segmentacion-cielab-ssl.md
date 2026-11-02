# Transversal v8: preparar piloto de segmentación, CIELAB y clasificador SSL

Estado: preparación; no se ha ejecutado el pipeline integrado.
Autorización: consultar memoria del VPS y estado remoto, elegir un candidato SSL
y preparar el piloto para después de validar la segmentación. Un lanzamiento
nuevo requiere autorización específica con configuración, recursos y destino.

## Objetivo observable

Ejecutar inicialmente un ejemplo técnico de imagen completa que recorra:
RGB → segmentador ResNet34 → selección de piel predicha → transformación de color
en CIELAB → RGB → encoder SSL congelado → lector de clasificación de Nicolás.

El ejemplo deberá registrar las entradas, las máscaras, el control de calidad,
la transformación y las probabilidades finales sin usar una máscara experta de
PAD como entrada. Su éxito técnico no demuestra una mejora científica.

## Candidato y evidencia

La [instantánea generada por consulta SSH](../../resultados/preparacion_v8/piloto_integrado_ssl_20261101/preflight.snapshot.json)
registra métricas, protocolo, hashes y estimaciones de tiempo. Engram VPS aporta
contexto; se contrastó con los artefactos reales de UNABIA.

El candidato para el primer ejemplo es la adaptación **MoCo-v3-style sobre
DINOv2 ViT-B/14, semilla 42**, del clasificador de Nicolás. Es el mayor macro-F1
PAD entre los brazos SSL comparables revisados de la campaña y del seguimiento.
Las semillas 43 y 44 son réplicas de la misma familia, no semillas de CV.
La elección por rendimiento previo hace que el piloto sea exploratorio.

La cola lo identifica con `encoder_uses_diagnostic_labels=false` y
`label_guided_sampling=false`. El lector final es una regresión logística
balanceada que sí aprende de las etiquetas de clasificación. La denominación
SSL corresponde al aprendizaje del encoder, no a un sistema final sin etiquetas.
Es una adaptación local; no se afirma reproducción oficial de MoCo v3.

Checkpoint candidato en UNABIA:
`/home/jovyan/work/clf1_clinical_reference_20261024/runs/moco-v3-s42-full-20261029/simsiam_dinov2_backbone_final.pt`.
Su SHA-256 fue recalculado y coincide con el recibo de extracción de características.
La instantánea conserva el hash completo.

Se conserva como referencia adicional la receta SimSiam+DINOv2 sobre imagen
completa. No se seleccionará la semilla ni el segmentador mirando los nuevos
resultados de clasificación del piloto.

## Condiciones antes de ejecutar

- [x] Terminan 030 y 031 con 300 épocas y concluye la evaluación del lote 026–031.
- [x] Se cotejan las identidades, los hashes de checkpoints y los pares de casos
      entre brazos; se documentan los logs truncados y los intentos rechazados.
- [ ] Se revisa la procedencia de las 100 imágenes IMA++ y sus exclusiones antes
      de habilitar un brazo IMA++. Un Dice alto no cierra ese punto. La referencia
      026 no usa ese cohorte supervisado adicional.
- [x] Se elige y congela el segmentador usando sus datos de selección y controles
      de máscara, sin optimizarlo con macro-F1 de PAD. El checkpoint debe cargar
      con la arquitectura ResNet34 correcta.
- [x] Se prueba el loader del encoder de Nicolás y se conserva su preprocesamiento
      exacto, la correspondencia de clases y los hashes de pesos y manifiestos.
- [x] Se define una transformación CIELAB precisa, con parámetros y QA congelados.
      No se confunde medir color en CIELAB con validar fototipo ni equidad clínica.
- [ ] Se define QA aplicable sin referencia experta: una contaminación calculada
      contra la máscara experta de ISIC no puede usarse directamente como filtro
      en PAD. `tau=0,20` queda como referencia de auditoría, no como gate operativo
      implementado. El umbral del lector de clasificación es otro parámetro.
- [ ] Se prepara un comando real, salida nueva, presupuesto y autorización de la
      acción. No se usa el script antiguo como si ya implementara este protocolo.

## Ejemplo técnico y comparación posterior

1. Congelar un manifiesto pequeño de 10 imágenes del conjunto de desarrollo de
   PAD mediante una regla reproducible anterior a ver resultados del piloto.
   No cambiar sus clases ni pacientes ni elegir solamente máscaras exitosas.
2. Ejecutar el recorrido completo con modelos congelados. Registrar identidad de
   imagen, hashes de entradas, checkpoint de segmentación, máscara predicha,
   parámetros CIELAB, decisión QA, encoder y probabilidades finitas de las seis
   clases en el orden `ACK, BCC, MEL, NEV, SCC, SEK`.
3. Si una imagen no permite estimar piel, conservarla y aplicar una política de
   fallback fijada antes del lanzamiento, con estado explícito. No eliminar
   silenciosamente imágenes del denominador.
4. Para la comparación de desarrollo posterior, usar los mismos 1.134 casos,
   manifiesto, overlay y particiones por paciente del clasificador. A0 es imagen
   original y A1 es la transformación completa con el mismo encoder SSL. El
   procedimiento del lector se mantiene igual en ambos brazos y se ajusta solo
   dentro de los folds de entrenamiento respectivos, sin ajustar el encoder.
5. Reutilizar los folds/OOF de referencia: 5 folds externos, 3 internos y las
   semillas CV `20261025, 20261026, 20261027`, con la misma grilla C del recibo.
   Comprobar primero que A0 reproduce la referencia desde los artefactos fijados;
   detener el contraste si la reproducción falla.
6. Medir macro-F1 como resultado principal, acompañado de ECE, métricas por
   clase y diferencia pareada por paciente. Las tres semillas CV no se cuentan
   como tres entrenamientos independientes del encoder. No declarar validación
   clínica ni usar `target_test` para elegir configuración.
7. Considerar las tres semillas SSL solo después del ejemplo técnico válido.
   Una comparación adicional con QA o mitigación de tono necesita definición
   y criterio registrados antes de observar su resultado. El diseño P4 previo
   debe reconciliarse explícitamente antes de sustituir su encoder o sus gates.

## Salidas y alcance

Salidas previstas: configuración revisada, manifiesto del ejemplo, recibo de
pesos, ledger de etapas, máscaras y controles QA, probabilidades, trazabilidad de
los folds y comparación A0/A1. Las tablas se generarán desde los JSON de salida.

Este documento no habilita ejecución ni cambia gates. No incluye reentrenar
encoders, reemplazar el servicio publicado, cambiar los procesos activos de
segmentación ni declarar acreditada la independencia por paciente.

## Cierre de análisis del lote — 2026-11-02

Los insumos finales se preservaron con hashes y recibos en
[`resultados/segmentacion/v8/ima_ablation_026_031/`](../../resultados/segmentacion/v8/ima_ablation_026_031/README.md).
El análisis post hoc incluye bootstrap pareado por imagen, condicionado a los
seis modelos; no acredita independencia clínica. Las cifras y tablas se generan
mediante `scripts/analyze_ima_ablation_026_031.py`, sin editarlas a mano.

La referencia operativa para el primer ejemplo técnico es **026, control,
semilla 42**. No se declara superioridad científica: se conserva la receta
control mientras se investiga la pérdida de Dice en lesiones pequeñas bajo
IMA++. IMA++ se mantiene como comparación; su menor contaminación del pool de
piel es un beneficio distinto que debe contrastarse en el flujo completo.

El clasificador candidato sigue siendo MoCo-v3-style, semilla 42, con encoder
congelado. Antes de ejecutar faltan QA sin referencia experta para PAD y los
adaptadores compatibles con ResNet34 y el protocolo clínico de Nicolás.
**El piloto no se ha iniciado y permanece deshabilitado.**

## Adaptadores y QA técnico — 2026-11-02

Implementación: `src/dermssl/pilot_adapters.py`. No modifica el pipeline legado.
El segmentador carga el teacher ResNet34 de 026 con estado estricto, hash de
checkpoint, identidad de corrida y hash del código del modelo. Reproduce la
inferencia cruda evaluada, sin añadir SLIC, rescate ni selección de componentes.

El encoder MoCo-v3-style conserva el preprocesamiento original y CLS normalizado
de 768 dimensiones, sin TTA. También se fija el hash del checkout offline de
DINOv2. No hay descarga automática de pesos ni carga insegura con pickle.

El QA implementado es **técnico y heurístico**: valida RGB, forma de máscaras,
campo válido, cobertura de lesión y disponibilidad de piel predicha. Si falla,
la transformación devuelve la imagen original y conserva el caso. No certifica
que la región predicha sea piel pura; por eso el gate de QA clínico anterior
permanece pendiente. La transformación de ejemplo es una perturbación fija de
L* sobre piel predicha, no una normalización ni un cambio de fototipo validados.

La [validación aislada en CPU](../../resultados/auditoria_v8/pilot_adapters_20261102/README.md)
usa checkpoints reales y una imagen sintética, sin imágenes de pacientes.
No equivale al piloto de diez casos. El lector clínico tampoco se ejecutó:
las probabilidades OOF antiguas pertenecen a las imágenes originales; el nuevo
conjunto de features CIELAB requiere recalcular el lector sobre los mismos folds
por paciente, manteniendo A0 como referencia.
