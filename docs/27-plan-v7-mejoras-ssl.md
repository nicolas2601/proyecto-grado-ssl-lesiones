# Plan v7 — mejoras SSL: segmentación, clasificación y comparativa justa

**Fecha:** 2026-09-26
**Estado:** plan aprobado para ejecutar por fases. Ningún experimento arranca sin su predicción registrada.
**Fuentes que integra:** pedidos del profesor Andrés (reunión 2026-09-25), [plan de corrección de Pau](26-plan-correccion-segmentacion-y-clasificador.md), [análisis final y escenarios ADA](24-analisis-final-resultados-y-escenarios-ada.md), [resultados vigentes](../resultados/README.md) y la literatura nueva (sección 5).

---

## 0. Por qué hay una v7

La v6 dejó tres resultados honestos:

1. **La adaptación de dominio sobre DINOv2 sí sirve**: calibración mucho mejor que DINOv2 público (el intervalo excluye el cero en los cuatro backbones) y AUC mejor en ViT-B.
2. **El emparejamiento por tono no aporta**: de 8 contrastes contra su propio control, cero lo favorecen. Pierde de forma afirmable en macro-F1 y balanced accuracy.
3. **La segmentación tiene subcobertura sistemática**: cubre ~0,13-0,15 del área cuando la referencia es ~0,21-0,27 (Pau, [docs/26 §2](26-plan-correccion-segmentacion-y-clasificador.md)).

Además aparecieron dos defectos de diseño que explican parte del resultado 2:
- `consistency_weight = 1.0` es la identidad: el brazo "propuesto" entrenó con la misma pérdida que el control, sin escalar. La palanca de tono estaba en su valor neutro.
- La comparación no estaba emparejada: el propuesto vio 2,8-3,3 % menos muestras, los IDs de validación estaban en el pool de preentrenamiento, y `code_revision`/`split_hash` no quedaron registrados en todos los brazos.

La v7 corrige la medición antes de sacar conclusiones nuevas, y agrega la comparación que pidió el profesor.

---

## 1. Regla SSL-estricta — se chequea en CADA experimento

El proyecto es de aprendizaje **auto-supervisado**. Ningún experimento de la v7 puede entrenar con supervisión.

| Permitido | Prohibido |
|---|---|
| DINOv2 público, declarado como encoder auto-supervisado (se preentrenó sin etiquetas en LVD-142M) | SAM, YOLO-SAM o cualquier segmentador entrenado con máscaras |
| SimSiam, MoCo, CCAM y demás SSL propios | Teachers supervisados (maestros entrenados con etiquetas) |
| Pseudo-máscaras propias, generadas sin etiquetas | Fine-tuning supervisado del backbone |
| Máscaras de referencia y etiquetas `dx` **solo para evaluar**, incluido el probe lineal sobre el backbone congelado (el protocolo estándar para evaluar representaciones SSL) | Active learning con etiquetas de experto (la parte "activa" de ADA, ver sección 5) |
| Máscara de referencia como **oráculo diagnóstico declarado**, solo en partición train/dev, nunca en test, y nunca como método final | Modelos generativos de remoción de pelo entrenados con máscaras de pelo |

**Cómo se aplica:** cada ficha de experimento de la sección 6 trae una línea `SSL:` que dice por qué cumple. Si un experimento no puede cumplir, no se corre.

---

## 2. De dónde partimos

- **Segmentación**: USL-Net (MoCo v2 → CCAM → incertidumbre → U-Net estudiante, 3 rondas). Dice 0,718 en ISIC 2018 (externo) y 0,785 en HAM10000 (dentro del pool). Detalle en [docs/26 §1](26-plan-correccion-segmentacion-y-clasificador.md).
- **Clasificación**: 6 backbones congelados (2 DINOv2 públicos, 2 plain, 2 propuestos), banco en HAM10000 fold-0, probe `logreg`, bootstrap pareado de 1000 remuestras. Tablas en [resultados/README.md §4](../resultados/README.md).
- **Demo**: FastAPI + probe exportado, verificado end-to-end (golden 7/7 en GPU y CPU). Ver [docs/23](23-modelo-descargable-y-uso.md).

---

## 3. Qué pidió el profesor

### 3.1 La comparativa justa: DINOv2 público + segmentación

**El problema que detecta el profesor:** hoy el propuesto usa segmentación (la pseudo-máscara decide dónde desplazar el tono) y DINOv2 público no la usa. Entonces "propuesto vs público" mezcla dos cosas: el método de tono y el uso de segmentación. No se sabe cuál de las dos explica la diferencia.

**Lo que pide:** darle segmentación también a DINOv2 público, para que la comparación contra el propuesto sea de igual a igual en ese eje. Si "público + segmentación" alcanza al propuesto, el tono no aporta nada más allá de la segmentación.

**Interpretación que tomamos (a confirmar con el profesor):** como DINOv2 público está congelado y no se preentrena, la única forma de darle segmentación es **en la entrada del clasificador**: la máscara guía qué parte de la imagen ve el backbone. Se aplica igual a los 6 backbones, para que la tabla sea factorial y no solo "el público con máscara". Es el experimento **C1**.

### 3.2 Segmentación

1. **DINO como maestro, U-Net como estudiante** (destilación) → experimento **S5**. Coincide con la Opción B de Pau.
2. **DINO segmenta directamente** → experimentos **S-diag** (sin entrenar) y **S4** (con cabeza entrenada). Coincide con la Opción A de Pau.
3. **Pasar la máscara al clasificador** para que ayude a clasificar → es el mismo **C1**.

### 3.3 Domain adaptation

→ experimento **D1** y la sección 5 (qué de ADA es SSL y qué no).

---

## 4. Qué propone Pau (docs/26)

Pau diagnosticó la causa raíz en el código y la contrastó con la literatura. Su plan va de lo barato a lo caro:

- **Segmentación** (ella es dueña de esta línea): unión de regiones en vez de una sola (**S1**), remoción de pelo con black-hat + TELEA dentro de la etapa `cache` (**S2**), y usar la `confidence` de FST que hoy se calcula y nadie lee (**S3**). Solo si esto no alcanza, se sube a DINO (**S-diag**, **S4**, **S5**).
- **Clasificador**: ablación emparejada Plain vs Proposed-oráculo, para saber si la calidad de la máscara explica la caída del propuesto (**C3**).

La v7 adopta su orden: primero lo barato, después lo caro, con gates entre medio.

---

## 5. Qué aporta la literatura — y qué se descarta por no ser SSL

| Paper | Qué aporta | ¿Se usa en la v7? |
|---|---|---|
| **Ye 2025**, *Enhancing Skin Lesion Classification Generalization with Active Domain Adaptation* (EMBC) | Workflow: DINO en ImageNet → **SSL en piel** → fine-tune supervisado → ADA activo. Dice textual: *"SSL can be an effective UDA method"*. | **Sí, la parte SSL.** Nuestros pasos 0+1 (DINOv2 público → SimSiam en piel) son exactamente sus pasos 0+1: nuestro trabajo **es SSL como domain adaptation no supervisada**, con respaldo publicado. Aporta el experimento de generalización **D1**. **No** se usa la parte activa: pide etiquetas de expertos, y eso rompe la regla SSL. |
| **Wang 2025**, *SM3: Self-supervised multi-modality learning* (CMPB) | SSL que contrasta imagen dermatoscópica con clínica del mismo paciente, más pseudo-etiquetas por clustering para los 7 atributos. | **No por ahora.** Necesita imágenes clínicas pareadas (derm7pt) que el pool no tiene. La idea de pseudo-etiquetas por clustering (sin etiquetas) queda como trabajo futuro. |
| **Ahammed 2025**, *Dermoscopic images and clinical metadata* (CVPRW) | La metadata clínica (edad, sexo, sitio) mejora la clasificación, y la transferencia le gana al supervisado entre 3 y 12 %. | **Opcional, baja prioridad.** HAM10000 sí trae metadata. Meterla como entrada extra del probe no supervisa al backbone, así que cumple la regla. El dato de transferencia > supervisado respalda el enfoque SSL. |
| USL-Net, Dino U-Net, SegDINO, INSID3, FKD, Dzieniszewska 2024, Bencevic 2024 | Base de la línea de segmentación. | Sí, vía el plan de Pau (docs/26 §5 y §9). |

---

## 6. Plan por fases

Cada ficha tiene: qué es, por qué, chequeo SSL, costo, **predicción registrada antes del dato** y criterio de decisión. La predicción se escribe y se commitea antes de correr. Un criterio decidido después de ver el número no separa explicar de racionalizar.

### FASE 1 — barata, sin reentrenar backbones (arranca ya)

**C1 — La segmentación como entrada del clasificador (6 backbones × 4 condiciones).**
- **Qué:** la pseudo-máscara SSL guía qué ve el backbone congelado antes del probe. Las cuatro condiciones:
  - (0) sin máscara (la base actual);
  - (a) recorte al bounding box de la lesión con margen;
  - (b) fondo neutralizado (fuera de la máscara → gris medio);
  - (c) pooling de los tokens de parche que caen dentro de la máscara, en vez del CLS global.
  Se aplican a los 6 backbones. Mismo banco (HAM10000 fold-0, probe `logreg`) y bootstrap pareado.
- **Por qué:** responde la pregunta central del profesor (3.1) y su pedido 3.2.3 en un solo experimento.
- **SSL:** la máscara es propia y sin etiquetas, los backbones son SSL y el probe es la evaluación estándar. Cumple.
- **Costo:** solo inferencia, del orden de minutos a una hora por condición. Las 24 celdas (6 × 4) son factibles en una tarde de GPU.
- **Predicción registrada:** la condición (c) mask-pooling sube el balanced accuracy de DINOv2 público, porque aísla la lesión del fondo de piel. El recorte (a) puede perder el contexto del borde y rendir igual o peor que (c). Si "público + (c)" alcanza o supera al propuesto con IC que excluye el cero, el aporte del propuesto se explica por la segmentación y no por el tono.
- **Criterio:** la tabla factorial con bootstrap pareado decide. Un "gana" solo cuenta si su intervalo excluye el cero.

**S-diag — DINO segmenta sin entrenar nada (pedido 3.2.2, variante INSID3).**
- **Qué:** features densas de DINOv2 congelado (tokens de parche) → similitud y clustering para segmentar la lesión, sin entrenar ninguna capa. Se compara contra el U-Net actual en el mismo protocolo de evaluación (Dice, HD95, cobertura, en ISIC 2018 externo y HAM10000).
- **Por qué:** es el diagnóstico de una tarde que recomienda Pau (docs/26 §5.3) antes de comprometerse a entrenar algo con DINO.
- **SSL:** DINOv2 es auto-supervisado, no se entrena nada y las máscaras de referencia solo evalúan. Cumple.
- **Costo:** solo inferencia.
- **Predicción registrada:** DINO sin entrenar alcanza o supera la cobertura de CCAM (~0,15), porque no tiene el filtro de región única que corta las lesiones fragmentadas. El Dice puede quedar por debajo del U-Net, que sí pasó por 3 rondas de refinamiento.
- **Criterio:** si la cobertura se acerca a la referencia (~0,21-0,27), se justifica S4 (entrenar la cabeza). Si no, DINO no rescata la saliencia de origen.

### FASE 2 — fixes baratos de segmentación (línea de Pau, en paralelo a la Fase 1)

**S1 — Unión de regiones en `select_central_region`** (docs/26 §2.4). Regenerar solo las pseudo-máscaras y comparar `mean_foreground_fraction` y `excluded_no_region` contra la tabla base (docs/26 §2.1).
**S2 — Remoción de pelo (black-hat + TELEA) dentro de la etapa `cache`** (docs/26 §3.4). Validar en los casos "worst" con pelo y en `tone_distribution`.
**S3 — Usar la `confidence` de FST** (docs/26 §6.4, paso 1). Reportar aparte los registros con `artifact_detected`.
- **SSL:** preprocesamiento clásico (morfología e inpainting sin entrenar) y lógica de selección. Cumple.
- **Predicción registrada:** S1 baja `excluded_no_region` y sube la cobertura sin reintroducir sombras de borde. S2 mejora sobre todo los casos con pelo visible.
- **Cierre de fase:** con las máscaras mejoradas se **vuelve a correr C1**. Así se mide si mejor segmentación trae mejor clasificación.

### FASE 3 — diagnóstico del clasificador y confusores (requiere reentrenar: caro)

**C3 — Ablación emparejada Plain vs Proposed-oráculo** (docs/26 §7).
- **Qué:** reentrenar Plain y Proposed-oráculo (máscara de referencia, **solo partición train/dev, nunca test, declarado como techo y no como método**) con los confusores arreglados:
  - pasos del optimizador e imágenes vistas **igualados** (se corrige el 2,8-3,3 %);
  - IDs de validación **excluidos** del pool SSL (se corrige lo image-transductive);
  - 2-3 semillas por brazo;
  - `code_revision` y `split_hash` registrados en los dos `run.json`.
- **SSL:** el oráculo es un techo diagnóstico declarado, no el método (sección 1). El método final nunca ve la máscara de referencia. Cumple bajo esa condición.
- **Costo:** alto (12 h-GPU por brazo y por semilla). Solo después de la Fase 1.
- **Criterio registrado de antemano** (docs/26 §5.3):
  - si Oráculo ≈ Plain, la calidad de máscara es la causa;
  - si Oráculo ≈ Pseudo < Plain, la máscara es inocente y el problema es el tono o la adaptación;
  - el umbral de "concluyente" se fija antes de mirar.

### FASE 4 — arquitecturas más pesadas (condicionales)

**S4 — DINO + cabeza de decodificación entrenada** (Opción A completa). Solo si S-diag muestra promesa.
**S5 — DINO maestro → U-Net estudiante** (pedido 3.2.1, Opción B). Solo si S-diag, S1 y S2 no alcanzan la cobertura de referencia, o si se necesita un modelo liviano para desplegar.
**T — Rediseño del objetivo de tono** (3 vistas, λ_tone real que sí escala su término, desplazamiento de tono con signo aleatorio). Ahora con la causa raíz identificada (`consistency_weight = 1.0` era la identidad).
- **SSL:** DINO congelado como maestro auto-supervisado, U-Net entrenado solo con pseudo-etiquetas del maestro, y el objetivo de tono es SSL. Cumple.

### FASE 5 — generalización de dominio (pedido 3.3)

**D1 — Generalización cross-dominio.**
- **Qué:** el probe se entrena en HAM10000 y se evalúa, sin reentrenar, en ISIC 2019 e ISIC 2020 como dominios objetivo. Se compara si la adaptación SSL generaliza mejor que DINOv2 público.
- **Por qué:** es la evaluación de dominio que valida Ye 2025, hecha sin la parte activa.
- **SSL:** evaluación cross-dominio; el entrenamiento nunca ve etiquetas del dominio objetivo. Cumple.
- **A resolver antes de correr:**
  - ISIC 2019 tiene 8 clases (HAM tiene 7) e incluye imágenes de HAM10000, así que hay que **deduplicar** y **mapear clases**;
  - ISIC 2020 es binario (melanoma / no melanoma).
  El mapeo de clases se registra antes de correr.

---

## 7. Quién hace qué

| Rol | Hace |
|---|---|
| **cluster** (pod UNABIA, GPU) | Ejecuta todos los experimentos, con TDD en el código nuevo. Registra `code_revision`, `split_hash` y la predicción antes de correr. |
| **opencode** (GPT-5.6-luna, razonamiento máximo) | **Validación independiente con otro modelo, como anti-sesgo.** Antes de cada corrida, revisa el código: ¿respeta la regla SSL? ¿hay fuga de datos (val en el pool, máscara de referencia en test)? ¿el preprocesamiento coincide con el banco? Después de cada corrida, revisa si la conclusión se sigue del dato. Se invoca con `opencode run -m openai/gpt-5.6-luna --dir <repo> -f <archivos> "<pregunta>"`. |
| **Nico + sesión planificadora** | Planifican, registran las predicciones, revisan lo crítico (veredictos, gates entre fases) y deciden cuándo pasar de fase. |
| **Pau** | Dueña de la línea de segmentación (S1-S3 son su plan). Revisa los resultados de segmentación. |

Cada avance a Pau lo redacta la sesión planificadora y lo envía cluster por WhatsApp Web.

---

## 8. Qué NO se hace y por qué

- SAM, YOLO-SAM y cualquier segmentador o teacher supervisado: rompen la regla SSL.
- Fine-tuning supervisado del backbone: rompe la regla SSL.
- ADA activo con etiquetas de experto: rompe la regla SSL.
- Modelo generativo de remoción de pelo: desproporcionado (docs/26 §3.3).
- Cambiar el probe a `logreg_balanced` porque se ve mejor: `logreg` es el probe registrado antes del dato.
- Correr las fases 3-5 antes de tener la Fase 1: son caras y dependen de lo que diga la Fase 1.
- Afirmar "gana" sin un intervalo que excluya el cero.

---

## 9. Pre-mortem

*Supongamos que la v7 falló en un mes. ¿Por qué?*

1. **C1 da un empate generalizado.** No es un fracaso: dice que la segmentación como entrada no mueve la aguja con estas máscaras. Por eso se vuelve a correr C1 después de la Fase 2: puede ser un problema de calidad de máscara y no de la idea.
2. **Fuga de datos no detectada** (IDs de val en el pool, máscara de referencia colada en test). Mitigación: la revisión de opencode antes de cada corrida chequea exactamente esto.
3. **El pod se cae a mitad de una corrida larga** (ya pasó cuatro veces). Mitigación: la cadena de checkpoints, el watchdog y `--max-retries` que ya funcionaron en la v6.
4. **El mapeo de clases de D1 queda mal definido** y la evaluación cross-dominio no es comparable. Mitigación: el mapeo se registra y se revisa antes de correr.
