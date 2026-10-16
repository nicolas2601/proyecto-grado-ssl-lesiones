# Plan de corrección: ruido de segmentación y ruta de decisión del clasificador

**Fecha:** 2026-09-26
**Propósito:** documentar, con causa raíz y fuentes, los fixes acordados para el pipeline de pseudo-máscaras (`camino-b-cnn`) y la ruta de experimentación recomendada para decidir si la fuga de máscaras explica la caída del clasificador `proposed` vs `plain`. Este documento junta lo ya registrado en `docs/25-diagnostico-mascaras-clasificador.md` con la exploración de código y de literatura hecha para decidir los fixes concretos.

División de trabajo acordada: la segmentación es un resultado propio de la tesis (alimenta además la estimación ITA no validada, que no es FST) y se corrige de forma independiente; el diagnóstico y las pruebas del lado del clasificador quedan a cargo de Nico. Ambas líneas corren en paralelo, ninguna bloquea a la otra.

---

## 1. Segmentación — arquitectura real confirmada

El pipeline en `plan-segmentacion/camino-b-cnn/` **no es un U-Net supervisado clásico**. Es una implementación de **USL-Net** (Uncertainty Self-Learning Network, Li et al. 2024 — ver sección 5, referencia [1]):

```
MoCo v2 (backbone, sin preentrenar)
   -> CCAM (mapa de saliencia clase-agnóstica, backbone congelado)
   -> uncertainty_module.py (genera pseudo-máscara 0/1/255-ignorar)
   -> U-Net estudiante (BCE + Dice suave, solo sobre píxeles "confiables")
   -> rondas de auto-entrenamiento (el estudiante reetiqueta con sus propias predicciones)
```

Las máscaras reales de HAM10000/ISIC2018 **solo se usan para evaluar** (`evaluate_segmentation.py`), nunca para entrenar (`DISENO.md:9`). Esto es intencional (evita etiquetado manual masivo) pero significa que cualquier sesgo del generador de saliencia se hereda directo por el red de segmentación entrenada con pseudo-máscaras.

---

## 2. Problema 1 — subcobertura sistemática

### 2.1 Evidencia (docs/25, sección 7 y per_image.csv)

| Dataset | Dice medio | p10 | Cobertura predicha | Cobertura referencia | % imágenes con subcobertura |
|---|---:|---:|---:|---:|---:|
| HAM10000 | 0,7847 | 0,5075 | 0,1977 | 0,2664 | 62,4 % |
| ISIC2018 | 0,7177 | 0,4015 | 0,1697 | 0,2140 | 50,2 % |

Datos de las rondas de auto-entrenamiento (`resultados/2026-09-21-final/pseudo_r{0,1,2}/pseudo_masks_stats.json`):

| Ronda | fuente | excluded_no_region | excluded_by_area | mean_foreground_fraction |
|---|---|---:|---:|---:|
| r0 | ccam (saliencia cruda) | 670 | 4.252 | 0,1305 |
| r1 | student | 913 | 1.640 | 0,1476 |
| r2 | student | 877 | 1.631 | 0,1433 |

**Verificado, no es un problema de tamaño de imagen:** se revisó `evaluate_segmentation.py` y los `summary.json` reales de `piloto`, `final`, `v2` y `baselines_v2` — predicción y máscara de referencia siempre se cargan al mismo `resolution` dentro de cada corrida, y HAM10000/ISIC2018 comparten checkpoint y resolución (224px) en todas las rondas revisadas. No confundir con la validación de FST/ITA sobre ISIC2019 (`docs/20`, sección 3), donde el resize también se sospechó y también se descartó por separado (diferencia de ~2,7 puntos entre resized y full-res) — son dos investigaciones distintas, ambas con resultado negativo para la hipótesis de resize.

**Lectura:** no hay sesgo que se retroalimente sin control — la cobertura mejora de r0 a r1 y se estabiliza en r1→r2. Pero el techo (~0,13-0,15) queda muy por debajo de la referencia (~0,21-0,27) en las tres rondas, y `excluded_no_region` (fallo de "ninguna región válida encontrada") se mantiene alto y estable en las tres rondas — no se autocorrige con más entrenamiento.

### 2.2 Causa raíz (código, `plan-segmentacion/camino-b-cnn/pseudo-mascaras/uncertainty_module.py`)

`select_central_region` (líneas 68-96) aplica dos filtros y se queda con **una sola región**:
1. Descompone la saliencia alta en componentes conectados, descarta los menores a `min_area_fraction=0.003`.
2. De las regiones que quedan, calcula la distancia de cada una al centro de la imagen y se queda **solo con la de menor distancia** (`DEFAULT_MAX_CENTER_DISTANCE=0.5` descarta las demasiado descentradas). El resto pasa a "incierto" (255), nunca a foreground.

### 2.3 Fuente — por qué existe este filtro y por qué no basta

Esto **no es un bug de esta implementación**: es el diseño original de USL-Net. Cita textual del paper (Li et al. 2024, sección 3.4):

> "In cases where multiple saliency regions exist, the regions close to the image boundaries are often associated with black shadow noise rather than true lesion regions... The region with the minimum weight is labeled as the foreground, while the remaining regions are considered as uncertain locations."

El filtro existe para **rechazar sombras de viñeta en el borde de la imagen**, no para descartar fragmentos legítimos de la misma lesión. El problema es que en presencia de pelo, brillos o lesiones irregulares, una lesión real puede partirse en varios blobs — y el diseño original solo estaba pensado para el caso "un blob real + ruido de borde", no para "una lesión fragmentada en varios blobs reales".

### 2.4 Técnica de corrección

**No** ampliar a "aceptar cualquier región" (eso reintroduciría el ruido de sombras de borde que el filtro original evita). En su lugar: **unión de todas las regiones que ya pasan el mismo filtro de área y centralidad**, en vez de quedarse solo con la de menor distancia.

- Mantener: descarte por `min_area_fraction` y por `max_center_distance` (esto sigue rechazando las sombras de borde, igual que en el diseño original).
- Cambiar: en vez de `foreground = region_con_menor_distancia`, usar `foreground = unión(regiones que pasan ambos filtros)`.
- Si ninguna región pasa los filtros, mantener el comportamiento actual (`REASON_NO_REGION`, exclusión de la imagen).

### 2.5 Cómo aplicarlo

**Archivo:** `plan-segmentacion/camino-b-cnn/pseudo-mascaras/uncertainty_module.py`, función `select_central_region` (líneas 68-96).

1. Después del filtro de área, calcular la distancia al centro de **cada** región que sobrevive (no solo ordenar y tomar la primera).
2. Filtrar por `max_center_distance` sobre todas ellas (no solo sobre la ganadora).
3. Marcar como foreground la unión (OR lógico) de las regiones que pasan ambos filtros.
4. Si el conjunto resultante es vacío, mantener la lógica de exclusión existente (`REASON_NO_REGION`).
5. Registrar en `pseudo_masks_stats.json` cuántas imágenes pasaron de 1 región a 2+ regiones aceptadas, para poder medir el efecto sin necesidad de reentrenar el red de segmentación entrenada con pseudo-máscaras todavía.

**Validación antes de tocar el red de segmentación entrenada con pseudo-máscaras:** regenerar solo la etapa de pseudo-máscaras (`make_pseudo_masks.py`) con el fix, y comparar `mean_foreground_fraction` y `excluded_no_region` contra la tabla de la sección 2.1. Si no se mueven, el problema está más en la saliencia de origen (CCAM) que en la selección de región.

---

## 3. Problema 2 — pelo, tinta y brillos no cubiertos

### 3.1 Evidencia

Los overlays "worst" ya generados (`resultados/2026-09-21-final/evaluacion_ham10000/overlays/worst.png` y equivalente ISIC2018) muestran fallas concentradas en pelo, tinta, brillos, reglas de medición y lesiones pálidas de bajo contraste.

`artifacts.py` ya filtra dos artefactos, ambos por umbral de color puro (dilatados 2px), usados en `uncertainty_module.py:34,130-131,154-155` para marcar píxeles como "ignorar":
- `ink_mask`: `B-R >= 0.08` y saturación `>= 0.25` (tinta violeta).
- `highlight_mask`: canal mínimo `>= 0.92` y spread `<= 0.06` (brillos especulares).

**No existe ningún código de remoción de pelo en todo el repositorio.** Solo se menciona como riesgo conocido en `README.md`/`DISENO.md`, nunca implementado. El pelo oscuro y las reglas de medición no tienen firma de color simple, así que quedan completamente fuera del filtro actual.

### 3.2 Fuente — esto es una limitación admitida por los propios autores de USL-Net

Cita textual de la sección de discusión del paper (Li et al. 2024, sección 5):

> "Due to the complex context information in challenging scenarios such as purple iodophor regions and hair occlusions, USL-Net does not completely avoid the failures in learning the pseudo-labels for salient regions."

Es decir: no es un defecto de esta implementación puntual, es una limitación reconocida y nunca resuelta del método base. Corregirla es una contribución real, no un parche cosmético.

El propio paper de USL-Net cita trabajo previo en la misma línea — Ramya et al. (2021, citado como [21] en Li et al. 2024, y catalogado también en Sharma et al. 2025 como referencia [52]) removieron tinta y tarjetas de color mediante transformada wavelet discreta + umbralización Otsu/histograma, antes de segmentar. Es el mismo principio: limpiar artefactos en la imagen de entrada, no en la máscara de salida.

Sharma et al. (2025), el review de arquitecturas de segmentación de piel más reciente disponible en la bibliografía del proyecto, confirma que la remoción de pelo como preprocesamiento de imagen (antes de la red) es práctica estándar en modelos recientes: TrUNet (2024, código público) y otro modelo con encoders Res2Net/Transformer usan explícitamente "hair removal, contrast enhancement" como paso previo a la segmentación.

### 3.3 Estado del arte reciente (2024-2025) en remoción de pelo

Búsqueda de literatura reciente, dos familias de técnicas:

**Familia pesada (modelos generativos entrenados específicamente para esto):**
- Hexa-GAN — inpainting con GAN de muestreo hexagonal (*Biomedical Signal Processing and Control*, 2024).
- GAD-VAE — disentanglement adversarial con autoencoders variacionales (2024).
- DPA-HairNet — red dual-encoder con atención (*IEEE Access*, 2025).
- DM-AHR — modelo de difusión condicional self-supervised (*Cancers*, 2024).

Todas requieren entrenar y mantener un modelo generativo propio, con su propio dataset de máscaras de pelo. Desproporcionado para este caso: el objetivo es corregir ruido en un pipeline de segmentación, no producir una contribución de remoción de pelo per se.

**Familia liviana (morfología + inpainting clásico), validada de nuevo en 2025, no reemplazada:**
- EHR (2025): combina Deep Residual U-Net con el algoritmo de inpainting TELEA para eliminar pelo antes de segmentar.
- Pipeline en *Scientific Reports* (2025): operaciones morfológicas para detectar el contorno del pelo, seguidas de inpainting para rellenar el hueco, antes de segmentar con Mask R-CNN + GrabCut.

### 3.4 Técnica recomendada y cómo aplicarla

**Recomendación: familia liviana — detección morfológica (black-hat) + inpainting (TELEA), como paso de preprocesamiento de imagen, antes de MoCo/CCAM.** No entrenar un modelo generativo nuevo: es la ruta desproporcionada para el alcance de este fix.

**Dónde va:** temporalmente sigue siendo antes de segmentar (antes de `contrastive(moco)`, que es donde arranca la segmentación) — eso no cambia. Lo que cambia es la ubicación en el código: no como paso aislado suelto, sino **dentro de la etapa `cache` que ya existe** (`entrenamiento/pipeline_stages.py:200-206`, función `_cache_stage`), que ya corre primera en el orden del pipeline (`env → cache → contrastive → ccam → pseudo → student → fst → evaluate`, línea 23) y que ya es la etapa que le entrega la imagen a todas las demás. Operando sobre la imagen RGB de entrada — no sobre la máscara ni sobre la saliencia.

**Procedimiento (por imagen):**
1. Convertir a escala de grises.
2. Aplicar un filtro morfológico black-hat con un kernel alargado (aproxima la forma lineal del pelo) para resaltar estructuras finas y oscuras sobre el fondo de piel.
3. Umbralizar la respuesta del black-hat para obtener una máscara binaria de pelo.
4. Aplicar inpainting (TELEA o Navier-Stokes, ambos disponibles en OpenCV) sobre la imagen original usando esa máscara, para reconstruir la piel/lesión debajo del pelo.
5. La imagen resultante, sin pelo, es la que entra a MoCo/CCAM en lugar de la original.

**Por qué antes de la red y no como filtro de máscara (como hoy `artifacts.py`):** si se filtra después (como con `ink_mask`/`highlight_mask` hoy), la saliencia de CCAM ya fue generada sobre una imagen con pelo — el daño a la calidad de la saliencia ya ocurrió. Limpiando la imagen de entrada, la saliencia en sí misma mejora, en vez de solo enmascarar el síntoma en la salida.

**Por qué dentro de `cache` y no como paso aislado, verificado en código:** `plan-segmentacion/comun/evaluacion/fst_pseudo_manifest.py` (la etapa `fst`, sección 6) **no lee la caché del pipeline de segmentación** — carga cada imagen directo desde `row["path"]` del manifest (`_load_image`, línea 99-111), que apunta al archivo fuente original, no al `images.npy` que produce `_cache_stage`. Si la remoción de pelo se insertara como paso suelto solo antes de `contrastive`/`ccam`, la estimación de ITA/FST seguiría viendo imágenes con pelo sin limpiar. Poniéndolo dentro de `cache`, en cambio, todas las etapas posteriores (`contrastive`, `ccam`, `pseudo`, `student` y `fst`) leen la misma imagen ya limpia, sin cablear el fix por separado en cada consumidor.

**Validación:** correr el pipeline completo desde `contrastive(moco)` con las imágenes limpias de pelo para un subconjunto de prueba, y comparar contra los casos "worst" ya identificados en los overlays — si la subcobertura en esos casos específicos (los que muestran pelo en el overlay) se reduce, el fix está funcionando. Adicionalmente, comparar el `tone_distribution`/`FST_distribution` del manifest de la sección 6 antes y después del fix, sobre el mismo lote de imágenes.

---

## 4. Qué se descartó y por qué

| Opción | Por qué se descarta (por ahora) |
|---|---|
| Más épocas de MoCo/CCAM | Los datos de las rondas r0→r1→r2 muestran que la cobertura se estabiliza en un techo (~0,13-0,15) que no se mueve con más auto-entrenamiento; no hay evidencia de que más épocas del backbone lo resuelva, y es la opción más cara (hay que re-correr todo el bootstrap desde cero). |
| Ajustar umbrales 0,35/0,65 | Son los valores reportados por el propio paper de USL-Net, no ajustados localmente; los mismos autores admiten que el método "relies on selecting an optimal saliency threshold... which makes it less generalized" — tocarlos sin medir antes es apostar a ciegas sobre algo que el paper original ya identificó como punto débil genérico del método, no específico de este dataset. |
| Modelo generativo de remoción de pelo (GAN/VAE/difusión) | Desproporcionado: requiere entrenar y mantener un modelo adicional con su propio dataset de máscaras de pelo, para resolver un problema de preprocesamiento dentro de un fix de ruido, no una contribución de remoción de pelo. |
| Cambiar de arquitectura (DINO como segmentador, o DINO-teacher → U-Net-student) | Escalada acordada con el profesor solo si, después de estos fixes, el U-Net corregido sigue rindiendo mal. Se prueba lo barato antes de lo caro. Ruta detallada y validada con literatura en la sección 5. |

---

## 5. Plan de escalamiento (Plan B) — cambio de arquitectura con DINO

Esto **no se activa ahora**. Es la ruta a seguir solo si, después de aplicar los fixes de las secciones 2 y 3 y remedir, el U-Net corregido sigue rindiendo mal. Se deja documentada de antemano, con implementación e evidencia de literatura, para no tener que improvisarla si llega el momento — y porque el proyecto ya usa DINOv2 como backbone del clasificador, así que hay infraestructura y familiaridad previas con el modelo.

Hay dos variantes distintas, no una sola "opción DINO". Se documentan ambas porque resuelven objetivos distintos.

### 5.1 Opción A — DINO/DINOv2/DINOv3 como segmentador directo

**Qué es:** usar el backbone DINO congelado como extractor de features denso, y entrenar (o ni siquiera entrenar) una cabeza liviana de decodificación encima, en vez de generar pseudo-máscaras vía CCAM + incertidumbre como hoy.

**Validación en literatura (2025-2026, todos posteriores al paper de USL-Net que usa el proyecto):**
- **Dino U-Net** — "Exploiting High-Fidelity Dense Features from Foundation Models for Medical Image Segmentation" (arXiv:2508.20909, 2025). Usa las features densas de un backbone tipo DINO congelado, con una cabeza de segmentación liviana encima, específicamente para imagen médica.
- **SegDINO** — "Introducing Multi-Scale Structure into DINO for Efficient Medical Image Segmentation" (arXiv:2606.17972, 2025). Mismo principio, con una estructura multi-escala agregada sobre las features de DINO para mejorar la segmentación eficiente en dominio médico.
- **INSID3** — "Training-Free In-Context Segmentation with DINOv3" (arXiv:2603.28480, 2025-2026). Variante más agresiva: segmentación **sin entrenar nada**, usando las features de DINOv3 directamente por similitud/contexto. Sería la opción más barata de probar primero dentro de esta ruta, antes de entrenar una cabeza de decodificación.
- **Dinov2_Mask R-CNN** — segmentación de instancias self-supervised con DINOv2 aplicada a úlceras de pie diabético (dominio dermatológico cercano, mismo tipo de imagen clínica de piel).
- El propio paper de USL-Net (Li et al. 2024, sección 2) ya reconoce el linaje: cita trabajo de "emerging properties in self-supervised vision transformers" (la familia DINO) como antecedente directo de su propio enfoque de features sin supervisión.

**Cómo se implementaría:** reemplazar la etapa `contrastive(moco) → ccam` del pipeline actual (`entrenamiento/pipeline_stages.py`) por: (1) extracción de features densas con DINOv2/DINOv3 congelado (mismo checkpoint que ya usa el clasificador, si la resolución/patch size son compatibles), (2) entrenar una cabeza de decodificación liviana (unas pocas capas convolucionales o un decoder tipo U-Net pequeño) sobre esas features, usando la misma lógica de pseudo-etiquetado por incertidumbre que ya existe en `uncertainty_module.py` — se reutiliza el módulo de incertidumbre, cambia lo que lo alimenta. Antes de eso, probar la variante training-free (INSID3-style) como diagnóstico rápido y barato: si ya da mejor saliencia que CCAM sin entrenar nada, la inversión en entrenar la cabeza está justificada.

**Trade-off:** DINOv2/v3 son backbones grandes (aunque congelados, no se reentrenan) — el costo real está en el cómputo de inferencia para extraer features sobre todo el dataset, no en entrenamiento. Es más simple de implementar que la opción B porque no hay destilación, solo una cabeza nueva.

### 5.2 Opción B — DINO-teacher → U-Net-student (destilación de conocimiento)

**Qué es:** usar un modelo DINO grande y congelado como "maestro" que nunca se despliega, y transferir su conocimiento a un U-Net pequeño ("red de segmentación entrenada con pseudo-máscaras") que sí se usa en producción/inferencia — el U-Net final queda liviano, pero entrenado con supervisión de un modelo mucho más capaz.

**Validación en literatura:**
- **Foundational Knowledge Distillation (FKD)** — trabajo muy reciente (2025-2026) que usa **DINOv3 congelado como "maestro semántico externo no sesgado"** para destilar representaciones robustas hacia un red de segmentación entrenada con pseudo-máscaras de segmentación. El argumento central de los autores es exactamente el problema que tenemos: las representaciones de DINO no están atadas a ningún dataset ni distribución de clases particular, así que evitan que el red de segmentación entrenada con pseudo-máscaras herede los sesgos del propio dataset de entrenamiento (relevante para nuestro caso, donde el generador de pseudo-etiquetas hoy hereda el sesgo de subcobertura de su propia saliencia).
- **Validación general del paradigma maestro-red de segmentación entrenada con pseudo-máscaras en segmentación de piel específicamente:** Dzieniszewska et al. (*Cancers*, 2024) — "Improving Skin Lesion Segmentation with Self-Training" — implementaron un ciclo Noisy Student de 4 pasos (maestro en datos etiquetados → pseudo-etiquetas → red de segmentación entrenada con pseudo-máscaras con datos reales+pseudo → segundo red de segmentación entrenada con pseudo-máscaras) con DeepLabV3 como maestro y red de segmentación entrenada con pseudo-máscaras, logrando mIoU 88,0 % en ISIC2018 y 87,54 % en PH2. No usa DINO, pero valida que el ciclo maestro-red de segmentación entrenada con pseudo-máscaras funciona bien en este dominio exacto.
- **Validación general en imagen médica (no específico de piel):** KD multi-maestro para segmentación de vasos retinianos (2025) mejoró el U-Net base hasta +8,44 puntos de F1 y +10,42 de IoU; MSKD (destilación estructurada) y destilación de forma-intensidad (2024) muestran que un U-Net red de segmentación entrenada con pseudo-máscaras liviano puede absorber conocimiento de un maestro más fuerte sin perder eficiencia de despliegue.

**Cómo se implementaría:** el DINO congelado (maestro) genera las pseudo-máscaras/features objetivo (reemplazando o complementando la saliencia de CCAM); el U-Net red de segmentación entrenada con pseudo-máscaras actual (`comun/estudiante/student_engine.py`) se mantiene como arquitectura de despliegue, pero su señal de entrenamiento pasa a venir del maestro DINO en vez de (o además de) las rondas de auto-entrenamiento actuales. La lógica de incertidumbre/exclusión de `uncertainty_module.py` se puede conservar tal cual, aplicada sobre la salida del maestro en vez de sobre CCAM.

**Trade-off vs. Opción A:** más complejo de implementar (hay un paso de destilación real, no solo una cabeza nueva), pero el resultado final es un modelo pequeño y rápido para desplegar — relevante si el prototipo clínico necesita corres en hardware limitado. La Opción A es más simple pero el modelo final es más pesado (aunque el backbone esté congelado, sigue siendo parte del pipeline de inferencia salvo que se destile después de todos modos).

### 5.3 Criterio de activación y cuál elegir

**Cuándo pasar a esta sección:** solo después de medir el efecto de los fixes de las secciones 2 y 3 (multi-región + remoción de pelo) contra la tabla base de la sección 2.1. Si `mean_foreground_fraction` y `excluded_no_region` no se acercan de forma material a la referencia (~0,21-0,27 de cobertura), es la señal para escalar.

**Cuál elegir primero:** Opción A (DINO como segmentador), empezando por la variante training-free (INSID3-style) como diagnóstico de una tarde antes de comprometerse a entrenar nada. Motivo: el proyecto ya tiene DINOv2 integrado para el clasificador, así que la curva de aprendizaje/infraestructura ya existe; la Opción B (destilación) queda como segundo paso solo si el objetivo final requiere explícitamente un modelo liviano para despliegue clínico, algo que hoy no está confirmado como requisito.

---

## 6. Estimación ITA no validada (no es FST) — qué tenemos, qué cambia con los fixes, y qué sigue roto

> **Nota: estimación ITA no validada (no es FST).** Lo que aquí aparece como «FST», «fototipo» o «bucket Fitzpatrick» por imagen es el ITA con los umbrales fijos de Groh (`src/dermssl/fitzpatrick.py`), y no está validado. Contra FST asignado por expertos acierta 28,5–30,8 % frente a 32,9 % de la baseline trivial en Fitzpatrick17k-C (11.098 imágenes, docs/20-validacion-externa-fst-fitzpatrick17k.md); en PAD-UFES-20, el ITA con umbrales fijos acierta 22,9 % frente a 51,3 % de la baseline de mayoría (Alipour et al., *Skin Research and Technology* 2026, PMC13459198; FST asignado por dermatólogo según ese artículo). Los terciles `tone_tercile` son terciles de luminancia (mediana de L\* de la piel), no tono de piel.

Se revisó el código real (`plan-segmentacion/comun/evaluacion/fst_pseudo_manifest.py` y `src/dermssl/fitzpatrick.py`) antes de escribir esta sección, específicamente para verificar si los fixes de las secciones 2-3 ya resuelven esto de por sí. **No lo resuelven del todo** — hay que ajustar el fix #2 y queda un hueco real sin cubrir.

### 6.1 Qué tenemos hoy

No existe ningún clasificador de red neuronal para Fitzpatrick en el proyecto. La etapa `fst` del pipeline (`entrenamiento/pipeline_stages.py:240-254`, script `fst_pseudo_manifest.py`) calcula ITA por procesamiento de imagen — CIELAB L*/b*, fórmula `ITA = arctan((L*-50)/b*) * 180/π` — usando como "piel" los píxeles que la propia pseudo-máscara del pipeline (la del último round de auto-entrenamiento) deja fuera de la lesión y fuera del borde/campo de visión: literalmente `skin = ~mask & ~border` (`fitzpatrick.py:312,323`). Es decir: la estimación de tono depende directamente de la calidad de la misma pseudo-máscara que se diagnosticó en las secciones 2-3.

Ya existe, además, un detector parcial de artefactos dentro de `fitzpatrick.py` (`_linear_artifact_present` o equivalente, líneas ~195-225): busca estructuras oscuras y de alto contraste (proxy de pelo/tinta) dentro de la región candidata a piel. Si detecta una fracción material, marca `artifact_detected=True` y multiplica la `confidence` del registro por 0,70 (`ARTIFACT_CONFIDENCE_FACTOR`, línea 309). **Pero no excluye esos píxeles del cálculo de L*/b*** — solo baja un número de confianza que, además, `fst_pseudo_manifest.py` no usa para filtrar las estadísticas agregadas (`_dataset_summary` no filtra por `confidence`). La señal existe, está calculada, pero nadie la lee todavía.

### 6.2 Qué cambia con los fixes de las secciones 2-3, y qué NO cambia

| Fix | Efecto real sobre la estimación ITA, verificado en código |
|---|---|
| Fix #1 (unión de regiones en `select_central_region`) | **No reduce la contaminación de ITA.** Los fragmentos de lesión que hoy se descartan quedan marcados como `255` (ignorar), y `border = pseudo == 255 \| ~field` ya los excluye de la muestra de "piel" — no se están contando como piel hoy por esta vía. Lo que sí mejora: menos imágenes caen en cuarentena por "ninguna región encontrada" (`excluded_no_region`), así que más imágenes van a tener un bucket de la estimación ITA asignado — mejora la **completitud** de la muestra, no la exactitud de cada estimación individual. |
| Fix #2 (remoción de pelo) | **Sí ayuda, pero solo si se cablea en la etapa `cache`** (ver sección 3.4, nota agregada) para que `fst_pseudo_manifest.py` lea la imagen ya limpia — hoy esa etapa carga el archivo fuente original de forma independiente, así que el fix, insertado solo antes de `contrastive(moco)`, no llega ahí. |

### 6.3 Lo que sigue sin resolver — y es el argumento real para el clasificador dedicado

`skin = ~mask & ~border` cuenta como piel **cualquier píxel que la pseudo-máscara no marque como lesión**, incluyendo los que nunca llegaron a ser candidatos de alta saliencia en CCAM — el caso típico de lesiones pálidas o de borde de bajo contraste, que ni siquiera entran al filtro de región de la sección 2. Ninguno de los fixes #1 o #2 toca esto: no es un problema de selección de región ni de artefactos de pelo, es que la saliencia de origen nunca marcó esos píxeles como sospechosos. Y es exactamente el patrón que Bencevic et al. (2024) documenta como el punto donde más falla la segmentación en piel oscura — bajo contraste, bordes graduales, zonas despigmentadas. Hay una hipótesis razonable, sin confirmar todavía, de que el ITA de este proyecto está sesgado más fuerte justo en la población que más importa auditar, por esta vía específica.

### 6.4 Recomendación, en dos pasos

**Paso 1 — barato, hacer ya junto con los fixes de segmentación:**
1. Cablear el fix #2 (remoción de pelo) en la etapa `cache`, no como paso aislado (ver nota en sección 3.4).
2. Hacer que `fst_pseudo_manifest.py` filtre o al menos reporte por separado los registros con `artifact_detected=True` / `confidence < 1.0` en `_dataset_summary` — la señal ya está calculada, solo falta usarla.
3. Después de aplicar 1 y 2, volver a generar el manifest de la estimación ITA no validada (no es FST) y comparar `tone_distribution`/`FST_distribution` contra la corrida actual. Esto responde, con datos y no con supuestos, cuánto de la sección 6.3 se explica por pelo/artefactos (que ya se corrige) contra cuánto queda por el hueco de saliencia (que no se corrige con nada de esto).

**Paso 2 — condicional, decisión de alcance separada, no se activa todavía:** si después del Paso 1 la distribución de tono sigue viéndose sospechosa (por ejemplo, muy pocas imágenes en el tercio oscuro comparado con lo esperado, o mucha diferencia entre ITA y el agrupamiento manual si se llega a hacer uno), ahí se justifica construir el clasificador dedicado de Fitzpatrick validado por Bencevic et al. (2024) y Groh et al.: VGG16 preentrenado en Fitzpatrick-17k (dataset que el proyecto ya usa en `docs/20`), afinado en datos de lesiones. Su ventaja real no es que sea "más preciso" en abstracto — es que **no depende de la pseudo-máscara de lesión en absoluto**, así que esquiva por completo tanto el hueco de la sección 6.3 como cualquier bug futuro del segmentador. Importante: el propio paper prueba que usar este clasificador para *mitigar* el sesgo (muestreo estratificado) no funciona — sirve como herramienta de medición/auditoría confiable, no como solución al sesgo de segmentación por sí sola.

---

## 7. Clasificador — ruta recomendada

Este apartado documenta la ruta de decisión ya discutida, para que Nico la use como base del diagnóstico del clasificador. No es una tarea de esta línea de trabajo (segmentación), pero queda registrada aquí para que ambas líneas compartan la misma evidencia.

### 5.1 Problema

`Proposed` (preentrenamiento SSL con transformación de tono usando pseudo-máscaras) pierde macro-F1/balanced accuracy contra `Plain` (sin transformación de tono), con intervalos de confianza que excluyen cero:

| Backbone | macro-F1 Proposed | macro-F1 Plain | Δ | IC 95% |
|---|---:|---:|---:|---|
| ViT-S | 0,4840 | 0,5307 | −0,0467 | [−0,0878; −0,0036] |
| ViT-B | 0,5237 | 0,5856 | −0,0619 | [−0,1131; −0,0109] |

### 5.2 Causa — NO demostrada como fuga de máscara

Auditoría por imagen (docs/25, sección 7): la fuga de máscara hacia la vista de tono es real (Dice 0,7826 pseudo vs referencia, 21,54 % de falsos negativos de lesión, 84,7 % de imágenes con alguna alteración), **pero no correlaciona con la caída del clasificador**: Spearman rho entre fuga y Δlog-loss (Proposed−Plain) = 0,00 (p=0,9986), patrón no monotónico por cuartil de fuga.

Confusores reales, sin resolver, identificados en el mismo diagnóstico:
- `Proposed` procesó 2,8–3,3 % menos muestras que `Plain` bajo el mismo presupuesto de 12h-GPU — no es una comparación emparejada.
- Ambos brazos (`Proposed` y `Plain`) rinden por debajo de DINOv2 público en macro-F1/BA — hay una degradación de adaptación SSL que no depende de la máscara.
- `run.json` no registra `code_revision` en ninguno de los dos brazos, y `split_hash` queda en cero para `Plain` — no se puede probar que el código fue idéntico entre brazos.
- La validación es *label-disjoint* pero *image-transductive*: 1.960 de los 2.004 IDs de validación también están en el pool de preentrenamiento SSL de `Proposed`.

### 5.3 Ruta de decisión recomendada

**Ablación emparejada mínima, no las 3-4 ramas completas:** correr solo `Plain` vs `Proposed-oracle-mask` (máscara de referencia real, solo en partición diagnóstica train/dev, nunca en holdout/test), con:
- Mismas IDs, mismo split, mismo orden/semillas.
- Mismo número de *optimizer steps* e imágenes vistas (arreglar primero el desbalance de 2,8–3,3 % antes de lanzar).
- Mínimo 2-3 semillas por brazo (con 1 sola corrida no se puede distinguir señal de ruido de entrenamiento).
- Registrar `code_revision` y `split_hash` correctamente en `run.json` para ambos brazos.
- Excluir los IDs de validación del pool de preentrenamiento SSL en esta corrida nueva, para que la evaluación deje de ser *image-transductive*.

**Criterio de interpretación, a definir antes de mirar los resultados:**

| Resultado | Lectura |
|---|---|
| Oracle ≈ Plain | La calidad de máscara es la causa dominante — vale la pena invertir en mejorar la segmentación para el clasificador también. |
| Oracle ≈ Pseudo (ambos < Plain) | La máscara es inocente; el problema es el desplazamiento de tono o la estrategia de adaptación SSL — no perseguir más la calidad de máscara para este síntoma. |
| Resultado intermedio | Definir de antemano qué diferencia mínima se considera concluyente, para no debatir el resultado después de verlo. |

### 5.4 Por qué esta ruta y no otra

Se descartó (por ahora) correr las 3-4 ramas completas de la Fase B (`Plain`/`Proposed-pseudo`/`Proposed-oracle`/`Proposed-filtered` × 3 semillas) porque el par `Plain` vs `Oracle` ya es suficiente para responder la pregunta binaria que importa primero (¿la máscara es o no la causa?); el resto de las ramas solo aportan si esa primera pregunta sale ambigua.

---

## 8. Resumen de rutas recomendadas

| Línea | Qué se corrige | Ruta recomendada | Qué se evita |
|---|---|---|---|
| Segmentador | Subcobertura sistemática (`select_central_region`) | Unión de regiones que pasan el filtro de área/centralidad existente, en vez de tomar solo la más central | Reintroducir ruido de sombras de borde; reentrenar CCAM/MoCo sin necesidad |
| Segmentador | Pelo/tinta/brillos no cubiertos | Detección morfológica (black-hat) + inpainting (TELEA) sobre la imagen, antes de MoCo/CCAM | Entrenar un modelo generativo de remoción de pelo propio (GAN/VAE/difusión) |
| Clasificador | Caída Proposed vs Plain, causa no demostrada | Ablación emparejada mínima Plain vs Proposed-oracle-mask, steps igualados, IDs de validación excluidos del pool SSL | Correr las 4 ramas completas de la Fase B antes de saber si es necesario |
| Segmentador (Plan B, condicional) | Los fixes de las secciones 2-3 no alcanzan | Opción A primero: DINO/DINOv2/DINOv3 como segmentador, empezando por la variante training-free (INSID3-style); Opción B (destilación DINO-teacher → U-Net-student) solo si se necesita un modelo liviano para despliegue clínico — ver sección 5 | Comprometerse a reescribir la arquitectura sin medir antes el efecto de los fixes baratos |
| Estimación ITA no validada (no es FST) | ITA depende de la pseudo-máscara; pelo no llega limpio a esa etapa; píxeles de lesión de bajo contraste nunca detectados se cuentan como piel | Cablear el fix #2 en la etapa `cache` (no aislado), usar la `confidence` que ya se calcula y hoy se ignora, remedir `tone_distribution` — clasificador dedicado (Bencevic/Groh) solo si después de eso sigue viéndose sesgado — ver sección 6 | Asumir que los fixes de segmentación arreglan el ITA automáticamente sin haber verificado el código de `fst_pseudo_manifest.py` primero |

---

## 9. Referencias

[1] X. Li, B. Peng, J. Hu, C. Ma, D. Yang, Z. Xie. "USL-Net: Uncertainty Self-Learning Network for Unsupervised Skin Lesion Segmentation." Preprint submitted to Elsevier, 2024. (`bibliografia/Importantes/Li2024_USL-Net.pdf`)

[2] A.L. Sharma et al. "Skin lesion segmentation: a comprehensive review." *Computers in Biology and Medicine*, 196 (2025) 110842. (`bibliografia/Importantes/Sharma2025_segmentation_review.pdf`)

[3] J. Ramya, H. Vijaylakshmi, H.M. Saifuddin. "Segmentation of skin lesion images..." (2023) 109524. Citado como [21] en Li et al. 2024 y como [52] en Sharma et al. 2025.

[4] W. Chen, Q. Mu, J. Qi. "TrUNet: Dual-branch network by fusing CNN and..." (2024). Citado como [109] en Sharma et al. 2025.

[5] "A Review of Automatic Hair Removal in Dermoscopy Images: From Image Processing to Deep Learning." *Journal of Imaging Informatics in Medicine*, Springer Nature. https://link.springer.com/article/10.1007/s10278-026-01950-y

[6] "Hair and artifact removal in dermoscopic images using deep learning for enhanced skin cancer detection." PMC. https://pmc.ncbi.nlm.nih.gov/articles/PMC13149812/

[7] "Hair removal and lesion segmentation of dermoscopic images for classification of skin cancer using deep neural networks." *ScienceDirect*, 2025. https://www.sciencedirect.com/science/article/pii/S1110866525002373

[8] "A deep learning framework for automated early diagnosis and classification of skin cancer lesions in dermoscopy images." *Scientific Reports*, 2025. https://www.nature.com/articles/s41598-025-15655-9

[9] "Integrated convolutional neural network for skin cancer classification with hair and noise restoration." PMC, 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC11913500/

[10] `docs/25-diagnostico-mascaras-clasificador.md` — diagnóstico interno del proyecto (auditoría de fuga de máscaras y su relación con la caída del clasificador), 2026-09-25.

[11] "Dino U-Net: Exploiting High-Fidelity Dense Features from Foundation Models for Medical Image Segmentation." arXiv:2508.20909 (2025). https://arxiv.org/pdf/2508.20909

[12] "SegDINO: Introducing Multi-Scale Structure into DINO for Efficient Medical Image Segmentation." arXiv:2606.17972 (2025). https://arxiv.org/pdf/2606.17972

[13] "INSID3: Training-Free In-Context Segmentation with DINOv3." arXiv:2603.28480. https://arxiv.org/pdf/2603.28480

[14] Foundational Knowledge Distillation (FKD) — DINOv3 como maestro semántico externo congelado para destilación en segmentación (2025-2026). https://arxiv.org/html/2602.07819v1

[15] A. Dzieniszewska, P. Garbat, R. Piramidowicz. "Improving Skin Lesion Segmentation with Self-Training." *Cancers*, 16 (2024) 1120. https://doi.org/10.3390/cancers16061120 (`bibliografia/Dzieniszewska2024_self_training_segmentation.pdf`)

[16] "Multi-teacher based knowledge distillation for retinal vessel segmentation." PubMed, 2025. https://pubmed.ncbi.nlm.nih.gov/40548065/

[17] "Dinov2_Mask R-CNN: Self-supervised Instance Segmentation of Diabetic Foot Ulcers." Diabetic Foot Ulcers Grand Challenge. https://dl.acm.org/doi/10.1007/978-3-031-80871-5_2

[18] M. Bencevic, M. Habijan, I. Galic, D. Babin, A. Pizurica. "Understanding skin color bias in deep learning-based skin lesion segmentation." *Computer Methods and Programs in Biomedicine*, 245 (2024) 108044. (`bibliografia/Importantes/Bencevic2024_skin_color_segmentation_bias.pdf`)
