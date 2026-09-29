# 30. PAD-UFES-20: prueba externa sin entrenar de las finales v7

**Registrado el 2026-09-28, antes de bajar el dato y antes de escribir el código que lo lee.** Amplía docs/29 §2.6 sin cambiar nada de lo que ahí se registró. `scripts/pad_ufes_eval.py` se niega a correr si este registro no está en un commit ancestro de HEAD o si el bloque registrado cambió desde ese commit.

## 1. Licencia y fuente (verificada el 2026-09-28)

- **Dataset:** Pacheco et al., *PAD-UFES-20: a skin lesion dataset composed of patient data and clinical images collected from smartphones*, Mendeley Data, v1, DOI [10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1). Leído de la API pública de Mendeley (`data.mendeley.com/public-api/datasets/zr7vgbcyr2`).
- **Licencia del dataset:** Creative Commons Attribution 4.0 International (CC BY 4.0), <http://creativecommons.org/licenses/by/4.0>. Permite compartir, copiar y modificar, también con fines comerciales, con tres condiciones: dar crédito, enlazar la licencia e indicar si hubo cambios, sin sugerir que los autores respaldan este uso.
- **Artículo:** Pacheco et al., *Data in Brief* 32 (2020) 106221, doi [10.1016/j.dib.2020.106221](https://doi.org/10.1016/j.dib.2020.106221), también CC BY 4.0 según Crossref (versión publicada). Preprint arXiv:2007.00478.
- **Qué permite para este trabajo:** evaluar los modelos sobre el dataset y publicar las métricas derivadas en la tesis y en el repositorio está permitido, con atribución. Las imágenes no se suben al repositorio: no hace falta, y así la atribución queda en un solo lugar (esta sección y la tesis).
- **Atribución obligatoria** en la tesis y en cualquier tabla publicada: citar el dataset (DOI de Mendeley) y el artículo de *Data in Brief*, nombrar la licencia CC BY 4.0 con su enlace y declarar los cambios: el mapeo de etiquetas de §2.2, la exclusión de SEK y el preprocesado del C1.
- **Ética (del registro del dataset):** aprobado por el comité de ética de la UFES (nº 500002/478) y por Plataforma Brasil (nº 4.007.097), con consentimiento de los pacientes.

## 2. Protocolo

<!-- registro-pad:inicio -->

### 2.1 Datos y controles antes de usarlos

- Cuatro archivos oficiales, con el tamaño y el sha256 que publica Mendeley: `imgs_part_1.zip`, `imgs_part_2.zip`, `imgs_part_3.zip` y `metadata.csv`, 3.592.712.972 bytes en total (~3,6 GB). `scripts/pad_ufes_download.py` los baja en el pod y rechaza cualquier archivo cuyo tamaño o sha256 no coincida.
- El manifiesto (`scripts/pad_ufes_manifest.py`) verifica, antes de cualquier extracción de features: 2.298 imágenes, 1.373 pacientes y 1.641 lesiones; los conteos por diagnóstico del artículo (ACK 730, BCC 845, MEL 52, NEV 244, SCC 192, SEK 235); `patient_id`, `lesion_id` e `img_id` sin vacíos; cada `img_id` resuelto a un solo archivo PNG, y ninguna lesión repartida entre dos pacientes. Guarda el sha256 de cada archivo. Si algo no coincide, aborta.

### 2.2 Mapeo de etiquetas PAD → HAM10000

| PAD | HAM10000 | Lectura |
|---|---|---|
| BCC | bcc | primaria y secundaria |
| MEL | mel | primaria y secundaria |
| NEV | nv | primaria y secundaria |
| ACK | akiec | secundaria |
| SCC (incluye Bowen) | akiec | secundaria |
| SEK | bkl | exploratoria, sin veredicto |

- **Primaria:** bcc, mel y nv (1.141 imágenes). **Secundaria:** además akiec = ACK + SCC (2.063 imágenes). Es la intersección registrada en docs/29 §2.6.
- **SCC → akiec es un mapeo parcial y se declara.** En HAM, akiec son queratosis actínicas y carcinoma intraepidérmico (Bowen). En PAD, SCC junta Bowen con carcinoma escamoso invasor, que no tiene clase en HAM. Por eso akiec va en la secundaria y no en la primaria.
- **Por qué SEK queda fuera de las lecturas con veredicto.** SEK tiene un análogo en HAM, bkl, pero bkl es una clase heterogénea: mezcla queratosis seborreica, léntigo solar y queratosis tipo liquen plano, y el probe aprende esa mezcla. Además, solo el 6,4 % de las SEK de PAD tiene biopsia, contra el 100 % de BCC, SCC y MEL. Por último, docs/29 §2.6 registró la intersección sin SEK, y sumarla ahora como lectura con veredicto sería cambiar un registro. Va como lectura **exploratoria** (cinco clases: akiec, bcc, bkl, mel, nv; las 2.298 imágenes), que se reporta sin veredicto.
- df y vasc no tienen análogo en PAD y no se evalúan.

### 2.3 Modelos y probe (el mismo protocolo del C1/v7)

- **Backbones:** DINOv2 ViT-B/14 público y las finales v7 de docs/28 §5.1: A\* y B\* con las semillas 42 y 43. No se elige ninguna semilla.
- **Probe:** `logreg` entrenado sobre el train COMPLETO del C1 de HAM10000 (8.011 ids, fingerprint `d347edd52d0d7648`, las siete clases), con la misma extracción: 224 px, resize a 255 y crop central, normalización ImageNet, token CLS. No se ve ninguna imagen de PAD en el entrenamiento, no se ajusta ningún umbral y no hay TTA. La extracción y el probe son los de `scripts/c1_final_eval.py`, no una segunda implementación.
- **Del probe a cada lectura:** de las siete probabilidades de HAM se toman las columnas de las clases de la lectura y se renormalizan para que sumen 1. Es la probabilidad condicionada a que la imagen sea de una de esas clases. Como dato descriptivo se reporta también la fracción de imágenes cuyo argmax sobre las siete clases cae fuera de la lectura.

### 2.4 Métricas y bootstrap

- **Métricas:** macro-F1 (decide), balanced accuracy, AUC one-vs-rest macro y ECE (15 bins), sobre las probabilidades renormalizadas.
- **Contrastes, por semilla:** A\* − público, B\* − público y B\* − A\*.
- **Bootstrap pareado, agrupado por paciente:** el de todo el proyecto (`paired_bootstrap`: semilla 20260923, 1.000 remuestras, percentil 95 %, las mismas remuestras para los dos brazos), con la unidad de remuestreo en el paciente y no en la imagen. PAD tiene varias imágenes por lesión y varias lesiones por paciente, y remuestrear imágenes angostaría los intervalos con dependencia que no es información nueva (docs/29 §2.6 ya lo registró agrupado por paciente).
- **Regla de lectura:** se escribe "mejora" solo si el IC pareado excluye el cero a favor del brazo, y "robusto a la semilla" solo si eso pasa con las dos semillas (la regla de docs/28 §4.1). Un IC que toca el cero es "indistinguible", se incline hacia donde se incline el punto.
- **Limitación declarada desde ya:** MEL tiene 52 imágenes. El F1 de mel, y con él el macro-F1 primario, va a tener intervalos anchos.

### 2.5 Estratificación por fototipo (exploratoria, pedido de Paula del 2026-09-28)

- **La columna existe.** El artículo (sección *Meta-data*) documenta `fitspatrick`, escrito así, con la errata, como "un entero que representa el fototipo de Fitzpatrick". El registro de Mendeley la lista entre las variables clínicas. No es una de las columnas que el artículo garantiza siempre presentes (`patient_id`, `lesion_id`, `img_id`, `age`, `region` y `biopsed`); las demás dependen de lo que respondió el paciente y quedan vacías si falta. Así que se esperan vacíos.
- **Bins: I, II y III-VI agrupado, más "sin FST" para los vacíos.** El artículo no publica la distribución de fototipos. Lo que sí documenta, al describir `background_father` y `background_mother`, es que muchos pacientes descienden de Pomerania (Europa central). Por eso se esperan sobre todo pieles claras, y III-VI se agrupa para que el bin tenga tamaño. Los bins se fijan acá, sin mirar el dato, y no se reajustan después. Un valor fuera de 1-6 cuenta como vacío y el manifiesto reporta cuántos hubo.
- **Qué se reporta:** la lectura primaria (bcc, mel y nv) por bin. Para cada bin: n de imágenes y de pacientes, las cuatro métricas por brazo con su IC y los contrastes pareados de §2.4. El bootstrap se hace dentro de cada bin, agrupado por paciente, con la misma semilla y las mismas remuestras. Si un bin tiene n < 100 imágenes, se reporta el n y los números con el rótulo "sin potencia", sin veredicto.
- **Es exploratorio.** No tiene predicción ni veredicto y no cambia las lecturas de §3. Además, el fototipo lo anotó el clínico en la consulta, no es una medición, y en PAD no se estima el tono desde la imagen como en el FST del proyecto (docs/20).

### 2.6 Qué no se hace

No hay fine-tune ni reentrenamiento del probe en PAD, ni ajuste de umbrales, de preprocesado o de mapeo después de ver el dato. Se evalúa una vez, y el script se niega a sobrescribir su salida. Si algo del protocolo resulta imposible con el dato real (por ejemplo, conteos que no coinciden), se para, se documenta y se registra el cambio antes de correr.

## 3. Predicción registrada (2026-09-28)

- **PAD-P1 (primaria, decide):** en la lectura primaria, la ventaja en macro-F1 de B\* sobre el público en HAM (+0,045 y +0,048 en val) **no se transfiere**: el Δ puntual de macro-F1 B\* − público es ≤ 0 con las dos semillas. Probabilidad estimada antes del dato: ≈60 %.
- **PAD-P2 (primaria):** lo mismo para A\*: Δ puntual de macro-F1 A\* − público ≤ 0 con las dos semillas. ≈60 %.
- **PAD-P3 (secundaria):** la ventaja en calibración sí se mantiene en signo: ECE B\* − público < 0 con las dos semillas, aunque más chica que en HAM (−0,10). ≈55 %.
- **PAD-P4 (secundaria):** B\* − A\* en macro-F1 es indistinguible (IC que cruza el cero) con las dos semillas, como en HAM. ≈75 %.
- **Por qué esa dirección:** las finales se adaptaron solo con dermatoscopía de HAM, y PAD son fotos clínicas de smartphone, sin dermatoscopio y con otra iluminación, otra escala y otra piel alrededor. S-diag (docs/28 §5.3) ya mostró que la adaptación degrada las features densas. El DINOv2 público se preentrenó con fotos naturales, que se parecen más a PAD que la dermatoscopía. Lo esperable es que la especialización que ayuda en HAM no ayude fuera de dominio. La ventaja en ECE viene sobre todo de que el público está mal calibrado ya en HAM, y eso no debería cambiar de signo.
- **Qué la refuta:** si B\* − público tiene un IC del macro-F1 por encima de cero con las dos semillas en la lectura primaria, la adaptación SSL generaliza fuera de dominio. Se reporta así, como resultado a favor, aunque contradiga la predicción.
- Se reporta el resultado sea cual sea, junto con las lecturas secundaria y exploratoria, y el resultado primario v7 de docs/28 §5.1 no cambia.

<!-- registro-pad:fin -->

## 4. Resultado

Pendiente. Se completa después de la única evaluación, desde el JSON que genera `scripts/pad_ufes_eval.py`.

## 5. Validación del estimador FST contra el fototipo de PAD (exploratoria, pedido T1 de Paula)

**Registrado el 2026-09-28, antes de escribir el código y de computar nada.** Es un bloque aparte del de §2-3 y tiene su propio guard: `scripts/pad_ufes_fst_validation.py` se niega a correr si este bloque no está en un commit anterior al script o si cambió desde entonces.

<!-- registro-pad-fst:inicio -->

- **Qué se valida:** el estimador de fototipo del proyecto, ITA con los umbrales fijos de Groh (`src/dermssl/fitzpatrick.py`, `FST_THRESHOLDS`), contra la columna `fitspatrick` de PAD (1 a 6). Se usan las dos condiciones de docs/20: la imagen completa (`estimate_fst_whole_image`) y el fondo por Otsu (`estimate_fst` sin máscara, que cae al fallback Otsu). No se aplica constancia de color (`identity`, la corrida principal de docs/20). El cálculo por imagen y las métricas base son los de `scripts/validar_fitzpatrick17k.py` (`evaluate_image`, `accuracy_metrics`), no una segunda implementación.
- **Qué es la referencia.** El artículo de PAD no dice quién asigna el fototipo: solo que las columnas que no son `patient_id`, `lesion_id`, `img_id`, `age`, `region` y `biopsed` dependen de lo que se relevó en la consulta. Alipour et al. 2026 la tratan como asignada por dermatólogo. En cualquier caso no es una medición del color de la piel, así que esto es un contraste externo y no una verdad clínica, igual que en docs/20.
- **Filas:** una por imagen. Las imágenes sin fototipo válido (vacío o fuera de 1-6, según `dermssl.pad_ufes.parse_fst`) quedan fuera y se cuentan. Las imágenes donde el estimador no llega a `status == "ok"` se cuentan en `n_unavailable` y quedan fuera de las métricas, con el mismo criterio que docs/20.
- **Métricas, por condición:** matriz de confusión, accuracy exacta, accuracy off-by-one y kappa de Cohen con pesos lineales. La baseline de mayoría es predecir siempre el fototipo más frecuente de las mismas filas evaluables; se reporta su accuracy exacta y su off-by-one. La accuracy exacta y el kappa llevan un IC 95 % bootstrap agrupado por paciente (semilla 20260923, 1.000 remuestras), que es descriptivo.
- **Predicción:** con las dos condiciones, la accuracy exacta del estimador **no supera** la baseline de mayoría en las mismas filas. Probabilidad estimada: ≈90 %. Antecedentes: en Fitzpatrick17k, 28,5-30,8 % contra 32,9 % de la baseline trivial (docs/20). En PAD-UFES-20, con parches de piel, 22,9 % contra 51,3 % de la baseline de mayoría (Alipour, Burke y Courtney, *Skin Research and Technology* 32(8):e70374, 2026, doi 10.1111/srt.70374, PMC13459198; cifras verificadas en el resumen).
- **Predicción secundaria:** el kappa lineal queda por debajo de 0,20 con las dos condiciones. ≈75 %.
- **Qué la refuta:** si en alguna condición la accuracy exacta supera la baseline, se reporta así. Aun en ese caso no se afirma que el estimador sea válido, porque la referencia no es una medición.
- **Es exploratorio.** No cambia las lecturas de §3 ni la estratificación de §2.5, que usa el `fitspatrick` de PAD tal cual y no el estimador.

<!-- registro-pad-fst:fin -->

## 6. Erratas y notas (fuera de los bloques registrados)

Los bloques de §2-3 y §5 no se editan: el guard los compara byte a byte contra su commit de registro. Lo que cambió después va acá.

- **Errata de §2.1, clave de lesión (2026-09-28).** Donde §2.1 dice "1.641 lesiones" y "ninguna lesión repartida entre dos pacientes", se asumió que `lesion_id` era global. No lo es: es un contador por paciente. Por ejemplo, la lesión 17 es una ACK de PAT_9 y una SEK de PAT_998. El manifiesto abortó en el pod por eso, con "250 lesiones con mas de un paciente". Se verificó sobre el `metadata.csv` oficial (sha256 `14d14523…`, igual al fijado): 2.298 imágenes, 1.373 pacientes, 1.641 valores distintos de `lesion_id` (que es lo que cuenta el artículo) y 1.891 lesiones reales si la clave es `(patient_id, lesion_id)`. Dentro de cada par, el diagnóstico es siempre el mismo (0 inconsistencias). Ahora el manifiesto usa esa clave compuesta: sigue exigiendo los 1.641 ids del artículo, aborta si un mismo par tiene dos diagnósticos y registra los dos conteos. El remuestreo sigue siendo por paciente, como está registrado, así que el análisis no cambia.
- **Nota sobre los grupos de fototipo (Paula, 2026-09-28).** El metadata tiene, por imagen: I 153, II 876, III 392, IV 62, V 10, VI 1 y 804 vacíos. Se reportan I, II y III-VI agrupado. El grupo III-VI es casi todo tipo III, así que no representa piel oscura. IV, V y VI van solo como conteo descriptivo, sin veredicto. La frase para la tesis es literal: "PAD-UFES-20 no permite concluir sobre FST V-VI". La nota con el porcentaje no se transcribe a mano: `dermssl.pad_ufes.fst_note` la calcula desde el metadata, y la escriben el manifiesto, el JSON de la evaluación y el de la validación FST, junto con la tabla `<salida>.fst_table.md`. Con los conteos de arriba sale "84 % tipo III (392/465)".
