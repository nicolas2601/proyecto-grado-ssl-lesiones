# Super prompt — figura de arquitectura profesional (Nano Banana Pro / Google Flow)

Uso: aistudio.google.com → modelo **gemini-3-pro-image-preview** (Nano Banana Pro)
→ adjuntar `assets/figuras/fig11_arquitectura_completa.png` como imagen de referencia
→ pegar el prompt de abajo. Aspect ratio 3:4 o 4:5, resolución máxima.

---

Redesign the attached technical diagram as a premium, publication-grade ML system
architecture figure, in the visual style of a Google DeepMind / NVIDIA technical
report: clean flat-vector design with subtle depth, soft drop shadows, rounded
cards with 1px borders, a cohesive muted palette (deep blue, teal green, amber,
coral red accents on an off-white background), crisp sans-serif typography with
clear hierarchy, minimalist line icons inside each card (database, image stack,
neural network, snowflake for frozen, chart for metrics, GPU chip), and smooth
curved connector arrows with clear direction. Add a subtle dotted grid background.

CRITICAL RULES:
- Reproduce ALL text from the reference image EXACTLY, character by character,
  in Spanish. Do not translate, paraphrase, abbreviate, or invent any text.
- Keep the exact same structure: 3 numbered horizontal bands stacked vertically.
- Keep every box, every arrow and every connection from the reference.
- The red dashed arrow labeled "etiquetas SOLO en evaluación" must go from the
  "HAM10000 etiquetado" card down the right margin into the "Métricas" card,
  without crossing any other card.
- The purple dashed arrow labeled "EMA" goes from "modelo de segmentación ViT-S/14" down to
  "Profesor (EMA)".
- No typos. No extra decorative text. No watermarks. No photorealistic elements.

Band 1 — "1 · DATOS" (blue accent): three cards:
"Pool SSL — SIN etiquetas" (~48-70k imágenes / HAM10000 (train) + ISIC 2020 /
dermatoscopia de piel / cero labels al extractor) → arrow →
"Transformación" (skin tone (Fitzpatrick) / constancia de color / crops por
lesión / flips · color-jitter · blur), and separate card
"HAM10000 etiquetado" (10 015 img · 7 clases / split StratifiedGroupKFold /
por lesion_id / assert: 0 lesiones compartidas), with red bold note
"las etiquetas NUNCA tocan el pre-entrenamiento".

Band 2 — "2 · PRE-ENTRENAMIENTO SSL (v6) — 100 % sin etiquetas · igual
presupuesto GPU por brazo · MLflow + checkpoint por época" (green accent),
two sub-panels fed by two arrows from "Transformación":
Left "Brazo A — DINO continual": "Multi-crop" (2 globales 224 px / 6 locales
96 px) → "modelo de segmentación ViT-S/14" (init: DINOv2 público / cabeza MLP · 65 536
proto / ve TODOS los crops / recibe gradientes) —EMA→ "Profesor (EMA)" (copia
EMA del estudiante / solo crops globales / centering + sharpening / sin
gradientes); both feed "Loss DINO — cross-entropy" (modelo de segmentación → profesor ·
stop-grad).
Right "Brazo B — SimSiam desde cero": "2 vistas aumentadas" (augmentation
aleatoria independiente) → "ResNet-50 + projector" (init ALEATORIA / (cero
pesos externos) / projector MLP 3 capas / compartido entre vistas) →
"Predictor / stop-grad" (predictor MLP 2 capas / (rama activa) / stop-grad en
rama objetivo) → "Loss — cos. negativa simétrica" (+ detector de colapso (std
embeddings)).

Band 3 — "3 · EVALUACIÓN (protocolo v5) — backbone congelado" (amber accent),
fed by arrows from both Loss cards: "Backbone congelado" (requires_grad =
False / model.eval() / assert trainable == 0) → "Features" (GAP / token CLS /
L2-norm coseno / cache en disco) → "3 probes" (kNN coseno k=20 / linear
(logreg) / prototipos) → "Métricas" (balanced acc · F1 macro / precision ·
recall macro / AUC OvR · acc · ECE). Below, a wide card "Comparación contra 7
baselines congelados" (AlexNet · VGG19 · EfficientNet-B0 · ResNet-50 ·
ViT-B/16 · SimSiam · DINOv2 / mismo split, mismos probes, mismas métricas →
la diferencia es atribuible al backbone).

Title at top: "Arquitectura completa del sistema — datos, pre-entrenamiento
SSL y evaluación", subtitle: "propuesta full SSL: el extractor jamás ve una
etiqueta; las etiquetas entran únicamente en la evaluación".

---

Checklist al recibir la imagen (NO negociable antes de meterla al README):
1. Leer TODO el texto letra por letra — Nano Banana mete typos sutiles
   (tildes, "lesion_id", "StratifiedGroupKFold", "65 536").
2. Verificar que la flecha roja no atraviese cajas.
3. Verificar que no haya cajas o texto inventado.
4. Si hay 1-2 typos: pedir edición puntual ("fix only the text X, keep
   everything else identical"). Si hay 3+: regenerar.


---

# PROMPT v3 — poster ilustrado (el que está en clipboard 2026-08-27)

Forget the layout of the previous versions. You are a senior scientific illustrator creating a museum-grade INFOGRAPHIC POSTER (portrait 3:4, maximum resolution) for a medical-AI thesis, in the illustrated style of Google DeepMind blog figures and Nature paper posters — NOT a boxes-and-arrows flowchart. Use the attached reference ONLY as the source of content and Spanish text; redesign the visual language completely.

VISUAL LANGUAGE (mandatory):
- Rich illustrated elements with soft isometric 3D depth, not plain rectangles:
  • Band 1 (DATOS): a stack of circular dermatoscopy photo thumbnails (realistic skin-lesion crops), a database cylinder, and a 6-step Fitzpatrick skin-tone scale bar; an augmentation strip showing one lesion image transformed 4 ways (flip, color-jitter, crop, blur).
  • Band 2 (PRE-ENTRENAMIENTO SSL): two illustrated neural towers — left, a Vision Transformer drawn as a 3D stack of translucent slabs (like classic CNN architecture figures) for "modelo de segmentación ViT-S/14", with a ghosted twin tower for "Profesor (EMA)" connected by a purple dashed EMA arrow; right, two siamese ResNet-50 towers sharing weights, with projector/predictor drawn as smaller stacked blocks and a clear stop-gradient cut mark on the target branch. Each arm ends in a glowing loss node.
  • Band 3 (EVALUACIÓN): the trained backbone shown FROZEN (ice-blue tint + snowflake + padlock), emitting a feature-vector ribbon of small colored cells that fans into 3 probe chips (kNN, linear, prototipos) and ends in a mini metrics dashboard with tiny bar/gauge charts.
- Big stat callouts in huge bold numerals: "~48-70k", "10 015", "7 clases", "65 536", "k=20", "7 baselines".
- Numbered badges ① ② ③ per band, left accent bars: deep indigo, emerald, amber.
- One-sentence caption under each band in italic gray.
- Background: very light warm paper with faint dot grid; soft shadows; premium editorial typography (Inter/SF style), perfect alignment.

TEXT (Spanish, reproduce EXACTLY, no translation, no inventions, no typos):
Title: "Arquitectura completa del sistema — datos, pre-entrenamiento SSL y evaluación". Subtitle: "propuesta full SSL: el extractor jamás ve una etiqueta; las etiquetas entran únicamente en la evaluación".
Band 1 "1 · DATOS": card "Pool SSL — SIN etiquetas": ~48-70k imágenes · HAM10000 (train) + ISIC 2020 · dermatoscopia de piel · cero labels al extractor. Card "Transformación": skin tone (Fitzpatrick) · constancia de color · crops por lesión · flips · color-jitter · blur. Card "HAM10000 etiquetado": 10 015 img · 7 clases · split StratifiedGroupKFold por lesion_id · assert: 0 lesiones compartidas. Red note: "las etiquetas NUNCA tocan el pre-entrenamiento".
Band 2 "2 · PRE-ENTRENAMIENTO SSL (v6) — 100 % sin etiquetas": left "Brazo A — DINO continual": Multi-crop: 2 globales 224 px · 6 locales 96 px; estudiante ViT-S/14: init: DINOv2 público · cabeza MLP · 65 536 proto · ve TODOS los crops · recibe gradientes; Profesor (EMA): copia EMA del estudiante · solo crops globales · centering + sharpening · sin gradientes; Loss DINO — cross-entropy: red de segmentación entrenada con pseudo-máscaras → profesor · stop-grad. Right "Brazo B — SimSiam desde cero": 2 vistas aumentadas · augmentation aleatoria independiente; ResNet-50 + projector: init ALEATORIA (cero pesos externos) · projector MLP 3 capas · compartido entre vistas; Predictor / stop-grad: predictor MLP 2 capas (rama activa) · stop-grad en rama objetivo; Loss — cos. negativa simétrica + detector de colapso (std embeddings). Footer line: "igual presupuesto de GPU-horas por brazo · MLflow · checkpoint por época".
Band 3 "3 · EVALUACIÓN (protocolo v5) — backbone congelado": Backbone congelado: requires_grad = False · model.eval() · assert trainable == 0; Features: GAP / token CLS · L2-norm coseno · cache en disco; 3 probes: kNN coseno k=20 · linear (logreg) · prototipos; Métricas: balanced acc · F1 macro · precision · recall macro · AUC OvR · acc · ECE. Wide footer card: "Comparación contra 7 baselines congelados": AlexNet · VGG19 · EfficientNet-B0 · ResNet-50 · ViT-B/16 · SimSiam · DINOv2 — "mismo split, mismos probes, mismas métricas → la diferencia es atribuible al backbone".
A red dashed arrow labeled "etiquetas SOLO en evaluación" runs from "HAM10000 etiquetado" down the RIGHT MARGIN (touching nothing) into "Métricas".

FORBIDDEN: plain flat rectangles as the only visual device, cramped spacing, tiny illegible text, typos, watermarks, English text in the figure, photorealistic people.



---

# PROMPT v4 — edición quirúrgica + didáctica (2026-08-27)

Edit this exact image. Keep the style, layout, illustrations and colors IDENTICAL — this is a surgical edit plus small didactic additions, not a regeneration.

TEXT FIXES (exact):
1. The red banner in band 1 currently reads "A las etiquetas NUNCA tocan el pre-entrenamiento". Remove the leading "A" so it reads exactly: "las etiquetas NUNCA tocan el pre-entrenamiento".
2. In band 3, the label "Trained backbono" is wrong. Replace it with: "Backbone entrenado".
3. In the metrics dashboard illustration, the tiny gray gibberish labels under the small charts must be replaced with clean readable micro-labels: "balanced acc", "F1 macro", "AUC".
4. Change "KNN coseno k=20" to "kNN coseno k=20".

DIDACTIC ADDITIONS (same typography system, italic gray captions):
- Under band 1 add the caption: "El modelo aprende de miles de imágenes de piel sin diagnóstico; las etiquetas quedan guardadas únicamente para el examen final."
- Under band 2 add the caption: "Dos estrategias SSL compiten con el mismo presupuesto de GPU: refinar un modelo público (DINO continual) vs. aprender desde cero (SimSiam)."
- Under band 3 add the caption: "El backbone se congela y se examina con 3 clasificadores ligeros: si sus features son buenas, gana sin haber visto una sola etiqueta."
- Add small tag labels on the main flow arrows: "imágenes" (pool → transformación), "crops aumentados" (transformación → band 2), "backbone entrenado" (band 2 → band 3), "features" (backbone → probes).
- At the very bottom add a thin glossary strip in small text: "SSL = aprendizaje auto-supervisado (sin etiquetas) · EMA = promedio móvil exponencial · probe = clasificador ligero sobre features congeladas · ECE = error de calibración esperado".

Everything else stays pixel-identical. No new boxes, no watermarks, no typos.

