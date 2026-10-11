# Pilotos de desarrollo W0–W2

Registrado antes de ejecutar estos contrastes; revisión fuente f8b1790. No diagnósticos, máscaras expertas ni PAD target_test.

## Selección y alcance

Conservar los 100 IDs y las tiradas de W0. Desglosar por dataset y área de
pseudolesión (<5%, 5–20%, 20–50%, ≥50%); no inferir tasas poblacionales de grupos
pequeños. Diagnóstico separado de cuarentena: semilla 43, 25 IDs por dataset,
sin reemplazo, ordenado por ID. No se reintroduce ningún caso al entrenamiento.

Comprobar manifiesto nativo y correspondencia índice/ID/dataset, igualdad exacta
RGB cache–PNG y máscara r2–PNG para entrenamiento, y hash semántico de máscara
para ambos grupos. Guardar hashes de cada slice nativo, flags y salidas.

## W1

I0: complemento histórico por estimate_fst; I1: ring_estimate sin pelo;
I2: el mismo anillo y exclusiones más elongated_hair_mask existente. Medir RGB
original, nunca inpainting. Conservar ITA, L*, b*, cobertura y abstenciones.
Casos cuarentena permanecen mask_ineligible en las tres variantes. Sin referencia
de color ni experta, solo diferencias descriptivas, sin exactitud ni fototipo.

## W2: semilla CCAM congelada

S0 reproduce pseudo_r0 con sus flags persistidos; exigir igualdad exacta en
todos los slices seleccionados antes de interpretar otros brazos. No es una
réplica de entrenamiento ni una nueva máscara del estudiante r2.

S1 cambia solo abstención: si max(raw)−min(raw) ≤ 0.01 (salida sigmoide original),
todo el mapa queda IGNORE. Hipótesis conservadora de desarrollo, umbral no
calibrado y fijado antes de sus resultados; mapas no finitos también abstienen.
No usar min–max para decidir debilidad. No elegir este umbral con Dice.

S2 cambia solo orientación: omitir inversión centro/borde; conservar normalización,
umbrales, campo, artefactos y selector de componentes de S0. No implica que el
signo sin invertir sea correcto; registrar cuáles cambian y revisar ambas máscaras.

S3 cambia solo máscara de artefactos: unión con pelo elongado existente, IGNORE
en esos píxeles; no inpainting. Mantener todos los demás flags de S0.

S4 exige volver a inferir el maestro con y sin inpainting, mismo checkpoint y
precisión. No se simula alterando una saliencia existente. Conservar RGB original
para ITA. Verificar procedencia del input y reproducción numérica antes de ejecutar.

Pruebas sintéticas obligatorias: mapa débil abstiene en S1; mapa fuerte conserva
S0; S2 omite solo inversión; S3 jamás etiqueta como lesión un píxel de pelo.
Acuerdo con r2 es acuerdo con un estudiante, nunca Dice anatómico.

## Puertas siguientes

W3: pilotos independientes SLED/espectral sobre manifiesto idéntico, licencia,
revisión y pesos explícitos, entorno separado. Comparar máscaras antes de estudiante.
W4: factorial de vistas y continuación de dominio con init/pool/pasos/semillas
idénticos. No lanzar automáticamente estudiantes/SSL mientras W2 carezca de
evaluación anatómica independiente. Registrar estado ejecutado/diseñado por brazo.
