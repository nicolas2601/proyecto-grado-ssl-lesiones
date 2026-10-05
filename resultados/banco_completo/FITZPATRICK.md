# Banco completo: Fitzpatrick descriptivo (E5)

Generado por `scripts/build_fitzpatrick_table.py` desde los JSON de `resultados/banco_completo/e5/`. No editar a mano. Plan y reglas: `docs/33-banco-completo-multidataset.md` (E5).

> **Aviso.** FST estimado por ITA, NO validado (22,9 % en PAD), pseudo-máscaras automáticas. Descriptivo y NO está validado: los buckets I-VI son la salida del estimador ITA con umbrales fijos, no el fototipo Fitzpatrick. Ese estimador acierta 22,9 % frente a 51,3 % de la baseline de mayoría en PAD-UFES-20 (Alipour et al., Skin Research and Technology 32(8):e70374, 2026, PMC13459198; docs/20, docs/30) y 28,5-30,8 % frente a 32,9 % en Fitzpatrick17k-C. Los terciles de L* son luminancia (mediana de L* de la piel), no es tono de piel, y son relativos a cada dataset: «oscuro» en uno no equivale a «oscuro» en otro. Ningún conteo de este archivo cambia el resultado primario del banco (docs/33, sección 5).

## Glosario

- **ITA (ángulo de tipología individual):** ángulo calculado de la mediana de L\* y b\* de la piel en CIELAB, `arctan((L\* − 50) / b\*)`. Es una medida de color, no de fototipo.
- **Bucket I-VI:** el tipo Fitzpatrick que resulta de cortar el ITA con los umbrales fijos de Groh (`fitzpatrick.FST_THRESHOLDS`). **Es la salida de un estimador no validado, no el fototipo real.** Un valor justo en un umbral va al tipo más oscuro.
- **Abstención:** la imagen no recibe bucket (columna «abstención») cuando no hay piel suficiente, la máscara es degenerada, b\* queda bajo el mínimo o el archivo no se pudo leer. Nunca se fuerza un bucket.
- **Estimador de Otsu (referencia):** quita la lesión con un umbral de Otsu y lee la piel del resto; es el mismo en los tres datasets, así que es el comparable.
- **Estimador de imagen completa:** todos los píxeles cuentan, sin quitar la lesión.
- **Anillo perilesional (ROI de Pau, `perilesional_roi`):** banda justo fuera de la máscara de la lesión, sin pelo, reflejos, píxeles saturados ni borde negro, con abstención. Las máscaras son **pseudo-máscaras automáticas** de `student_r3`: nativas 450 × 600 en HAM10000 y del pool 224 × 224 en ISIC.
- **Estado de la máscara:** `c1_native` (HAM10000), `eligible` (máscara del pool usable), `quarantine` (el manifiesto la rechazó, p. ej. área degenerada) o `absent` (la imagen no está en el pool).
- **Borde sin máscara (`no_mask_border`):** las imágenes sin máscara elegible usan el marco de la imagen como muestra de piel, con las mismas exclusiones y abstención. Se cuenta aparte: asume que el borde es piel, lo que una lesión en el margen o una viñeta rompen.
- **L\*:** luminosidad CIELAB (0 negro, 100 blanco). **Tercil de L\*:** el dataset se divide en tres tercios por la mediana de L\* de su piel; es luminancia, **no es tono de piel**, y sus cortes son propios de cada dataset.
- **Descriptivo:** solo conteos. No hay exactitud, ni intervalos, ni contrastes: no se compara contra ningún fototipo real.
- **n/d:** no disponible o no aplicable.

## Cómo leer esta tabla

- Los conteos por bucket describen lo que el estimador ve, no la composición de fototipos del dataset: contra fototipos asignados por dermatólogos acierta 22,9 % frente a 51,3 % de la baseline de mayoría en PAD-UFES-20 (PMC13459198; docs/20, docs/30).
- Los terciles son relativos a cada dataset: «oscuro» en HAM10000 y «oscuro» en ISIC 2020 no son la misma luminancia, y ninguno es piel oscura.
- Que dos estimadores coincidan (cuadro Otsu × ROI) no los valida: dos estimadores no validados pueden coincidir y equivocarse juntos. Es acuerdo, no exactitud.
- Los splits son los que ya existen en cada dataset (no se unifican, docs/33, sección 1): HAM10000 la partición de C1 agrupada por `lesion_id`, ISIC 2019 `StratifiedKFold` y ISIC 2020 agrupada por `patient_id`; «val» es el primer fold de cada una.
- ISIC 2020: split reproducible solo con scikit-learn 1.7.2 (guard en el loader).
- E5 no cambia el resultado primario del banco (docs/33, sección 5).

## HAM10000

pendiente: este dataset todavía no se ha corrido (`scripts/e5_fitzpatrick.py` en el pod); no hay ningún número hasta que exista su JSON.

## ISIC 2019

pendiente: este dataset todavía no se ha corrido (`scripts/e5_fitzpatrick.py` en el pod); no hay ningún número hasta que exista su JSON.

## ISIC 2020

pendiente: este dataset todavía no se ha corrido (`scripts/e5_fitzpatrick.py` en el pod); no hay ningún número hasta que exista su JSON.
