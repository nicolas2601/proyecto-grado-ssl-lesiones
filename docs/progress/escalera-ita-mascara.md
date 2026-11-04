# Progreso: escalera ITA — cuánto cuesta el error de máscara en grados

Pregunta que responde esta línea: el brazo de segmentación v8, ¿sirve de verdad
como insumo del estimador de tono, o es un adorno? Se contesta midiendo el error
de ITA de cada opción de enmascarado contra un techo de máscara de verdad.

Protocolo de referencia: acta 2026-09-18 (`plan-segmentacion/README.md` §1) y
`docs/estado/clf-medicion-completa-a0-b-t.md`.

- [x] 0. Rescate de 12 módulos sin seguimiento | `f14a63a` | `python -m ruff check <los 12>` + `PYTHONPATH=src pytest tests/test_tone_augment_lab.py tests/test_tone_augment_fst.py tests/test_tone_normalization.py tests/test_pretrain_tone_augment_pairs.py tests/test_paired_factor_gate.py tests/test_seg_hair_removal.py tests/test_seg_field_roi.py` → 140 passed
- [x] 1. Escalera de 4 brazos sobre S-SYNTH (n=600) | sin commit (artefactos en UNABIA) | `~/work/ita-ladder-20261104/join4.py`
- [x] 2. Escalera de 4 brazos sobre ISIC 2018 Task 1 (n=2.594) | sin commit (artefactos en UNABIA) | `~/work/ita-ladder-20261104/final_isic.py`
- [x] 3. Causa del 28 % sin ITA | sin commit (artefactos en UNABIA) | `~/work/ita-ladder-20261104/diag_b.py`
- [x] 4. Tono contra contraste como conductor del sesgo | sin commit (artefactos en UNABIA) | `~/work/ita-ladder-20261104/contrast.py`
- [x] 5. `tone_augment_lab_v2` con Corbin eq. 7 en grados ITA | `8da01ad` | `PYTHONPATH=src pytest -q tests/test_tone_augment_lab_v2.py tests/test_tone_augment_lab.py` → 43 passed (ventana anclada al ITA medido, corrige colapso de `60d165f`)
- [ ] 6. Validación de la transformación contra melanina verdadera en S-SYNTH
- [ ] 7. Sesgo en PH2 y varianza del brazo B entre tonos
- [ ] 8. Rejilla completa del brazo T

## Resultados medidos

Error de ITA contra el techo de máscara experta, ISIC 2018, denominador común
n=1.813 (los casos donde los cuatro brazos resuelven):

| brazo | sesgo mediano | MAE | p95 |
|---|---:|---:|---:|
| sin máscara | −1,44 | 5,25° | 24,92 |
| Otsu (fallback actual) | −0,09 | 1,88° | 8,02 |
| teacher 026 | −0,03 | **0,91°** | 4,28 |

Por estrato de área de lesión, el mismo error:

| área | n | sin máscara | Otsu | 026 |
|---|---:|---:|---:|---:|
| 0,00–0,10 | 801 | 0,75 | 1,26 | 0,29 |
| 0,10–0,20 | 365 | 2,14 | 1,34 | 0,36 |
| 0,20–0,30 | 215 | 4,08 | 1,62 | 0,62 |
| 0,30–0,40 | 155 | 5,83 | 1,92 | 0,84 |
| 0,40–0,50 | 91 | 11,41 | 2,87 | 1,58 |
| >0,50 | 186 | **28,63** | 5,39 | 4,74 |

Lecturas que se sostienen:

- El segmentador reduce el error de ITA entre 2x y 6x según el estrato. No lo
  elimina: 4,74° en lesiones que ocupan más de la mitad del cuadro.
- Otsu depende del dominio. En dermatoscopia le gana a no enmascarar en todos
  los estratos ≥ 0,10; en los renders de S-SYNTH daba MAE 15,71 y era
  catastrófico. No afirmar "Otsu es dañino" sin nombrar el dominio.
- El Dice del teacher 026 medido aquí, 0,8319, reproduce el 0,8320 publicado.
- El sesgo del segmentador es de **contraste**, no de tono: Spearman con el
  Dice da contraste Lab +0,604 contra tono +0,193, y el área no participa
  (−0,014, p=0,71). El tono sobrevive como efecto propio pequeño (parcial
  +0,136, p=3e−04).

## Caveats que viajan con estas cifras

- Cobertura 69,9 %. El 30 % restante se abstiene por `MIN_B_FOR_ITA = 5.0`, y
  esa abstención crece con el área de lesión (20,3 % → 39,0 %). Toda la tabla
  es condicional a los casos que resuelven.
- La abstención no es un defecto: de las imágenes que no producen ITA, el 24 %
  tiene b\* negativo, y sin la guarda se les asignaría un ITA negativo, es
  decir fototipo V–VI. Sin esa guarda, imágenes con dominante azulada se
  etiquetarían como piel oscura.
- Ningún dataset de esta línea tiene piel oscura real. El tercil "oscuro" de
  ISIC 2018 cae en *intermediate/tan* de Del Bino. Decir "brecha por tercil de
  ITA", nunca "brecha por fototipo".

## Desajustes con el protocolo, abiertos

1. El acta manda seguir el flujo de Corbin, que muestrea `ITA'` dentro del
   rango del FST objetivo. El propio `plan-segmentacion/README.md` declara que
   ITA→FST no está validado (22,9 % contra 51,3 % de la baseline de mayoría en
   PAD-UFES-20). Corbin se adapta a grados ITA continuos, no se copia.
2. El acta exige separar bien lesión y piel "también en piel oscura", pero ni
   el pool ni ISIC 2018 la tienen, y Benčević descarta ISIC para evaluar sesgo.
   El objetivo no es testeable en los datos elegidos.
3. El acta apunta al tono; la medición dice que el conductor es el contraste.

Bloqueos: ninguno para los pasos 5–8. El paso 0 los desbloqueó.

## No verificado

- `tests/test_pretrain_tone_view.py` no corre en el entorno local: falta
  `torchvision`. Queda sin ejecutar, no pasado ni fallado. Se comprueba con
  `CI_REQUIRE_FULL_DEPS=1` en un entorno con las dependencias completas.
- Los 12 módulos del paso 0 se versionaron **tal como estaban, sin revisión**,
  para frenar el riesgo de pérdida primero. Su contenido no está auditado.
