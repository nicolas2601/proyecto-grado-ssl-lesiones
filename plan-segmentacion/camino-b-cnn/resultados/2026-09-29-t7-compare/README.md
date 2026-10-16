# T7 — Experimento de segmentación 5 reentrenado con oscurecimiento y sobremuestreo del tercil oscuro (exploratorio)

Generado por `scripts/t7_compare.py` desde `t7_compare.json`; no editar a mano. Registro, predicción y salvedades: docs/27, bloque T7. Los bins son `tone_tercile` del manifiesto publicado: luminancia, no es tono de piel. La luminancia está confundida con el diagnóstico, el dermatoscopio y la iluminación, y HAM10000 es transductivo (limitación del diseño).

Bootstrap: 1000 remuestreos, semilla 20260923, IC percentil 95 %. G, R y S (la brecha y sus contrastes: estratificado por oscuro y resto, con los mismos índices para todos los segmentadores). ΔD global: pareado por imagen, sin estratificar, como en T6 y en el registro.

## HAM10000 (veredicto)

n oscuro = 4082, n resto = 5740, sin tercil = 193.

| Segmentador | Dice oscuro | Dice medio | Dice claro | HD95 oscuro | G oscuro − resto [IC 95 %] |
|---|---:|---:|---:|---:|---|
| Experimento de segmentación 5 | 0,7623 | 0,8134 | 0,8615 | 25,30 | -0,0583 [-0,0653; -0,0515] |
| T7 | 0,7583 | 0,8110 | 0,8587 | 25,39 | -0,0599 [-0,0668; -0,0531] |
| Experimento de segmentación 5, semilla 43 | 0,7662 | 0,8164 | 0,8648 | 24,63 | -0,0574 [-0,0643; -0,0508] |
| T7-aug | 0,7598 | 0,8134 | 0,8664 | 25,27 | -0,0616 [-0,0687; -0,0547] |

| Brazo | R = \|G(Experimento de segmentación 5)\| − \|G(brazo)\| [IC 95 %] | ΔD global vs Experimento de segmentación 5 [IC 95 %] |
|---|---|---|
| T7 | -0,0016 [-0,0024; -0,0006] | -0,0031 [-0,0035; -0,0026] |
| Experimento de segmentación 5, semilla 43 | +0,0009 [+0,0002; +0,0016] | +0,0034 [+0,0031; +0,0037] |
| T7-aug | -0,0033 [-0,0041; -0,0024] | -0,0006 [-0,0010; -0,0001] |

S = R(T7) − R(Experimento de segmentación 5, semilla 43), pareado: -0,0024 [-0,0034; -0,0015]. R(T7) − R(T7-aug), descriptivo: +0,0017 [+0,0009; +0,0026].

Parte 1 (brecha): **no cumplida**. Parte 2 (global): **se sostiene**. Predicción de Pau: **no cumplida**. Ruido de semilla: **no se separa del ruido de semilla**.

Lectura registrada: **no cumplida**.

## HAM10000 fold-0 val (descriptivo)

n oscuro = 820, n resto = 1140, sin tercil = 44.

| Segmentador | Dice oscuro | Dice medio | Dice claro | HD95 oscuro | G oscuro − resto [IC 95 %] |
|---|---:|---:|---:|---:|---|
| Experimento de segmentación 5 | 0,7635 | 0,7934 | 0,8840 | 24,87 | -0,0444 [-0,0619; -0,0286] |
| T7 | 0,7589 | 0,7910 | 0,8786 | 24,92 | -0,0461 [-0,0640; -0,0306] |
| Experimento de segmentación 5, semilla 43 | 0,7674 | 0,7968 | 0,8867 | 24,23 | -0,0438 [-0,0613; -0,0286] |
| T7-aug | 0,7609 | 0,7941 | 0,8872 | 24,83 | -0,0482 [-0,0655; -0,0326] |

| Brazo | R = \|G(Experimento de segmentación 5)\| − \|G(brazo)\| [IC 95 %] | ΔD global vs Experimento de segmentación 5 [IC 95 %] |
|---|---|---|
| T7 | -0,0017 [-0,0036; +0,0002] | -0,0036 [-0,0046; -0,0027] |
| Experimento de segmentación 5, semilla 43 | +0,0006 [-0,0006; +0,0019] | +0,0035 [+0,0029; +0,0041] |
| T7-aug | -0,0037 [-0,0055; -0,0018] | -0,0005 [-0,0015; +0,0006] |

S = R(T7) − R(Experimento de segmentación 5, semilla 43), pareado: -0,0023 [-0,0042; -0,0003]. R(T7) − R(T7-aug), descriptivo: +0,0020 [+0,0003; +0,0039].

## ISIC 2018, réplica (descriptivo)

n oscuro = 158, n resto = 564, sin tercil = 1872.

| Segmentador | Dice oscuro | Dice medio | Dice claro | HD95 oscuro | G oscuro − resto [IC 95 %] |
|---|---:|---:|---:|---:|---|
| Experimento de segmentación 5 | 0,7808 | 0,8154 | 0,8367 | 24,58 | -0,0444 [-0,0725; -0,0159] |
| T7 | 0,7802 | 0,8144 | 0,8355 | 24,36 | -0,0439 [-0,0709; -0,0164] |
| Experimento de segmentación 5, semilla 43 | 0,7848 | 0,8176 | 0,8384 | 24,35 | -0,0424 [-0,0691; -0,0143] |
| T7-aug | 0,7785 | 0,8156 | 0,8384 | 24,52 | -0,0476 [-0,0754; -0,0183] |

| Brazo | R = \|G(Experimento de segmentación 5)\| − \|G(brazo)\| [IC 95 %] | ΔD global vs Experimento de segmentación 5 [IC 95 %] |
|---|---|---|
| T7 | +0,0005 [-0,0034; +0,0046] | -0,0009 [-0,0029; +0,0008] |
| Experimento de segmentación 5, semilla 43 | +0,0021 [-0,0004; +0,0049] | +0,0024 [+0,0016; +0,0034] |
| T7-aug | -0,0032 [-0,0071; +0,0003] | +0,0002 [-0,0011; +0,0017] |

S = R(T7) − R(Experimento de segmentación 5, semilla 43), pareado: -0,0016 [-0,0060; +0,0026]. R(T7) − R(T7-aug), descriptivo: +0,0037 [-0,0000; +0,0079].
