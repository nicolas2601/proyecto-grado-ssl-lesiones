# T4 — Shades of Gray (p = 6) antes del extractor, lectura única de val

Generado por `scripts/build_t4_table.py` desde `t4_final_eval.json`; no editar a mano. Registro, predicciones y salvedades: docs/31. Los terciles son terciles de luminancia, no tono de piel.

Val: n = 2004 (oscuro 820, medio 957, claro 183, sin tercil 44). Bootstrap pareado: 1000 remuestreos, semilla 20260923. Reproducción de v7.1 sin SoG: máx. |Δ| = 0,0000.

## Veredictos registrados

- **P1: no se cumple** (|Δ macro-F1| ≤ 0,0100 con IC que incluye 0; por peso: A* sí, B* no).
- **P2: no se cumple** (|Δ| del público mayor que el de A* y B*; |Δ|: Público 0,0077, A* 0,0083, B* 0,0241).
- **P3: no se cumple** (|Δ brecha| ≤ 0,0300 con IC que incluye 0; por peso: Público no, A* no, B* no).

## Contrastes primarios: SoG contra sin SoG en el protocolo fijo

| Peso | Configuración SoG | Macro-F1 | Δ brecha absoluta oscuro − claro (menor es mejor) |
|---|---|---|---|
| Público | `public@224|cls4|la0.5|tta` | -0,0077 [-0,0406; +0,0270] indistinguible | +0,0770 [-0,1044; +0,1679] indistinguible |
| A* | `A_soup_a1.0@224|cls_attn|la0.5|tta` | -0,0083 [-0,0314; +0,0155] indistinguible | +0,0453 [-0,1015; +0,1386] indistinguible |
| B* | `B_s42_a1.0@224|cls4|balanced|tta` | -0,0241 [-0,0532; +0,0085] indistinguible | +0,0604 [-0,1352; +0,1733] indistinguible |

## Contrastes secundarios: ganador SoG del holdout contra sin SoG

| Peso | Configuración SoG | Macro-F1 | Δ brecha absoluta oscuro − claro (menor es mejor) |
|---|---|---|---|
| Público | `public@224|cls4|balanced|tta` | -0,0055 [-0,0372; +0,0271] indistinguible | +0,1143 [-0,1007; +0,2003] indistinguible |
| A* | `A_soup_a1.0@224|cls_mean|la0.5|tta` | -0,0003 [-0,0282; +0,0263] indistinguible | +0,0164 [-0,1053; +0,1159] indistinguible |
| B* | `B_s42_a1.0@224|cls4|balanced|tta` | -0,0241 [-0,0532; +0,0085] indistinguible | +0,0604 [-0,1352; +0,1733] indistinguible |

## Macro-F1 por tercil (descriptivo)

| Entrada | oscuro | medio | claro |
|---|---|---|---|
| Público, sin SoG | 0,6882 | 0,6670 | 0,7260 |
| Público, SoG | 0,6314 | 0,6955 | 0,7462 |
| A*, sin SoG | 0,6640 | 0,7434 | 0,7309 |
| A*, SoG | 0,6764 | 0,7095 | 0,7885 |
| B*, sin SoG | 0,6692 | 0,7333 | 0,7016 |
| B*, SoG | 0,6024 | 0,7318 | 0,6953 |

Potencia: el tercil claro tiene pocas imágenes, así que un IC de la brecha que incluye el 0 se lee «no se detecta», no «no hay efecto».
