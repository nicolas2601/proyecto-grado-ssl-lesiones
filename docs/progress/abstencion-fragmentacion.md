# Abstención por fragmentación en el contrato de máscara

## 2026-11-05 — Regla implementada, opt-in

**Qué.** `src/dermssl/mask_manifest.py` admite el criterio de abstención por
fragmentación. `build_mask_record(..., max_significant_components=...)`
cuarentena las máscaras con más de un componente conexo significativo
(componente ≥ 0,5 % del área de lesión, 4-conectividad).

**Por qué es abstención y no reparación.** Medido contra las 2.594 máscaras
expertas de ISIC 2018: quedarse con el componente mayor **empeora** el Dice
−0,0056, IC95 [−0,0090; −0,0023], que excluye el cero; sobre el subconjunto
fragmentado, −0,0334. Cuando el blob mayor no llega al 80 % del área los
componentes secundarios son lesión real. Abstenerse, en cambio, sube el Dice
retenido de 0,8319 [0,8244; 0,8393] a 0,8818 [0,8763; 0,8871] — intervalos que
no se solapan.

**Costo de cobertura, declarado de antemano.** 16,8 % en ISIC 2018 y
**41,0 % en PAD-UFES-20**. `build_mask_manifest` ahora reporta
`coverage_fraction` y `quarantined_by_reason` para que una corrida que pierde
41 % de una cohorte no pueda parecer una que no perdió nada.

**Es opt-in y no mueve nada existente.** Con `max_significant_components=None`
(el default) el registro es idéntico byte a byte al contrato previo: no se
cuentan componentes, no se agrega campo y `eligible` no cambia. La precedencia
de `quarantine_reason` conserva el orden anterior (confianza, luego área), así
que `fragmented` solo puede aparecer en máscaras que el contrato viejo aceptaba.
`SCHEMA_VERSION` pasa a 1.1.

**Comando que pasa.**
`PYTHONPATH=src python -m pytest tests/test_mask_manifest_fragmentation.py tests/test_mask_manifest.py -q` → 21 passed.
`python -m ruff check src/dermssl/mask_manifest.py tests/test_mask_manifest_fragmentation.py` → All checks passed.

**Límites.** El umbral de 0,5 % y la 4-conectividad replican cómo se midió la
tasa de fragmentación; no están calibrados contra PAD, donde el Dice del
teacher 026 sigue **desconocido** por ausencia de máscaras expertas. La regla
transfiere; el número 0,8818 no.

**No hecho.** No se regeneró ningún manifiesto, no se activó la regla en
ninguna corrida y no se tocó ningún gate de ejecución. Activarla cambia la
cohorte que consume el clasificador: avisar antes de correr encima.
