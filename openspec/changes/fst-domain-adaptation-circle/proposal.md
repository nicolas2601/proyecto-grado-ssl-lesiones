# Propuesta: FST, adaptación de dominio y CIRCLe

## Intención

El diagnóstico FST actual usa máscaras de respaldo y carece de validación externa; SSL y CIRCLe no forman una comparación integral. Se buscan validación FST provisional e impacto medible sobre la equidad, sin afirmar precisión clínica.

## Alcance

- Procesar 68.472 imágenes: HAM10000 (10.015), ISIC2019 redimensionado (25.331) e ISIC2020 redimensionado (33.126), con procedencia y cobertura.
- Iniciar con máscaras automáticas o pseudo-máscaras de lesión; registrar productor, versión y calidad. Calibración provisional con Fitzpatrick17k; validación experta posterior.
- Comparar DINOv2 SSL (referencia) y ResNet50 ImageNet (control supervisado): base, adaptación de dominio y adaptación más CIRCLe, con splits comparables y SSL sin etiquetas.
- JSON para manifiestos, métricas, intervalos y puertas; CSV opcional para auditoría.

## Fuera de alcance

ISIC2019 fullres o ISIC2018 como datos principales; entrenar ahora una CNN; afirmar validez clínica FST o mejora integral desde probes. Sin código ni experimentos en esta fase.

## Capacidades

### Nuevas

- `evidencia-fst`: conjunto versionado, máscaras trazables y calibración provisional.
- `adaptacion-dominio-ssl`: comparación reproducible sin etiquetas de diagnóstico o tono en el preentrenamiento.
- `evaluacion-equidad-tono`: resultados globales y por tono, incertidumbre y puertas.
- `mitigacion-circle`: transformaciones de tono y regularización de consistencia que preservan la lesión; CIRCLe no identifica FST.

### Modificadas

Ninguna; `openspec/specs/` no contiene especificaciones principales.

## Fases y puertas

1. **FST:** separar cobertura y validez; contrastar imagen completa y estimación enmascarada. Si las máscaras fallan, considerar ISIC2018 solo para una CNN auxiliar de segmentación de lesión, nunca para clasificar FST ni sustituir CIRCLe.
2. **Adaptación:** reconciliar selectivamente la rama de Nico; comparar base y adaptación sin filtración de etiquetas.
3. **Transformación:** verificar cambios de tono acotados y preservación de lesión, con procedencia de máscaras.
4. **CIRCLe:** ante fallo de equidad, comparar base, solo transformación y transformación con consistencia. Integrar al modelo completo solo tras beneficio creíble.

## Evaluación y éxito

Preservar accuracy, balanced accuracy, macro-F1, macro-precision, macro-recall, AUC, ECE, matriz de confusión, intervalos bootstrap, probe y semillas. Añadir F1, recall, accuracy y AUC por tono, EOD, NAR y brecha máxima. Éxito: reducir la brecha de equidad sin caída absoluta superior a 0,02 en macro-F1 global ni AUC global frente a la base comparable.

## Áreas afectadas

`src/dermssl/fitzpatrick.py`, `scripts/diagnostico_fst_comparativa.py` y `src/dermssl/data_resolver.py` (FST); `src/dermssl/pretrain/` (adaptación); `src/dermssl/circle.py`, `src/dermssl/finetune/`, `src/dermssl/gates.py` y `src/dermssl/metrics.py` (mitigación y evaluación).

## Riesgos y decisiones abiertas

Riesgos: pseudo-máscaras sesgadas, grupos escasos, rama divergente y GPU externa. Pendiente: umbrales de calidad/cobertura y tamaño mínimo, objetivo y checkpoint de adaptación, brecha prioritaria, activación e integración final de CIRCLe. FST estimado no equivale a verdad experta.

## Reversión

Conservar evidencia previa; si fallan las puertas, volver a la referencia versionada sin promover conclusiones de equidad.
