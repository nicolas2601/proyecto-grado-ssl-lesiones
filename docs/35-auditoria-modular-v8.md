# 35. Auditoría modular v8 y experimentos sin etiquetas

Registrado el 2026-10-10 (America/Bogota), antes de nuevos resultados científicos.
Base: `feature/thesis-figures`, revisión `07bb49761e6522451b025a341ee20d55aed525f9`.
Rama de trabajo: `experiment/modular-audit-v8`. Esta base contiene el historial de
`proyecto-de-grado`; no se incorporan cambios experimentales a las ramas históricas.

<!-- registro-auditoria-v8:inicio -->

## Propósito y restricciones

Aislar geometría de vistas SSL, selección de piel para ITA, calidad de pseudomáscaras
y continuación SSL de dominio. Integrar solo después de evaluar cada componente.

- Segmentación y continuación SSL sin diagnósticos, fototipos ni máscaras expertas.
- Las pseudomáscaras se usan en la auditoría para visualizar, no como entrada al encoder.
- Máscaras expertas exclusivamente en evaluación separada. Ni checkpoint ni umbral
  se eligen con el test. Los test ya publicados no recuperan condición de inéditos:
  las nuevas lecturas son exploratorias. Confirmación requiere una reserva intacta.
- Se conservan exclusiones, splits por lesión/paciente y deduplicación; no se reconstruye
  un split con otra versión de scikit-learn. PAD target_test no entra a la auditoría.
- Resultados históricos, pesos y salidas se conservan. Una carpeta por corrida; no sobrescribir.
- JSON/JSONL son fuente de cifras. Tabla generada por script, comprobable con `--check`.
- Estado explícito: diseñado, implementado, smoke sintético, ejecutado real, evaluado.

## W0. Auditoría de vistas del dataloader

**Implementación inicial:** `scripts/audit_ssl_views.py`, observador de geometría en
`src/dermssl/pretrain/torch_view_audit.py` y contratos en `src/dermssl/view_audit.py`.
Se llama al SimSiamPairDataset real. Sus imágenes y fotometría no se reimplementan.
El observador repite la geometría sobre pseudomáscaras con interpolación nearest;
restaura RNG y no cambia las vistas. No modifica el entrenamiento.

Muestreo uniforme sin diagnósticos: 100 imágenes elegibles del pool de entrenamiento,
sin reemplazo, semilla fijada. Primera auditoría sobre la receta pareada actual.
Se publican pares si tone_view=false, ternas si tone_view=true.
No sustituir una muestra fallida por otra: registrar/corregir el fallo.

Salidas: vistas desnormalizadas, pseudomáscaras geométricamente alineadas, overlays,
rejilla por caso, coordenadas del recorte, fracción de lesión retenida, área relativa,
máscara vacía y recorte sin lesión. Cambio RGB de lesión antes de augmentations,
hashes de entradas, código, exclusiones, versiones y receta.

Retención es un indicador respecto a una pseudomáscara, no verdad anatómica.
La geometría no convierte en fiable una máscara errónea. Inspeccionar originales y
mapas simultáneamente. La rejilla es diagnóstica, no una figura final de artículo.

El RNG se fija por índice de la lista seleccionada en un proceso CPU. Son nuevas
tiradas reproducibles de las transformaciones actuales, no reconstrucción de las
tiradas históricas de workers de entrenamiento. Auditar después por dataset y
artefactos; selección por diagnóstico prohibida.

**Puerta para el paso siguiente:** alineación correcta, provenance clara de v1/v2/tone,
reporte de recortes vacíos y cambio de color. No hay una tasa universal registrada
que convierta automáticamente el recorte en correcto. La inspección de estos datos
no debe utilizar etiquetas de test para seleccionar nuevos aumentos.

## W1. Selección de piel e ITA

La rama base YA contiene `src/dermssl/e5_roi.py`: anillo perilesional, exclusión
de saturación/reflejos/píxeles negros, detector de pelo inyectable y abstención.
Se audita y reutiliza, sin presentarlo como una implementación nueva de v8.

Tres regiones distintas: lesión, piel candidata y artefactos/no piel. El complemento
de lesión y el borde de una foto clínica no garantizan piel. El anillo también
puede incluir lesión omitida, eritema, sombra o fondo; evaluar esos casos.

| Brazo | Píxeles para ITA | Imagen RGB usada |
|---|---|---|
| I0 | Complemento histórico de lesión | Original |
| I1 | Anillo con exclusiones existentes, sin detector de pelo | Original |
| I2 | Mismo anillo y exclusiones + máscara de pelo | Original |

Máscara de lesión y anillo idénticos entre I1/I2. No usar inpainting en la medición:
se excluye pelo, no se mide color sintetizado. Un cuarto control con regiones de
piel manuales puede usarse SOLO en evaluación, nunca para entrenar ni ajustar
umbral en test. Comparar error de ITA, L*, b*, cobertura, abstención y estabilidad;
si no hay referencia de color/experta, reportar descriptivamente, no exactitud.
ITA y fototipo Fitzpatrick no son equivalentes. Conservar ITA continuo.

## W2. Inconsistencias de segmentación antes de cambiar arquitectura

Auditar amplitud original de saliencia antes de min–max, inversión por centro/borde,
restricción al campo, límites de área y selección de componentes. Umbrales de mapas
normalizados no se interpretan como probabilidades calibradas.

| Brazo piloto | Único cambio frente al control inmediato |
|---|---|
| S0 | Semillas/pipeline actual congelados |
| S1 | Abstención para mapas débiles; criterio registrado en desarrollo sin etiquetas |
| S2 | Orientación alternativa sin inversión automática solo por centralidad |
| S3 | Máscara de pelo excluida de semillas, sin inpainting |
| S4 | Inpainting antes del maestro, conservando la imagen original para medir ITA |

S1 y S2 requieren especificación y tests antes de ejecutar; sus criterios no se
inventan después de ver Dice. Comparar primero pseudomáscaras, después el mismo
estudiante y presupuesto. No variar simultáneamente maestro, decoder y aumentos.
Unión de componentes y remoción de pelo ya existen; registrar flags/checkpoint
reales en lugar de tratarlas como técnicas nuevas.

Inspección y métricas: fallos catastróficos, Dice/IoU, p10, HD95, sensibilidad de
lesión, cobertura y contaminación del ROI de piel. Contrastes pareados; estratos
por artefactos, tamaño y contraste. Luminancia no se presenta como fototipo.
Identificar grupos dentro/fuera del pool; estimar estratos también para externos.

Validación sobre pseudomáscaras mide acuerdo con el maestro; no prueba anatomía.
Autoentrenamiento añade señal solo si las semillas mejoran independientemente.
Pelo sintético sirve como prueba de robustez emparejada, no sustituye pelo real.

## W3. Métodos alternativos compatibles con entrenamiento sin etiquetas

| Método | Evidencia y papel | Limitación |
|---|---|---|
| SLED | Específico de lesiones; superpíxeles, entropía estructural y detección de anomalías. Primer maestro independiente candidato. | Depende de color/textura/contraste; compatibilidad y reproducción local pendientes. |
| MaskCut/TokenCut | Grafo de afinidad de representaciones SSL para generar máscaras sin anotaciones. Segundo maestro candidato. | Evidencia de objetos generales; transferencia a dermatoscopia es hipótesis. El Camino A ya explora espectral. |
| USL-Net | Referencia más cercana al Camino B: contrastivo, CAM, ignorar incertidumbre y ciclos. | Nuestro pipeline no es réplica exacta. Verificar mezcla de backbones, inicialización, datos y presupuesto. |
| Consenso entre maestros independientes | Lesión/fondo solo donde coinciden; desacuerdo ignorado; mismo estudiante. | Propuesta experimental propia, no réplica ni garantía de acierto. Medir cobertura; desacuerdo no implica cuál es correcto. |

Primero un piloto sin estudiante y sobre el mismo manifiesto. Conservar implementación,
revisión y licencia de cada tercero; no incorporar dependencias al entorno histórico.
Comparar máscaras antes de invertir GPU en aprendizaje.

SAM/MedSAM con pesos entrenados con máscaras humanas puede ser referencia externa,
pero no se presenta como método estrictamente libre de supervisión.
DINO/DINOv2 tiene conocimiento externo autosupervisado; distinguirlo del brazo
entrenado exclusivamente en el pool propio. Auditar origen de pesos de cada maestro.

Boosting convencional es supervisado: necesita objetivos para reponderar errores.
Con pseudomáscaras, optimiza el error contra esas semillas, no contra anatomía.
No es prioridad; consenso no se denomina boosting. Un probe con boosting tendría
etiquetas fuente y cambiaría el downstream, no corregiría segmentación.

## W4. SSL y dominio después de W0–W2

Ablación factorial: original–original sin término adicional; original–transformada
sin término adicional; original–original con tercera vista de tono; y receta
pareada con tercera vista. Igualar init, imágenes, exclusiones, pasos y semillas;
reportar costo adicional por tercera vista. No cambiar recetas v7 en silencio.

Dominio: encoder público intacto; continuación solo fuente desde público;
continuación fuente+target_adapt desde público. Mismos pasos, vistas y semillas.
Comparar muestreo mezclado contra balance explícito por dominio en otro contraste,
no cambiarlo a la vez. Receta/probe fuente fijados antes de evaluación externa.
Reportar salida completa y lectura condicional BCC/MEL/NV separadas.

Estos entrenamientos quedan **diseñados, no ejecutados** en esta entrega.

## Secuencia y entregables

1. W0 exportación y reporte descriptivo.
2. W1 anillo y pelo excluido, sin alterar preentrenamiento.
3. W2 correcciones aisladas de semillas.
4. W3 SLED y método espectral como pilotos independientes.
5. W4 ablation del tono y continuación de dominio.
6. Reintegración solo con evidencia por componente.

Código en src/scripts/tests; planes en docs/ y openspec/; salidas nuevas en
`resultados/auditoria_v8/<run>/` o directorio externo de salidas grandes.
Figuras finales: español, fondo blanco, Times New Roman, paleta piel/marrones,
trazos finos. Vistas/máscaras clínicas no se sintetizan para ilustrar resultados.

<!-- registro-auditoria-v8:fin -->

## Fuentes verificadas y alcance de lectura

- Li et al., USL-Net, DOI 10.1016/j.bspc.2023.105769:
  https://arxiv.org/html/2309.13289v1 (texto completo). Descripción metodológica;
  no se trasladan cifras entre splits.
- Zeng et al., SLED, ICDM 2023:
  https://arxiv.org/abs/2309.01899 (resumen) y https://github.com/zengguangjie/SLED
  (README/código disponible). Sin réplica ni evaluación local todavía.
- Wang et al., CutLER, CVPR 2023:
  https://arxiv.org/abs/2301.11320 y https://github.com/facebookresearch/CutLER
  (resumen y documentación oficial). No valida resultados en piel.
- Ulrich et al., Beyond Fitzpatrick:
  https://www.nature.com/articles/s41746-025-01770-4 (texto completo).
- Benčević et al., Understanding skin color bias:
  https://doi.org/10.1016/j.cmpb.2024.108044 (resumen).
- AdaBoost: documentación oficial scikit-learn,
  https://scikit-learn.org/stable/modules/ensemble.html#adaboost.

## Implementación y verificación (fuera del registro)

La entrega inicial añade observación del dataloader, exportador y generador de tabla.
No cambia pérdidas, geometría de entrenamiento, estimador ITA ni maestro.
No se afirma exportación de 100 imágenes reales: exige acceso al pool y pseudomáscaras
del clúster. Ver comandos y estado en [resultados/auditoria_v8](../resultados/auditoria_v8/README.md).



## Ejecución del 2026-10-11 (posterior al registro)

W0/W1/W2 ya ejecutados en copia separada del clúster. El estado diseñado/no ejecutado anterior corresponde a la entrega inicial. Ver [resultados y puertas](39-resultados-y-puertas-v8.md), protocolos 36–38 y [agregados de ejecución](../resultados/auditoria_v8/20261011/README.md). La evaluación ISIC2018 adicional es exploratoria y sus 100 IDs coinciden con el pool del maestro; no cierra la puerta de evaluación independiente. W3/W4 permanecen sin ejecución de nuevos maestros o entrenamientos.
