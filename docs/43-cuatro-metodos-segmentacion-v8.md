# 43. Cuatro adaptaciones de segmentación en v8

2026-10-13, America/Bogota. Base `5c6dd0f`; rama **experiment/modular-audit-v8**.
Solicitud: incluir SLED, USL-Net, RPI-Net y TriDermNet-HAGO, guiados por los papers,
con carpetas independientes y comprobación ordenada. Amplía 41/42; no sustituye
resultados históricos ni habilita W4.

## Entrega y alcance

Se añadieron implementaciones propias, configuración y guías por método en
`src/dermssl/segmentation_methods/{sled,usl_net,rpi_net,tridermnet_hago}`.
`common/` contiene datos, procedencia, backbones/SSL, training, exportación,
refinamiento y overlays locales. El entry point es `scripts/run_segmentation_v8.py`.
Pruebas separadas en `tests/segmentation_v8/`; índice en
`plan-segmentacion/metodos-v8/README.md`.

La ausencia de código público no impide reimplementar ideas del artículo. Las
convenciones y discrepancias se documentan aquí y en cada README. **No se afirma
réplica exacta, superioridad ni exactitud anatómica** por pasar tests. Los pesos
preentrenados, selección de datos y presupuesto siguen siendo parte del método.

| Método | Implementación | Decisiones principales de adaptación |
|---|---|---|
| SLED | SLIC, grafo autoajustado, solver H2 Python, bisección, IForest, multiescala, GHT, selector | Python en vez de Java; centroides incluyendo cero; gaussiano completo; semilla y serial; no dependencia PyOD |
| USL-Net | Tres extractores SSL, fusión, CCAM, UM, FCN, preentrenamiento y refinamiento | Seguir prosa en centralidad/BCE; geometría y fusión declaradas; objetivo CCAM existente; procedencia por encoder |
| RPI-Net | Cuatro pseudofuentes, IPC, PIA, ramas mu/sigma, DUL e inferencia RGB | Target CAM energía SSL; cluster por borde; tokens14²; alpha detach y límites sigma; actualización P2 registrada |
| TriDermNet-HAGO | Region/Edge/Token, TER, pérdidas pseudo, control de gradientes HAGO | Clasificación diagnóstica reemplazada por cobertura pseudo; signo de balance según prosa; residual TER; ruido apagado |

Los cuatro `config.json` mantienen `execution_enabled=false`. Se puede consultar
config o ejecutar comprobación sintética; generar/entrenar con datos reales requiere
puertas, pesos y presupuesto. Código implementado no significa ejecución científica.

## Correspondencia con las fuentes

### SLED

Fuente: Zeng et al., ICDM2023, arXiv2309.01899; referencia de código fijada
`SELGroup/SLED@3213eacb65577c0d52f424369158fb56c609238e`.

Se implementa el objetivo de entropía estructural de dos niveles, merges con
beneficio positivo y relocaciones que disminuyen H2. Orden de empates y límite de
30 pases son propios. No se ha comparado bit a bit con Java. El pipeline usa
IsolationForest de sklearn directamente, invirtiendo su decision_function; el
campo inválido permanece excluido de saliencia final. Se evita el JAR y PyOD.
Los controles `upstream_zero` y `upstream_area` aíslan el centroide cero y la
selección por área frente a los corregidos. No son la réplica completa original.

Degeneraciones (volumen cero, bisección sin contraste, score constante) producen
abstención registrada, nunca una máscara inventada. La selección espacial sigue
el esquema registrado en42 y el llenado de huecos se restringe a valid.

### USL-Net

Fuente: PDF arXiv2309.13289v3 aportado, DOI10.1016/j.bspc.2023.105769.

`ContrastiveFusion` requiere tres orígenes diferenciados SimCLR/MoCov1/MoCov2 y
mantiene sus encoders congelados durante CCAM. Se concatena conv4/conv5 por
extractor; el último stride1 conserva geometría. Se utiliza cabeza1x1 background
más complemento y el objetivo CCAM ya existente del repo, sin cambiar históricos.
Las dimensiones/fusión impresas en el paper no cierran toda la receta: nuestras
proyecciones y canales son elecciones explícitas, no una inferencia de pesos.

UM sigue mínima suma de distancias y prosa BCE con ignore, no las expresiones
contradictorias del documento42. FCN usa conv2/3/4 (layer1/2/3 torchvision),
proyectadas a256, suma y bilinear. Se evita confundir sumas de distancias con
centroides. Corte máximo de distancia aún sin valor inventado; debe registrarse
antes del primer ensayo. Presupuesto nativo5x300 se distingue del piloto propio.

### RPI-Net

Fuente: PDF completo, DOI10.1016/j.bspc.2025.108433, disponible online14-08-2025.

Se conservan las cuatro fuentes MoCov3 conv1, MoCov3 CAM, MoCov2 conv1 y DINO
final; no sustituir DINO por DINOv2 implícitamente. El target de GradCAM++ en un
encoder SSL no se especifica en la fuente: adoptamos energía de features y fórmula
positive-gradient-power, con orientación de clustering por mayoría en borde.
Son adaptaciones, no diagnósticos. P2 es un target suave; P1/P3/P4 binarios.

IPC usa queries imagen y contexto pseudo; PIA agrega atención entre distintas
fuentes y residual. La predicción final mu usa sólo rama imagen; sigma depende de
interacción. ResNet50 y proyecciones multiescala256; encoder pseudo compartido.
El grid14² limita memoria cuadrática; el límite del constructor impide tokens de
imagen completa. Inferencia posterior no necesita pseudomáscaras como entrada.

DUL evalúa20–23. Sigma se interpreta como log-varianza; límites[-8,12] y alpha
detach por defecto son convenciones de estabilidad registradas. Se reporta fracción
clipped. La función softplus(sigma)^2 es la definición del autor, no entropía
Bernoulli. Hay controles para `detach_alpha=false`. El número de fases nativo no
está completamente especificado; en refinamiento propio se conserva diversidad
P1/P3/P4 y sólo se actualiza P2. No se llama réplica literal a esa elección.

### TriDermNet-HAGO

Fuente: PDF completo, DOI10.1016/j.bspc.2026.111536, online08-10-2026.

Region usa dilataciones1/2/4 y compresión/residual; Edge usa Sobel, convoluciones y
sigmoid gate; Token usa patches/posiciones/atención/FFN. TER modula características
espaciales con gamma/beta y contexto. El GAP singleton da atención sin competencia:
se añade residual de tokens explícito para preservar información. Width32,
token64, stem stride2 y pooling acotado son decisiones de recursos, no las cifras
de parámetros del artículo.

El protocolo W3 prohíbe diagnósticos y máscaras expertas en entrenamiento.
La variante sustituye clasificación por cobertura pseudo y consistencia con área,
y limita supervisión de bordes a vecindarios confidentes. No enseña diagnóstico
ni extrapola los Dice supervisados del artículo. Un brazo supervisado futuro
requeriría otro protocolo y reserva de evaluación; no se habilita aquí.

HAGO: warmup, gradientes por tareas, estadísticas por capa, balance y entropía.
La ecuación42 favorece tarea dominante pese a prosa contraria; default la suprime,
con `paper_logistic` como control. Eq44 repite wcls; se utilizan los dos pesos del
algoritmo. La combinación gstab/gbal no definida se resuelve escalando por capa el
gradiente balanceado. Base AdamW, ruido0 por defecto. Entropía de cobertura binaria,
no confianza diagnóstica. Activar perturbación es otro cambio que exige registro.

## Preentrenamiento y procedencia

Se incluyen componentes SSL locales SimCLR/MoCov1/MoCov2, variante EMA de
MoCov3 y ViT con distilación tipo DINO. Son opciones **locales adaptadas**, con
receta propia flip+fotometría, no recreaciones de la preconfiguración original.
DINO local usa por defecto ViT384/depth12/heads6/patch16 y head1024: presupuesto
diferente del original. MoCov3 local usa predictor y negativos de batch; no se
ocultan cambios de augmentations, batch global, inicialización ni épocas.

Alternativamente, banco con pesos externos revisados. `source_bank` exige cada
checkpoint y JSON de procedencia, método, revisión, SHA y `input_normalization`
(unit_rgb o imagenet). El formato propio carga sólo tensores con weights_only,
state_dict estricto; no ejecuta pickles, hub ni scripts de terceros, y no descarga.
Normalización se aplica igual en conv1 y capas profundas. Un checkpoint de otro
formato necesita conversión revisada a archivo NUEVO, nunca alterar el original.

La preparación no llama a entrenar ni elige pesos automáticamente. Ausencia de
checkpoint sólo permite tests sintéticos o iniciar después SSL local autorizado y
presupuestado, nunca generar pseudomáscaras de redes aleatorias como fiables.

## Manifiesto completo y contratos

No hay límite fijo de100. El manifiesto define TODOS los originales elegibles;
100 es muestra de revisión, no tamaño definitivo ni nueva partición.

Formato de alto nivel:

```json
{
  "schema_version": 1,
  "partition": "train",
  "count": 40896,
  "eligibility": {
    "verified": true,
    "excludes_target_test": true,
    "patient_lesion_groups_preserved": true,
    "deduplication_preserved": true,
    "source_manifest_sha256": "<hash del manifiesto fuente real>",
    "exclusions_sha256": "<hash de exclusiones reales>",
    "verification_revision": "<revision que verifico elegibilidad>"
  },
  "records": []
}
```

Ejemplo de esquema, NO manifiesto ejecutable: records debe contener count entradas,
con IDs únicos, dataset, image_path y image_sha256. 40896 refleja el conteo
histórico documentado, no una constante del código ni una nueva verificación del
pool hoy. No convertir la muestra development de100 en train por cambiar una
etiqueta: hay que vincular el índice elegible completo ya congelado y exclusiones.

Para training: `pseudo_paths`, SHA por fuente y procedencia con uses_expert_masks
false y uses_diagnoses false; 1 mapa0/.5/1 en USL/TriDerm, cuatro mapas[0,1] en
RPI. Valid opcional con SHA y binario. Labels/GT/fototipo/ITA se rechazan como
campos de entrada. Hashes se comprueban antes y durante lectura. Las afirmaciones
de elegibilidad requieren auditoría real: sus flags no la producen automáticamente.

Geometría: RGB bilinear, pseudos/valid nearest, misma resolución y flips pareados
con RNG por índice/época. No nuevos splits, no lectura de diagnósticos, no selección
por métrica de test. Si una fuente necesita inpainting, se registra aparte; estos
adaptadores por defecto reciben original RGB sin modificar color de ITA.

## Ejecución, salidas y ciclos

- `check`: consulta configuración y opcionalmente hashes de originales.
- `synthetic`: pruebas del método, `--include-common` añade contratos compartidos.
- `pretrain`: fuentes SSL propias; `--ssl-origin` explícito.
- `ccam`: entrenamiento cabeza USL sobre banco congelado.
- `generate`: semillas SLED/USL/RPI; TriDerm consume maestros, no finge ser maestro
  independiente por generar máscaras sin una red entrenada.
- `fit`: entrenamiento propio con pseudomáscaras verificadas.
- `predict`: máscaras finales; RPI puede inferir únicamente desde RGB.

Cada output nuevo preserva input_count, éxitos y abstenciones. Directorio por caso,
pseudoN.npy, valid, probabilidad final, original resized y overlays locales.
JSON/JSONL son evidencia; no subir imágenes/IDs/rutas clínicas a W&B. Si hay
abstenciones, se conserva el denominador y se exige política downstream explícita;
no se filtra silenciosamente un pool parcial como si fuera completo.

`common/refinement.py` escribe otro manifiesto con las máscaras refinadas, sin
modificar anteriores. Cada ciclo necesita puerta vinculada al nuevo hash y
presupuesto; no se lanza automáticamente5x300 ni se sustituyen todas las fuentes
RPI por una sola predicción. La orquestación de ciclos largos aún requiere una
receta aprobada en desarrollo; no se considera una corrida ya realizada.

## Puertas y límites de recursos

`execution_enabled=false` mantiene etapas reales bloqueadas. Evidencia gate debe
coincidir con método y hash de manifiesto; data_verified, resource_budget_measured,
weights_provenance_verified y smoke_passed true. Training exige además evaluación
anatómica independiente resuelta. El esquema de gate no crea esa validación.

El launcher ejecuta un grupo de proceso propio, con timeout y máximoRSS; no mata
otros jobs. Presupuesto de disco positivo y estimación conservadora del pool antes
de exportar, con reserva libre. CUDA exige máximo de memoria medido y reserva;
se limita el allocator del proceso. Default CPU, ninguna GPU usada en pruebas.
Presupuestos deconfig son límites de prueba, no autorización para todo el pool.

W&B online es obligatorio en etapas ejecutadas del launcher, con logger común,
`WANDB_NETRC_FILE`, carpeta tracking nueva, code hashes y métricas agregadas.
Console/code/media deshabilitados; no incorporar netrc al repo. Verificación local
puede ejecutarse offline con `verify_segmentation_v8_local.py`, reportando que no
hubo run remoto. Las pruebas locales no se presentan como monitorizadas.

## Comprobación realizada y limitaciones actuales

Pruebas CPU en dependencias aisladas de scratch: PyTorch2.11.0+cpu, Python3.12;
no se modificó la .venv histórica del clúster. Evidencia local versionada en
`resultados/auditoria_v8/cuatro_metodos_20261013/verification_local.json`.
Comprueban forward/backward, ecuaciones, ignorados, procedencia, restricciones,
geometría, checkpoint y un paso de optimizer sobre datos sintéticos. Eso no es
entrenamiento científico ni segmentación sobre pacientes.

Verificación final del paso 2: 30 pruebas ejecutadas, 0 fallos, 0 errores y
0 omitidas. Los cuatro comandos `check` pasaron con ejecución real deshabilitada.
Se corrigió una importación del escritor de evidencia que impedía guardar el JSON;
la suite se volvió a ejecutar y el informe quedó guardado correctamente.

SLED default completo (224², diez escalas200..650) terminó sobre imagen sintética
en aproximadamente7.8s y156MiB de máximoRSS local. Es diagnóstico de recursos,
NO pronóstico fiable para cada original, cluster o GPU; no habilita puerta real.

Acceso seguro llegó a JupyterLab, pero después browser bloqueó observación por
estado nativo de credenciales no recuperable. No se continuó ni se intentó leer
credenciales. Por ello los módulos nuevos **no están sincronizados/verificados hoy
en el clúster**, y no se creó nueva corrida W&B. Lo anterior de42 permanece como
historial distinto. Se publica código comprobado localmente en v8; la verificación
clúster/remota queda pendiente explícita, sin inventar acceso ni resultados.

Próxima secuencia: recuperar acceso operativo; comprobar pool/recetas/bancos;
primer original con recursos y W&B; maestros contra S0; evaluación independiente;
mismo estudiante/presupuesto; decidir integración y W4. No elegir ganador con
acuerdo entre pseudomáscaras ni con métricas de confident pixels solamente.

## Fuentes

- SLED: https://arxiv.org/abs/2309.01899; referencia código fijada arriba.
- USL-Net: https://arxiv.org/abs/2309.13289, v3 PDF aportado (hash en42).
- RPI-Net: https://doi.org/10.1016/j.bspc.2025.108433, PDF aportado (hash en42).
- TriDermNet-HAGO: https://doi.org/10.1016/j.bspc.2026.111536, PDF aportado.
- Objetivo CCAM propio preexistente: `plan-segmentacion/camino-b-cnn/pseudo-mascaras/ccam_engine.py`.
- Protocolos:35/36/39/41/42. Las adaptaciones no alteran sus restricciones.
