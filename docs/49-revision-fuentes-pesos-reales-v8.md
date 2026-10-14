> Estado más reciente: [descarga Hugging Face y validación local](52-simclr-huggingface-validacion-local-v8.md). Las afirmaciones siguientes sobre SimCLR pendiente describen el estado previo a esa validación.

# Revisión de fuentes para pesos reales V8

> **Actualización, 2026-10-14 UTC:** MoCo v1/v2/v3 y DINO original ya tienen archivos reales y recibos de conversión estricta. No hay comparación con un SHA-256 completo publicado. El adaptador 3 prepara el perfil SAME exclusivo de SimCLR, todavía sin pruebas runtime ni checkpoint real. Ver [51](51-cierre-preparacion-y-bloqueos-lanzamiento-v8.md). La revisión documental que sigue conserva su alcance y fecha histórica.

2026-10-13, America/Bogota. Continúa `22ef5129`; no es una validación de archivos
de pesos ni una repetición de las auditorías de datos o modelos.
Evidencia estructurada: [public-weights.evidence.json](../plan-segmentacion/metodos-v8/public-weights.evidence.json).

## Qué está confirmado y qué falta

| Archivo candidato | Evidencia oficial revisada | Resultado documental | Pendiente del archivo real |
|---|---|---|---|
| SimCLR ResNet50 1x | README, `data_util.py`, `resnet.py`, licencia Google Research | SimCLR v1; encoder con cabeza lineal y sin proyección; RGB en [0,1]; repositorio Apache 2.0 | Obtener ZIP; alcance de licencia sobre el artefacto; tensores/shards/hash y semántica del encoder |
| MoCo v1 200 épocas | README, LICENSE y evaluación lineal en revisión histórica oficial fijada | URL exacta; normalización ImageNet; prefijo MD5 `b251726a` | Aclarar alcance del relicenciamiento sobre pesos; contrastar bytes, carga y selección del paper |
| MoCo v2 800 épocas | Mismas fuentes oficiales históricas | URL exacta; normalización ImageNet; prefijo MD5 `a04e12f8`; un archivo compartido por USL/RPI | Igual que MoCo v1; no confundirlo con la variante de 200 épocas |
| MoCo v3 R50 1000 épocas | README, CONFIG, LICENSE, evaluación lineal | Se selecciona pretrain, no linear-eval; normalización ImageNet; proyecto CC BY-NC 4.0 | Bytes/hash/estructura y alcance sobre artefacto; confirmar la elección como adaptación V8 |
| DINO ViT-S/16 | README, hubconf, eval_linear, vision_transformer, LICENSE | Backbone DINO original: 384 dimensiones, 12 bloques, 6 cabezas, patches16 a 224; ImageNet; repositorio Apache 2.0 | Bytes/hash/carga del backbone y alcance de licencia; no sustituir por DINOv2/v3 |

Las fuentes con revisión y SHA de blob están fijadas en el JSON. Los modelos
elegidos coinciden con las familias requeridas por V8, pero no se afirma que sean
los mismos archivos o presupuestos de preentrenamiento usados por los papers de
USL/RPI. Esa selección continúa siendo una adaptación explícita.

## MoCo: el enlace a main ya no conserva el código

La punta `8976944da9c6b94cbd9158d7ebe50912aef807ef` tiene árbol vacío; por eso no
se usó `main` como referencia de licencia o checksum. Se recuperaron README,
LICENSE y main_lincls desde el padre histórico oficial
`132aea9da9bb995a411c915c0420be3db62052f3`, sin usar mirrors de pesos.

Su README todavía declara CC BY-NC 4.0; LICENSE contiene MIT. El commit oficial
`7397dfe146c7ca6bbb58e9c382498069178ba764` documenta el relicenciamiento del código
a MIT, lo que explica la diferencia. La revisión no encontró una declaración
separada que resuelva explícitamente su alcance sobre los checkpoints publicados.
No se convierte ese cambio de licencia del código en una certificación de pesos.

MoCo v3 sí presenta README y LICENSE consistentes en CC BY-NC 4.0. DINO y SimCLR
declaran Apache 2.0 en sus repositorios. Estas son constataciones documentales;
las revisiones del plan siguen `approved=false`, no se fabrican aprobaciones por archivo.

## Checksums: prefijo MD5 y SHA-256 son evidencias diferentes

La tabla oficial de MoCo publica solo ocho dígitos hexadecimales del MD5, no los
32 del digest completo. El preparador ahora admite `published_md5_prefix`,
calcula el MD5 completo del archivo y exige coincidencia de ese prefijo antes
de promoverlo. También calcula el SHA-256 completo y conserva ambos valores.
Un prefijo que no coincide deja el archivo `.part` preservado, sin publicarlo.

El estado `matches_published_md5_prefix` distingue esta comprobación de
`matches_expected_sha256` y de `local_sha256_only`. La coincidencia de 32 bits
de MD5 es una comprobación limitada de identidad, no una garantía criptográfica
de autenticidad. No se rellenan los 24 dígitos faltantes ni se presenta el SHA-256
local como si lo hubiera publicado el autor. No se identificó un SHA-256 oficial
completo para estos candidatos en las fuentes revisadas.

## SimCLR: carga estricta no equivale a reproducción numérica

Se confirmó RGB en [0,1] en `preprocess_image`; no corresponde aplicar de forma
automática la normalización ImageNet usada por MoCo/DINO. El encoder local usa
`MaxPool2d(3, stride=2, padding=1)`, mientras la fuente TensorFlow utiliza SAME.
Para la salida de stem de 112 píxeles, SAME distribuye un píxel de padding total
como 0 antes y 1 después, frente al padding simétrico local. Ambos pueden producir
formas compatibles pese al desplazamiento espacial.

Esta es una diferencia estática de operaciones; no se ejecutó un encoder real.
Antes de afirmar equivalencia hay que decidir si se conserva como adaptación
documentada o se implementa y contrasta un camino fiel al origen. No se cambia
silenciosamente el encoder compartido de los cuatro métodos. El `last_stride=1`
de los ResNet locales también es una adaptación de geometría densa ya declarada,
frente al backbone original. Los tensores y su mapeo aún requieren el ZIP real.

## Estado de ejecución y evidencia nueva

No se accedió al clúster, descargaron checkpoints, instalaron paquetes ni se
ejecutaron modelos. El 403 de SimCLR es el último resultado confirmado desde el
clúster; no se repitió la consulta. El bloqueo de observación de Jupyter permanece.

Solo se comprobaron las tres rutas nuevas de prefijo MD5 con respuestas simuladas:
coincidencia, rechazo sin promoción y validación de la longitud. El informe está
en [publisher-checksum.json](../resultados/preparacion_v8/staging_20261013/publisher-checksum.json).
Las 14 pruebas previas del preparador y las suites históricas de segmentación no
se repitieron. El archivo de esas pruebas solo ajusta su fixture para que el
checksum del candidato real no se aplique a sus ocho bytes simulados.

El siguiente trabajo con archivos reales sigue siendo desplegar en un checkout
independiente, cerrar las revisiones por artefacto y descargar en el clúster.
Después se registran hashes, claves excluidas, normalización y carga estricta;
no se habilita entrenamiento por completar únicamente esa preparación.

Actualización posterior: el [documento 50](50-intento-transferencia-y-revision-preparacion-v8.md)
registra la decisión documental para preparación no comercial y el intento real
de transferencia. No sustituye los pendientes físicos o semánticos de esta revisión.

