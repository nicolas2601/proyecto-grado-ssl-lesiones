# SimCLR: descarga Hugging Face y validación local V8

Actualización de 2026-10-14. Esta sección sustituye las afirmaciones previas de pruebas espaciales pendientes o ausencia total de pesos SimCLR; el ZIP TensorFlow original sigue sin descargarse.

## Resultado comprobado

- Fuente alternativa: `lightly-ai/simclrv1-imagenet1k-resnet50-1x`, revisión `476674d2c9804bc2cd4abd8bde6121b6d6b19990`, archivo `resnet50-1x.pth`.
- Descarga completa de 102.542.786 bytes; SHA-256 coincide con el objeto LFS publicado: `fdd96bbcc92008232f31e0f7de46d30810e7115a05005d7cff926e78e6fa37ea`.
- Carga segura `weights_only=True`: 320 tensores; conversión estricta de 318 tensores del encoder. Se descartan únicamente `fc.weight` y `fc.bias`.
- Cinco pruebas nuevas: 5 aprobadas, 0 errores, 0 omitidas. No se repitieron pruebas anteriores.
- Comparación con referencia PyTorch explícitamente adaptada al perfil V8: error absoluto máximo 0 en las cuatro etapas para una entrada sintética 224×224. Formas: 56×56, 28×28, 14×14 y 14×14.
- La referencia publicada usa maxpool simétrico; SAME requiere padding adicional al final para una entrada par. El problema es de implementación espacial, no del hardware del clúster. La adaptación de stride final 2→1 está declarada para segmentación densa.
- Esto valida el mapeo PyTorch local, no equivalencia numérica con TensorFlow original ni identidad binaria con el ZIP de Google. Se conserva `upstream_numerical_equivalence_verified=false`.

## Fuentes y versiones

[Hugging Face](https://huggingface.co/lightly-ai/simclrv1-imagenet1k-resnet50-1x) publica una conversión PyTorch de terceros; [tonylins/simclr-converter](https://github.com/tonylins/simclr-converter) documenta su origen. Referencia revisada: `139d3cb0bd0c64b5ad32aab810e0bd0a0dddaae0`.

[sthalles/SimCLR](https://github.com/sthalles/SimCLR) ofrece una implementación independiente y checkpoints ResNet50 entrenados en STL10 durante 50 épocas. No son sustitutos identificados del ResNet50 ImageNet seleccionado.

SimCLRv1 introduce aprendizaje contrastivo de dos vistas. SimCLRv2 amplía redes y proyector, añade encoder de momento y negativos almacenados; su procedimiento completo incluye preentrenamiento, ajuste con etiquetas y destilación. `tf2` significa TensorFlow 2 y también contiene exportaciones de SimCLRv1; `1x` significa anchura del backbone.

La figura 6 adjunta compara clasificación ImageNet con 1% y 10% de etiquetas, según tamaño del modelo y ajuste/destilación. No demuestra resultados de segmentación de lesiones sin etiquetas.

## Pendiente en el clúster

No se confirmó transferencia del checkpoint HF ni despliegue del adaptador nuevo. La revisión automática rechazó la carga del parche de código por exigir autorización explícita de despliegue; después el navegador volvió a fallar con `native credential state cannot be safely resumed`. No se intentó eludir ese bloqueo.

Las dos copias `d55c5cd` y sus cambios locales se conservaron. No se inició entrenamiento. Continúan pendientes agrupación paciente/lesión, evaluación independiente, presupuestos completos, planes del controlador y manifiesto congelado de pseudo-máscaras SLED/TriDerm. `launch_ready=false`.

Las evidencias reproducibles están en `resultados/preparacion_v8/readiness_20261014/`. El script de comparación requiere los pesos y la referencia fijada; no incluye los binarios en Git.
