# Preparación de lanzamiento V8

Estado: borradores deshabilitados. Estos archivos no son planes ejecutables del controlador.

## Avance independiente de ISIC2018

- `resources.draft.json`: techos operativos propuestos y fórmulas de almacenamiento por imagen/etapa. Los límites de 64 GiB RAM y 20 GiB VRAM son propuestas bajo la capacidad observada, no mediciones de consumo. Timeout, conteos y almacenamiento acumulado requieren cierre.
- `pipeline.draft.json`: dependencias por método, bancos reales, gates y recetas ya registradas. No incluye preentrenamiento SSL nuevo: se reutilizan los pesos públicos verificados.
- Paquete de snapshot completo preparado localmente: `v8-execution-dd0fc88.patch.zip`, 97.589 bytes, SHA-256 `dda8fa0ea1f3d9e5802f2a48438e2702e2dc141fb296893cb6ae2af71e44f965`. Contiene el delta de 52 archivos desde `d55c5cd` a `dd0fc88` y los blobs esperados. La transferencia al clúster NO está confirmada.

## Copia de ejecución pendiente

Crear una copia independiente desde el checkout base limpio, aplicar el delta fijado y contrastar cada archivo con el blob de GitHub. Registrar su identidad de snapshot y revisión local. Comparar la corrección CUDA RPI de la rama con la copia protegida antes de reutilizarla. No cambiar los checkouts protegidos ni tomar como identidad upstream un commit local diferente.

La carga por el selector de archivos quedó bloqueada por protección de datos de credenciales. La revisión automática rechazó reconectar la pestaña existente; la recuperación indicada en una pestaña nueva también falló con `native credential state cannot be safely resumed`. No se intentó transferir por una vía alternativa después del bloqueo.

## Condiciones que no deben ocultarse

1. CCAM usa gate `method=usl_ccam`, distinto del gate del estudiante USL; falta fijar su presupuesto de épocas. Su implementación actual guarda `ccam.pt` al final y no ofrece reanudación intermedia: considerar esto al aprobar tiempo e interrupciones.
2. Los cinco ciclos USL y la fase RPI 50+30 requieren entradas, salidas, inicializaciones y revisiones propias. El refresco P2 necesita un paso concreto; no se atribuye automáticamente al refinamiento genérico.
3. Los gates deben corresponder al SHA-256 del manifiesto real y a la revisión que se ejecutará. Las pruebas previas se reutilizan con su alcance original; no se atribuyen a código diferente sin evidencia.
4. La aceptación del candidato HF SimCLR y sus límites de equivalencia deben declararse. No afirmar paridad con Google TensorFlow.
5. W&B online usa el proyecto `dermssl-segmentacion`; comprobar conexión al lanzar. La entrega de alertas sigue sin confirmarse.

## Dependencias de ISIC2018

Al finalizar la descarga existente: revisar el recibo/checksums, el ZIP de imágenes y las 2.594 máscaras esperadas, establecer la reserva anatómica y excluir solapamientos paciente/lesión e imagen del entrenamiento. Solo entonces fijar manifiestos, hashes y gates. No incorporar máscaras expertas a objetivos de entrenamiento.

No se ejecutaron modelos, pruebas ni entrenamientos en esta preparación.
