# Revisión local de CCAM y divergencia con GitHub

Fecha: 2026-10-17. Alcance: revisión estática; sin pruebas, despliegue ni entrenamiento.

## Estado Git después de fetch

- Rama: `experiment/modular-audit-v8`.
- HEAD local: `4a50725db635aa72c44ff77684a97c39db5acbd9`.
- Remoto observado: `012110fd2394ee652183f025b3a38d9515578681`.
- Base común: `f2d0885d8f55a3cc66249056e1900c86cbfc81a9`.
- Divergencia: un commit exclusivo local y 25 exclusivos remotos.
- Diez de los 16 archivos cambiados en remoto coinciden con los archivos locales
  al comparar blobs. Las diferencias restantes no muestran cambios semánticos:
  corresponden a terminaciones de línea en documentación y cuatro configs draft.
- Varios archivos presentes en remoto aparecen localmente como no versionados
  porque la historia local todavía no contiene los commits que los incorporaron.
- Se ejecutó únicamente `git fetch origin` para actualizar referencias. No se
  integraron ramas, borraron archivos, reescribieron commits ni hizo push.

## Qué incorpora el commit local

`src/dermssl/segmentation_methods/usl_net/seeds.py` guarda por época cabeza,
optimizador, historial, saliencia previa y RNG; restaura ese estado al reanudar.
La utilidad común escribe un temporal y aplica `os.replace`. El CLI admite
`--resume` para CCAM. Esta implementación supera la limitación del código anterior
que guardaba únicamente la cabeza final.

La estabilidad registrada es la correlación Pearson global entre saliencias
aplanadas de ocho imágenes fijas de aprendizaje en épocas consecutivas. La
primera época no tiene comparación. Es una medición del proceso, no prueba de
calidad anatómica ni de igualdad absoluta de las saliencias.

El techo recomendado de 30 épocas se documenta como precedente MoCo no medido
para CCAM. No se presenta como presupuesto nativo ni convergencia acreditada.

## Pendientes identificados

1. La identidad del checkpoint CCAM no incorpora `source_hashes`. Con despliegues
   por ZIP y HEAD constante, cambiar código puede no invalidar la recuperación.
   Referencia: `seeds.py:55` y contrato común en `common/training.py:180`.
2. JSONL/W&B se registran antes de sustituir el checkpoint. Una interrupción entre
   ambas operaciones puede duplicar los registros de una época al recuperarse.
   La escritura atómica del checkpoint no vuelve transaccionales los registros.
   Referencia: `seeds.py:88`.
3. El techo recomendado y selector de estabilidad son utilidades; el ciclo sigue
   usando `config["epochs"]`. El último `resume.pt` se sobrescribe: seleccionar
   retrospectivamente una época anterior requiere conservar sus pesos.
4. La revisión estática y los tests añadidos con encoders de juguete no acreditan
   despliegue ni ejecución GPU del cambio. No se ejecutaron tests en esta revisión.

## Sincronización autorizada posterior a la revisión

El usuario autorizó sincronizar la rama. Se integró el remoto mediante el commit
de merge `9589eb8bfa2608735df36e70487ffea85e6a716e`, sin conflictos y sin
reescribir el commit local `4a50725`. Ambos historiales quedan conservados.
Las copias locales de las 16 rutas entrantes se preservaron previamente en el
stash `61cfb58c4427f021d4f45242b41817324869b331`, con mensaje
`backup(v8): preserve local copies before authorized branch synchronization`.
No se reaplica ese stash: las rutas correspondientes ya están integradas y las
diferencias detectadas eran de terminación de línea. El respaldo se conserva.
Los otros archivos sin seguimiento no se incluyeron en la sincronización.

## Siguiente acción

Resolver conjuntamente identidad de recuperación, registros y política de
selección/conservación; concretar la configuración de la siguiente ejecución y
documentar el techo de 30 épocas como referencia de planificación no medida
para CCAM. Después preparar un ZIP de los cambios para UNABIA y su revisión
previa al lanzamiento. No repetir las comprobaciones operativas cerradas.
Este informe no habilita configuraciones ni altera gates o manifiestos. La
sincronización de GitHub no despliega los cambios al clúster.
