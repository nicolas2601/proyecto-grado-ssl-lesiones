# Diagnóstico del estimador FST v1

Evidencia de que el `FST_bucket` del estimador previo al guard de b* no medía
tono de piel sino el signo de b*, y de que ese signo está fuertemente asociado
al diagnóstico.

## Contenido

| Archivo | Qué es |
|---|---|
| `diag_ita_signo.py` | Script: recalcula L*/b* de la piel y los cruza contra el bucket FST |
| `diag_ita_signo.txt` | Salida completa del anterior |
| `diag_confusion_adquisicion.py` | Script: cruza signo de b* y tercil de L* contra el diagnóstico |
| `diag_confusion_adquisicion.txt` | Salida completa del anterior |
| `histograma_ita.txt` | Histograma de ITA en HAM. Diagnóstico ad hoc, no versionado como script |

Ambos scripts son de solo lectura: no escriben nada ni modifican salidas.

## Hallazgo en una línea

Los buckets 1 y 6 difieren en 0,74 de L* (69,55 contra 68,81) pero en el signo
de b* difieren casi perfectamente (5728/5729 positivos contra 3429/3431
negativos). El bucket "fototipo VI" es piel de la misma claridad que el
"fototipo I", y concentra 2,26 veces la tasa de melanoma.

## Reproducibilidad — leer antes de reejecutar

Los dos scripts leen `salidas/final/fst/fst_pseudo_manifest.json`, que **el
rerun con el estimador corregido sobrescribe**. Ejecutarlos después del rerun
NO reproduce estas cifras: darían los números del estimador nuevo.

Para reproducir el diagnóstico hay que apuntarlos al respaldo del manifiesto v1:

    salidas/final/fst.v1_signo_b/fst_pseudo_manifest.json

Los demás insumos (`cache/images.npy`, `cache/field.npy`,
`pseudo_r2/pseudo_masks.npy`) no los toca el rerun, así que siguen sirviendo.

Las rutas dentro de los scripts son las del pod y se dejaron tal cual: estos
archivos van como evidencia de lo que se ejecutó, no como herramienta a
reejecutar.

## Entorno

Pod UNABIA `jupyter-nmoreno534`, 2026-09-21. Python del entorno base
(numpy 2.5.3, scipy 1.18.0, scikit-image 0.26.0). Estado del repo en el momento
del diagnóstico: `91814a9`, antes de aplicar `c272bdb` (guard de b*) y `bbd9bab`
(terciles de tono).
