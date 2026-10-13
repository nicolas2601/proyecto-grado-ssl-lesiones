# SLED: adaptación Python

Fuentes: https://arxiv.org/abs/2309.01899 y código del autor fijado en
`SELGroup/SLED@3213eacb65577c0d52f424369158fb56c609238e` (revisión, no vendoring).

`entropy.py` minimiza H2 mediante merges que mejoran el objetivo y relocaciones
monótonas. `pipeline.py` implementa SLIC, afinidad autoajustada con distancia de
centroides, bisección por varianza, sklearn IsolationForest, mezcla multiescala y
GHT. El selector reutiliza el gaussiano corregido del documento 42.

Diferencias registradas: solver Python/orden de empates propios; centroides de
TODOS los superpíxeles corregidos; semilla explícita; IForest directo de sklearn,
no wrapper PyOD; 30 pases de refinamiento por defecto; huecos restringidos al
campo válido; resolución común 224. No se declara equivalencia con Java.
Controles: `centroid_mode=upstream_zero`, `selection_variant=upstream_area`.

Sin volúmenes, contraste o scores válidos se abstiene y conserva el fallo. No
sustituye fallos por otras imágenes. Todas las escalas son seriales.
Config de ejecución: `config.json`; estados y recursos: documento 43.
