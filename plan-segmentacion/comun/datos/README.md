# Datos

Estado: pendiente.

Alcance previsto:

- Inventario y splits de HAM10000, ISIC 2019 e ISIC 2020 (~68k imágenes, sin máscaras ni etiquetas).
- Redimensionado único a 224-256 px y caché en disco, para no decodificar JPEG grandes en cada época. En el clúster `/dev/shm` es de 64 MB y obliga a `workers=0`.
- Registro de los test sets: ISIC 2018 Task 1, PH2 y Dermofit (confirmar el tipo de imagen de Dermofit).

Candidatos del repositorio a revisar antes de reutilizar: `src/dermssl/data_resolver.py`, `src/dermssl/pool_manifest.py` y `src/dermssl/mask_manifest.py`.

Los datos y la caché pesada van en `salidas/`, que está en el `.gitignore` de esta carpeta.
