# Cuatro métodos v8

La implementación está en [segmentation_methods](../../src/dermssl/segmentation_methods/README.md).
Cada método reúne código, `config.json` y `README.md` en su carpeta:

- [SLED](../../src/dermssl/segmentation_methods/sled/README.md)
- [USL-Net](../../src/dermssl/segmentation_methods/usl_net/README.md)
- [RPI-Net](../../src/dermssl/segmentation_methods/rpi_net/README.md)
- [TriDermNet-HAGO](../../src/dermssl/segmentation_methods/tridermnet_hago/README.md)

[Documento 43](../../docs/43-cuatro-metodos-segmentacion-v8.md): decisiones, estados,
manifiestos, recursos y comparación. Las configuraciones no autorizan ejecución
real automáticamente. El launcher es `scripts/run_segmentation_v8.py`.

Preparación de pesos públicos: [inventario de candidatos](public-weights.plan.json)
y [guía reanudable](../../docs/48-preparacion-reanudable-pesos-v8.md).
`stage_public_ssl_weights_v8.py` prepara archivos con revisión explícita; no crea
gates ni activa las configuraciones científicas.

La [evidencia documental fijada](public-weights.evidence.json) y la [revisión de
pesos reales](../../docs/49-revision-fuentes-pesos-reales-v8.md) distinguen
checksum publicado, licencia de código y validación pendiente de cada archivo.

[Plan revisado para preparación académica](public-weights.reviewed-research.plan.json)
y [estado de transferencia al clúster](../../docs/50-intento-transferencia-y-revision-preparacion-v8.md).
La revisión documental no certifica archivos aún no descargados.


[Revisión actual y puertas de lanzamiento](../../docs/51-cierre-preparacion-y-bloqueos-lanzamiento-v8.md): cuatro pesos convertidos y banco candidato RPI cargado; SimCLR y las puertas científicas permanecen pendientes. El documento nuevo actualiza los estados históricos.

Actualización: [SimCLR Hugging Face y validación local](../../docs/52-simclr-huggingface-validacion-local-v8.md).
