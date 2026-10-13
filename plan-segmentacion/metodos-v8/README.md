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
