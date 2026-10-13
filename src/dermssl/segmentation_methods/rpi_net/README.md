# RPI-Net: adaptación propia de cuatro fuentes

Fuente: PDF completo aportado, DOI 10.1016/j.bspc.2025.108433.

`seeds.py` integra MoCov3 conv1 K-means, MoCov3 CAM, MoCov2 conv1 K-means y DINO
final. La orientación por mayoría en borde y el target CAM de energía SSL son
DECISIONES de adaptación: el paper no especifica ese target ni polaridad.
P2 se conserva como saliencia suave. No se sustituye DINO por DINOv2.

`model.py`: queries imagen, keys/values pseudo en IPC; PIA entre fuentes distintas
con residual; sigma por fuente y mu desde rama imagen. ResNet50 en imagen y
pseudomáscaras; encoder pseudo compartido entre fuentes. Token grid14 y proyección
multiescala a256 son parámetros declarados para evitar atención224² cuadrática.
Inferencia final permite sólo RGB; no requiere nuevas pseudomáscaras.

`loss.py`: ecuaciones20-23. Interpretación sigma como log-varianza, límites[-8,12],
alpha separado por muestra y `detach_alpha=true` son convenciones explícitas.
El "entropy" del artículo se calcula como softplus(sigma)^2, no entropía binaria.
Se registra clipping. `common/refinement.py` actualiza P2 y mantiene P1/P3/P4:
esto es una adaptación conservadora, NO la réplica de una regla no especificada.

Preentrenamiento local MoCov3/DINO tiene arquitectura/receta propia documentada;
para pesos externos se exige conversión revisada al formato propio y procedencia.
Budget nativo inicial50, fases30, número no cerrado; aún sin ejecutar.
