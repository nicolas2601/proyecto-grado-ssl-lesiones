# Camino A: maestro DINOv2 destilado en la U-Net

Estado: pendiente, no iniciado.

Idea: DINOv2 (preentrenado sin etiquetas con datos externos) produce pseudo-máscaras por métodos espectrales, y la U-Net las aprende. Usa conocimiento externo, por eso no es autosupervisado "puro" sobre nuestros datos.

| Carpeta | Contenido previsto |
|---|---|
| `pseudo-mascaras/` | Maestro: DINOv2 ViT-S/14 a 448 px, grafo de afinidad entre parches y bipartición con el vector de Fiedler (TokenCut o Deep Spectral Methods). Se ejecuta una sola vez sobre todo el pool |
| `entrenamiento/` | Destilación en la U-Net (red de segmentación entrenada con pseudo-máscaras de `../comun/estudiante/`), ignorando en la pérdida los píxeles de confianza intermedia (0,3 a 0,7) |
| `resultados/` | Métricas livianas (JSON/CSV) y figuras pequeñas. Los artefactos pesados van en `salidas/` |

Pendientes:

- Pesos de DINOv2 desde Hugging Face (confirmar el identificador exacto del modelo ViT-S/14).
- Criterio para elegir la componente que es lesión y verificación visual por bucket de tono.
- Ningún paper revisado prueba un maestro DINOv2 espectral en dermoscopía: es una hipótesis a validar.
