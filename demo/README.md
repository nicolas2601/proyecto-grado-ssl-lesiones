# Demo web — Clasificador de lesiones cutáneas

Interfaz Gradio para probar el modelo: arrastrás una imagen dermatoscópica y ves las 7 clases con su
probabilidad (calibrada) + la alerta de melanoma. **Herramienta de apoyo, no de diagnóstico.**

## Requisitos

1. Descargar el checkpoint del [Release `modelo-v1-preliminar`](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/modelo-v1-preliminar)
   → `best_resnet50_ssl.pth`. Por defecto se busca en `~/Descargas/`.
2. Entorno con `torch`, `torchvision`, `gradio`, `pillow` (CPU basta).

```bash
python3 -m venv .venv-demo
.venv-demo/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv-demo/bin/pip install gradio pillow
```

## Correr

```bash
.venv-demo/bin/python demo/app.py
# abrir http://127.0.0.1:7860

# con otra ruta de modelo:
MODEL_PATH=/ruta/best_resnet50_ssl.pth .venv-demo/bin/python demo/app.py
```

## Notas

- El checkpoint trae embebidos: `classes`, `temperature`, `mel_screening_threshold`, `norm_mean/std`,
  `img_size`. El demo los usa directo (no hay que configurar nada).
- Corre en CPU; la primera predicción tarda ~1–2 s.
- Es un modelo **preliminar** (recursos de estudiante); sin validación externa.
