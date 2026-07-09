#!/usr/bin/env python3
"""Demo web de clasificación de lesiones cutáneas (proyecto de grado UNAB).

Carga el checkpoint entrenado, aplica la calibración (temperature scaling) y el
umbral de screening de melanoma que vienen guardados dentro del propio .pth, y
expone una interfaz Gradio: arrastrás una imagen dermatoscópica y ves las 7 clases
con su probabilidad + la alerta de melanoma.

Uso:
    .venv-demo/bin/python demo/app.py
    # o con otra ruta de modelo:
    MODEL_PATH=/ruta/best_resnet50_ssl.pth .venv-demo/bin/python demo/app.py

⚠️ Herramienta de APOYO, no de diagnóstico. Resultado preliminar (recursos limitados).
"""
import os
import torch
import torch.nn.functional as F
import gradio as gr
from torchvision import models, transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.environ.get(
    "MODEL_PATH", os.path.expanduser("~/Descargas/best_resnet50_ssl.pth")
)

# Nombres legibles para mostrar
NOMBRE_LARGO = {
    "akiec": "Queratosis actínica / Bowen (akiec)",
    "bcc": "Carcinoma basocelular (bcc)",
    "bkl": "Queratosis benigna (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "MELANOMA (mel)",
    "nv": "Nevus / lunar benigno (nv)",
    "vasc": "Lesión vascular (vasc)",
}

# ---------------------------------------------------------------------------
# Carga del modelo + metadatos embebidos en el checkpoint
# ---------------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No encontré el modelo en {MODEL_PATH}.\n"
        "Descargalo del Release 'modelo-v1-preliminar' o poné MODEL_PATH."
    )

_ck = torch.load(MODEL_PATH, map_location=DEVICE)
CLASSES = _ck["classes"]
MEL_IDX = CLASSES.index("mel")
IMG_SIZE = _ck.get("img_size", 224)
TEMPERATURE = float(_ck.get("temperature", 1.0))
MEL_THRESHOLD = float(_ck.get("mel_screening_threshold", 0.5))
NORM_MEAN = _ck.get("norm_mean", [0.485, 0.456, 0.406])
NORM_STD = _ck.get("norm_std", [0.229, 0.224, 0.225])

_net = models.resnet50(weights=None)
_net.fc = torch.nn.Linear(_net.fc.in_features, len(CLASSES))
_net.load_state_dict(_ck["model_state_dict"])
_net.eval().to(DEVICE)

_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


@torch.no_grad()
def predecir(pil_img):
    if pil_img is None:
        return {}, "Subí una imagen dermatoscópica para empezar."
    x = _tf(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = _net(x) / TEMPERATURE          # probabilidades calibradas
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    # dict {nombre_largo: prob} para gr.Label
    etiquetas = {NOMBRE_LARGO[c]: float(p) for c, p in zip(CLASSES, probs)}

    pred_idx = int(probs.argmax())
    pred = CLASSES[pred_idx]
    conf = float(probs[pred_idx])
    mel_prob = float(probs[MEL_IDX])
    alerta = mel_prob >= MEL_THRESHOLD

    lineas = [
        f"### Predicción: **{NOMBRE_LARGO[pred]}**",
        f"- Confianza: **{conf:.1%}**",
        f"- Probabilidad de melanoma: **{mel_prob:.1%}** "
        f"(umbral de screening {MEL_THRESHOLD:.1%})",
    ]
    if alerta:
        lineas.append("\n## ⚠️ ALERTA DE MELANOMA — derivar a dermatología")
    else:
        lineas.append("\n✅ Sin alerta de melanoma por el umbral de tamizaje.")

    # abstención: si ninguna clase supera 0.5, sugerir especialista
    if conf < 0.5:
        lineas.append(
            "\n🤔 Confianza baja (< 50 %): el modelo **duda** → derivar a especialista."
        )
    return etiquetas, "\n".join(lineas)


_INIT = _ck.get("init", "?")
DESCRIPCION = f"""
Modelo **ResNet-50** ({_ck.get('arch','resnet50')}, init `{_INIT}`, macro-F1
{_ck.get('macro_f1', 0):.3f}) del proyecto SSL de lesiones cutáneas, entrenado sobre
HAM10000. Aplica calibración (T = {TEMPERATURE:.3f}) y umbral de screening de melanoma
({MEL_THRESHOLD:.1%}), ambos guardados en el checkpoint.

> ⚠️ **Herramienta de APOYO, no de diagnóstico.** Resultado **preliminar** (recursos
> limitados). Entrenado en piel clara (Fitzpatrick I–III); sin validación externa aún.
> No sustituye a un dermatólogo.
"""

with gr.Blocks(title="Clasificador de lesiones cutáneas — UNAB") as demo:
    gr.Markdown("# 🔬 Clasificador de lesiones cutáneas (SSL) — Proyecto de grado UNAB")
    gr.Markdown(DESCRIPCION)
    with gr.Row():
        with gr.Column():
            entrada = gr.Image(type="pil", label="Imagen dermatoscópica")
            boton = gr.Button("Analizar", variant="primary")
        with gr.Column():
            salida_probs = gr.Label(
                num_top_classes=7, label="Probabilidades por clase (calibradas)"
            )
            salida_txt = gr.Markdown()
    boton.click(predecir, inputs=entrada, outputs=[salida_probs, salida_txt])
    entrada.change(predecir, inputs=entrada, outputs=[salida_probs, salida_txt])

if __name__ == "__main__":
    print(f"Modelo: {MODEL_PATH} | clases: {CLASSES} | device: {DEVICE}")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
