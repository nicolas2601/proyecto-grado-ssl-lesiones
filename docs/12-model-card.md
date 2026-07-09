# 12 — Model Card

Ficha técnica del modelo entrenado, siguiendo el formato *Model Cards* (Mitchell et al., 2019).

## Detalles del modelo

| Campo | Valor |
|---|---|
| Nombre | `best_resnet50_ssl.pth` |
| Arquitectura | ResNet-50 (torchvision) + cabeza lineal de 7 clases |
| Inicialización (modelo entregable) | **SSL domain-adaptive** — pipeline corregido para exportar el SSL. ⚠️ El artefacto del Release actual es aún el brazo ImageNet de la corrida 1; se reemplaza por el SSL en la próxima corrida |
| Backbone SSL base | `backbone_domain_adaptive_r50.pth` (SimSiam domain-adaptive, HAM+ISIC) |
| Ablación (referencia) | empate técnico con ImageNet en macro-F1 (Δ 0.0007, ruido); SSL gana en AUC |
| Metadatos embebidos | `classes`, `temperature`, `mel_screening_threshold`, `norm_mean/std`, `img_size`, `init`, `arch`, `macro_f1` |
| Tamaño aprox. | ~90–100 MB (`state_dict` de ResNet-50) |
| Framework | PyTorch |
| Resolución de entrada | 224 × 224 RGB |
| Calibración incluida | Temperature scaling, T = 0.6757 (guardada en el checkpoint) |
| Umbral screening melanoma | 0.0541 (guardado en el checkpoint) |
| Autores | Nicolás Santiago Moreno Monroy · María Paula Saavedra |
| Fecha | 2026-07-07 |
| **Descarga** | [Release `modelo-v1-preliminar`](https://github.com/nicolas2601/proyecto-grado-ssl-lesiones/releases/tag/modelo-v1-preliminar) |

> Nota (verificada leyendo el checkpoint): el `.pth` del **Release actual** (corrida 1) tiene
> metadato `init = imagenet` — el notebook había exportado el ganador de la ablación (ImageNet, por
> 0.0007), no el SSL. **Esto ya se corrigió en el pipeline:** el notebook ahora fija `best_name="ssl"`
> y evalúa + exporta el **modelo SSL** como entregable (ver [D-09](04-decisiones-tecnicas.md)). La
> próxima corrida (reset de cuota GPU de Kaggle) reemplaza el artefacto por el SSL con sus propias
> métricas clínicas.
>
> ⚠️ **Naturaleza preliminar:** estos pesos son un **resultado de prueba con recursos limitados**
> (GPU T4 de estudiante, dataset pequeño), no el modelo definitivo. Representan el techo alcanzable
> con el cómputo actual; el entrenamiento definitivo se hará con las GPUs de la UNAB.

## Uso previsto

- **Uso previsto:** herramienta de **apoyo** al tamizaje de lesiones cutáneas sobre imágenes
  dermatoscópicas. Prioriza sensibilidad al melanoma.
- **Usuarios previstos:** investigación académica y personal clínico como segunda opinión.
- **Fuera de alcance:** **NO** es un dispositivo de diagnóstico. No sustituye a un dermatólogo. No
  validado clínicamente ni certificado por ninguna autoridad sanitaria.

## Datos de entrenamiento

- HAM10000 (Tschandl et al., 2018), 7 clases. Split por `lesion_id` (anti-leakage).
- Pretrain SSL (domain-adaptive) sobre pool HAM-train + ISIC 2019 (~33 k imágenes, sin etiquetas).

## Métricas de evaluación (val, split honesto)

| Métrica | Valor |
|---|---|
| macro-F1 | 0.7454 [0.7005, 0.7807] |
| Accuracy | 0.8659 |
| AUC-ROC macro | 0.9343 |
| ECE calibrado | 0.0598 |
| Sensibilidad melanoma (screening) | 0.8018 |
| Especificidad melanoma (screening) | 0.8873 |

Detalle completo en [11 — Resultados finales](11-resultados-finales.md).

## Consideraciones éticas y limitaciones

- **Sesgo por fototipo:** entrenado en piel clara (Fitzpatrick I–III). Rendimiento en piel oscura
  **no garantizado** (la literatura reporta caídas de sensibilidad de 0.69 → 0.23).
- **Sin validación externa** aún → generalización cross-dataset no confirmada.
- **`df` (dermatofibroma)** con recall 0.41: clase poco fiable.
- Debe usarse **siempre** con el mecanismo de abstención activo (derivar casos dudosos).

## Cómo servir el modelo (inferencia)

El notebook expone `predict_path(path)` (celda 21). Esqueleto mínimo para servirlo fuera de Kaggle:

```python
import torch, json
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL_IDX = CLASSES.index("mel")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Parametros guardados junto al checkpoint (ver RESULTS_SUMMARY.json) ---
TEMPERATURE = 0.6757        # temperature scaling (calibracion)
MEL_THRESHOLD = 0.0541      # umbral de screening de melanoma

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_model(weights_path="best_resnet50_ssl.pth"):
    net = models.resnet50(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, len(CLASSES))
    state = torch.load(weights_path, map_location=DEVICE)
    # el checkpoint puede ser el state_dict directo o venir envuelto en un dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.eval().to(DEVICE)
    return net

def predict_path(path, net):
    img = Image.open(path).convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = net(x) / TEMPERATURE          # calibracion
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(probs.argmax())
    mel_prob = float(probs[MEL_IDX])
    return {
        "pred": CLASSES[pred_idx],
        "conf": float(probs[pred_idx]),
        "mel_prob": round(mel_prob, 4),
        "mel_alert": mel_prob >= MEL_THRESHOLD,
        "probs": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }

if __name__ == "__main__":
    model = load_model()
    print(json.dumps(predict_path("ejemplo.jpg", model), ensure_ascii=False, indent=2))
```

### Seguridad al servir como API (si se expone por HTTP)

Si se envuelve este `predict_path` en una API (FastAPI/Flask) para la demo:

- **Validar la entrada:** aceptar solo imágenes (`Content-Type` image/*), limitar tamaño (p. ej. ≤ 10 MB)
  y verificar que Pillow puede abrirla antes de procesar. Rechazar todo lo demás con 400.
- **Sanitizar rutas:** nunca pasar rutas del usuario directo al sistema de archivos (path traversal);
  guardar el upload en un directorio temporal con nombre generado, no con el nombre del cliente.
- **Rate limiting:** limitar peticiones por IP para que la inferencia (costosa en GPU) no sea un
  vector de DoS.
- **No filtrar detalles internos** en los mensajes de error (nada de stack traces al cliente).
- **HTTPS** obligatorio si sale de localhost; es dato sensible (imagen médica).
- **Aviso legal en la respuesta:** incluir siempre que es apoyo, no diagnóstico.
