# =====================================================================
# EXPORTAR EL MODELO SSL DE VERDAD (init=ssl)  —  proyecto grado UNAB
# =====================================================================
# Objetivo: guardar el brazo SSL fine-tuneado como best_resnet50_sslinit.pth,
# para SERVIR el modelo SSL (no el baseline ImageNet).
#
# NO repite el pretrain SSL de 6 h. Solo hace el fine-tune del brazo SSL
# (~15-25 min en T4) partiendo del backbone domain-adaptive ya entrenado.
#
# ---------------------------------------------------------------------
# CÓMO CORRERLO EN KAGGLE (para ahorrar las 6 h):
#   1. Subí `backbone_domain_adaptive_r50.pth` como INPUT del notebook
#      (Add Input -> Upload), o dejalo en /kaggle/working si la sesión sigue viva.
#   2. Si lo subiste como input, ANTES de correr esta celda seteá la ruta:
#         DA_BACKBONE = "/kaggle/input/<tu-dataset>/backbone_domain_adaptive_r50.pth"
#   3. Corré las celdas de imports/datos/definiciones: 0, 1, 2, 5
#      (NO corras la celda 4 = pretrain SSL de 6 h; la saltamos).
#   4. Corré ESTA celda. Al final descargá best_resnet50_sslinit.pth desde Output.
#
# Si la sesión del run completo sigue viva (results en memoria), reusa el brazo
# SSL sin re-entrenar nada.
# =====================================================================
import numpy as np, torch, torch.nn.functional as F
from torchvision.models import resnet50
from sklearn.metrics import roc_curve

# 1) Obtener el brazo SSL: reusar si ya está en memoria; si no, fine-tunearlo
if "results" in globals() and "ssl" in results:
    f1_ssl, state_ssl, lg_ssl, gt_ssl = results["ssl"]
    print("Reusando el brazo SSL que ya está en memoria (no re-entreno).")
else:
    print("Entrenando SOLO el brazo SSL desde el backbone domain-adaptive...")
    f1_ssl, state_ssl, lg_ssl, gt_ssl = train_arm("ssl", tag="SSL-EXPORT")

# 2) Calibración propia del SSL (temperature scaling sobre SUS logits)
def _fit_T(logits, labels):
    lg = torch.tensor(logits); lb = torch.tensor(labels)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / T.clamp(min=1e-2), lb)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1e-2))

T_ssl = _fit_T(lg_ssl, gt_ssl)

# 3) Umbral de screening de melanoma (sensibilidad >= 0.80) con probs calibradas del SSL
probs_ssl = F.softmax(torch.tensor(lg_ssl) / T_ssl, dim=1).numpy()
mi = classes.index("mel")
fpr, tpr, thr = roc_curve((gt_ssl == mi).astype(int), probs_ssl[:, mi])
ok = np.where(tpr >= 0.80)[0]
thr_ssl = float(thr[ok[0]]) if len(ok) else None

# 4) Guardar con los mismos metadatos que el otro checkpoint, pero init=ssl
OUT_SSL = "/kaggle/working/best_resnet50_sslinit.pth"
torch.save({
    "model_state_dict": state_ssl,
    "classes": classes,
    "arch": "resnet50",
    "img_size": SUP_RES,
    "init": "ssl",
    "macro_f1": float(f1_ssl),
    "temperature": float(T_ssl),
    "mel_screening_threshold": thr_ssl,
    "norm_mean": MEAN, "norm_std": STD,
}, OUT_SSL)
print(f"OK -> {OUT_SSL}")
print(f"   init=ssl | macro_f1={f1_ssl:.4f} | T={T_ssl:.4f} | thr_mel={thr_ssl}")
print("Descargá best_resnet50_sslinit.pth desde Output para servir el modelo SSL de verdad.")
