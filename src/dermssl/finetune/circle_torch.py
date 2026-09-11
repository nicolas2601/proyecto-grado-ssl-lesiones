"""Regularizacion conjunta de Stage 2: un termino MSE entre la prediccion
del probe entrenable sobre la imagen original y su contraparte sintetica con
shift de fototipo (dermssl.circle.shift_skin_tone), aplicado solo a la
cabeza del probe. El backbone queda congelado durante todo Stage 2 - la
misma promesa de "ningun gradiente toca el backbone" que ya usan los probes
de v5, llevada al entrenamiento de Stage 2.

Esto es la maquinaria GPU de Stage 2, construida antes de la corrida real de
Stage 1 en Kaggle (todavia pendiente, ver docs/20). Nada aca mide ni afirma
una mejora de equidad real: son las primitivas de entrenamiento (contrato de
congelado, optimizador solo del probe, loss MSE conjunta, un paso de
entrenamiento) que se conectan a una corrida real de Kaggle una vez exista
el fst_manifest.csv y el gate de equidad de Stage 1 de FAIL.

Solo GPU: importa torch a nivel de modulo, misma convencion que el resto de
archivos torch_*.py del proyecto. No puede correr en este entorno (sin GPU,
sin torch instalado) - se verifica solo con py_compile + ast
(tests/test_circle_torch_static.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BackboneNotFrozenError(RuntimeError):
    """Se lanza cuando stage2_train_step detecta un parametro del backbone
    con requires_grad=True. Excepcion explicita en vez de un assert pelado,
    para que el chequeo sobreviva a python -O / PYTHONOPTIMIZE=1 y el
    llamador pueda capturarla especificamente."""


def freeze_backbone(backbone: nn.Module) -> None:
    """Pone requires_grad=False en todos los parametros del backbone, en el
    lugar, y lo deja en modo eval.

    Se llama una vez antes de arrancar el entrenamiento de Stage 2.
    stage2_train_step tambien verifica este invariante en cada paso: defensa
    en profundidad, para que un llamador que se olvida de congelar, o que
    reactiva un parametro por error, falle fuerte en vez de actualizar el
    backbone en silencio.
    """
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()


class ProbeHead(nn.Module):
    """La unica parte entrenable de Stage 2: una capa lineal sobre las
    features del backbone congelado, igual que el probe logreg de v5 pero
    entrenado con descenso de gradiente en vez de scikit-learn, para que el
    termino MSE conjunto pueda fluir hacia el."""

    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def build_probe_optimizer(probe: nn.Module, lr: float = 1e-3) -> torch.optim.Optimizer:
    """Optimizador que solo ve al probe, nunca al backbone.

    Funcion separada (en vez de escribir torch.optim.AdamW(probe.parameters())
    en cada lugar) para que el invariante "los parametros del backbone nunca
    llegan a un optimizador" tenga un solo lugar para auditar.
    """
    return torch.optim.AdamW(probe.parameters(), lr=lr)


def joint_regularization_loss(
    probe: nn.Module,
    features_original: torch.Tensor,
    features_shifted: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    *,
    reg_weight: float = 1.0,
) -> torch.Tensor:
    """Loss de la tarea sobre las features originales, mas un termino MSE de
    regularizacion entre la salida del probe sobre las features originales y
    las que tienen shift de fototipo.

    features_original/features_shifted son salidas del backbone congelado
    para el mismo batch; la mitad con shift se genera con
    dermssl.circle.shift_skin_tone (numpy, CPU) antes del forward del
    backbone. Nunca reemplaza al loss de la tarea, se le suma.
    """
    logits_original = probe(features_original)
    logits_shifted = probe(features_shifted)
    task_loss = criterion(logits_original, targets)
    regularization = F.mse_loss(logits_original, logits_shifted)
    return task_loss + reg_weight * regularization


def stage2_train_step(
    backbone: nn.Module,
    probe: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_original: torch.Tensor,
    batch_shifted: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    *,
    reg_weight: float = 1.0,
) -> dict:
    """Un paso de entrenamiento de Stage 2.

    Verifica el invariante de backbone congelado ANTES de tocar el
    optimizador. Las features del backbone se calculan bajo torch.no_grad()
    tanto para el batch original como el desplazado, asi el backbone no
    aporta ningun grafo de autograd - no solo "sin gradiente por
    requires_grad=False", sino estructuralmente inalcanzable por
    loss.backward().

    Lanza BackboneNotFrozenError (nunca un assert pelado, que desaparece con
    python -O / PYTHONOPTIMIZE=1) si algun parametro del backbone todavia
    tiene requires_grad=True: un llamador que se olvido de llamar a
    freeze_backbone, o que reactivo un parametro por error, falla fuerte en
    vez de actualizar el backbone en silencio.
    """
    if any(p.requires_grad for p in backbone.parameters()):
        raise BackboneNotFrozenError(
            "Invariante de Stage 2 violado: el backbone debe quedar congelado"
        )
    backbone.eval()
    with torch.no_grad():
        features_original = backbone(batch_original)
        features_shifted = backbone(batch_shifted)

    optimizer.zero_grad(set_to_none=True)
    loss = joint_regularization_loss(
        probe, features_original, features_shifted, targets, criterion, reg_weight=reg_weight
    )
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item())}
